import sys
import numpy as np

sys.path.append("../")

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.log_likelihood import GaussianProcessLogMarginalLikelihood

from assembleToOrder.assembleToOrder import AssembleToOrderPES
from assembleToOrderExtended.assembleToOrderExtended import AssembleToOrderExtended
from multifidelity_KG.obj_functions import RosenbrockNoiseFreePES, RosenbrockNewNoiseFreePES
from load_and_store_data import load_data_from_a_min_problem, obtainHistoricalDataForEGO, \
    create_listPrevData, createHistoricalDataForPES, match_pickle_filename
from pes.covariance import ProductKernel
from pes.model import scale_forward
from pes.slice_sampler import SliceSampler
import assembleToOrderExtended.sql_util_miso as sql_util_miso

__author__ = 'jialeiwang'

""" For this algorithm, Predicted Entropy Search from the paper Multi-task bayesian optimization, we agree that all IS
are accessible. Unlike misoKG, which assumes IS0 is hidden, here IS0 is no different than the other IS.
"""
"""
IS 0 = truth
IS 1 = biased
IS 2 = unbiased but noisy
"""

n_burnin = 5
n_samples = 10
##############################################
###
obj_func_min = AssembleToOrderPES(mult=-1.0)
# obj_func_min = AssembleToOrderExtended(mult=-1.0)
# obj_func_min = RosenbrockNoiseFreePES(mult=1.0)
# obj_func_min = RosenbrockNewNoiseFreePES(mult=1.0)

func_name = obj_func_min.getFuncName()
# how many prev datasets should be incorporated? This determines the dim of the GP and the num of hypers
list_previous_datasets_to_load = []
indexFirstIS = 0  # id of the first IS, used to label data accordingly
separateIS0 = 0  # 1 if IS0 is not observable and hence the data supplied for IS1, IS2, ...
### in coldstart this should be 0

### Load the training data
pathToPickles = "/fs/europa/g_pf/pickles/miso"
init_data_pickle_filename_prefix = func_name + '_IS_0_1_2_3_200_points_each_repl_'  # the dataset for IS 0
replication_no = 0  # random.randint(0,2) # There are 3 pickle sets with 200 points. Pick one at random
# complete_list_prev_datasets_to_load = ["{0}_repl_{1}".format(dataset_name, replication_no) for dataset_name in list_previous_datasets_to_load]
listPrevData = []  # create_listPrevData(obj_func_min, complete_list_prev_datasets_to_load, replication_no, pathToPickles, init_data_pickle_filename_prefix)
init_data_pickle_filename = init_data_pickle_filename_prefix + str(replication_no)
init_pts, init_vals = load_data_from_a_min_problem(pathToPickles, init_data_pickle_filename)
# init_pts and init_vals contain a list of lists of pts/vals, one dataset for each IS

### Load in correct ordering to compute hypers
###########
### Load truth IS as IS 0
############
# select_dataset = 2
# corresponding_IS = 0
# noise_vars = numpy.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
# listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))
# select_dataset = 1
# corresponding_IS = 2
# noise_vars = numpy.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
# listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))
# select_dataset = 0
# corresponding_IS = 1
# noise_vars = numpy.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
# listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))
#############
### load them one after another
index_dataset = 0 # position of the dataset in the lists (init_pts and init_vals) that corresponds to this IS -- usually in the same ordering
for indexIS in obj_func_min.getList_IS_to_query():
    noise_vars = np.array([obj_func_min.noise_and_cost_func(indexIS, pt)[0] for pt in init_pts[index_dataset]])
    listPrevData.append((init_pts[index_dataset], init_vals[index_dataset], noise_vars))
    index_dataset += 1
##############

data = createHistoricalDataForPES(obj_func_min.getDim(), listPrevData, indexFirstIS)
prior_alpha_mean = np.var(listPrevData[0][1])
###############################################
# TODO: encapsulate the code for getting data above, ideally just need one line to load obj_func_min, obj_func_max, data, and other constants needed in the algorithm

table_name = "pes_hyper_{0}".format(obj_func_min.getFuncName())
print table_name
################################
### Transform data to (0,1)^d space
lower_bounds = obj_func_min._search_domain[:, 0]
upper_bounds = obj_func_min._search_domain[:, 1]
print "lower_bounds {0}\n upper_bounds {1}".format(lower_bounds, upper_bounds)

transformed_data = HistoricalData(obj_func_min.getDim() + 1)
for pt, val, var in zip(data.points_sampled, data.points_sampled_value, data.points_sampled_noise_variance):
    transformed_data.append_sample_points([[np.concatenate(([pt[0]],scale_forward(pt[1:], lower_bounds, upper_bounds))), val, var], ])
# print "num_data {0}, dim {1}\n samples \n{2}\n vals {3}\n noise {4}".format(transformed_data.num_sampled,
#                                                                             transformed_data.dim,
#                                                                             transformed_data.points_sampled,
#                                                                             transformed_data.points_sampled_value,
#                                                                             transformed_data.points_sampled_noise_variance)
################################
# slice sampling for pes begins
# construct prior for entries of Cholesky of Kt
num_is = obj_func_min.getNumIS()
kt_mean = -1. * np.ones(
    int(num_is * (num_is + 1) / 2))  # the entries are in log space
i = 0
advance = 1
while i < len(kt_mean):
    kt_mean[i] = 0.0  # diagonal entries are set to e^(0.0) = 1
    advance += 1
    i += advance
kt_var = 100. * np.ones(len(kt_mean))
# construct prior for hypers of regular exp kernel
prior_mean_IS_0 = np.concatenate(([prior_alpha_mean], [1.] * obj_func_min.getDim()))
prior_var_IS_0 = np.concatenate(([np.power(prior_alpha_mean / 2., 2.)], [25.] * obj_func_min.getDim()))
prior_mean = np.concatenate((kt_mean, prior_mean_IS_0))
prior_var = np.concatenate((kt_var, prior_var_IS_0))

### setup cov and gp model
cov = ProductKernel(prior_mean, obj_func_min.getDim() + 1, num_is)
gp_likelihood = GaussianProcessLogMarginalLikelihood(cov, transformed_data)
slice_sampler = SliceSampler(prior_mean, prior_var, 1, prior_mean)
hypers_mat = np.array([None] * (n_samples * len(prior_mean))).reshape((n_samples, len(prior_mean)))
for i in range(n_burnin):
    h = slice_sampler.sample(gp_likelihood)
    print "burnin {0}, {1}".format(i, h)
for i in range(n_samples):
    hypers_mat[i, :] = slice_sampler.sample(gp_likelihood)
    print "sample {0}, {1}".format(i, hypers_mat[i, :])
    sql_util_miso.write_array_to_table(table_name, hypers_mat[i, :])
