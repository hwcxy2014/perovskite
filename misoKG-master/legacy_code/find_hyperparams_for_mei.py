import sys
import numpy as np


from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.covariance import SquareExponential

from assembleToOrderExtended.assembleToOrderExtended import AssembleToOrderExtended
from multifidelity_KG.model.hyperparameter_optimization import NormalPrior, hyper_opt
from load_and_store_data import load_data_from_a_min_problem, obtainHistoricalDataForEGO, \
    create_listPrevData, createHistoricalDataForMisoEI, match_pickle_filename
import sql_util


__author__ = 'jialeiwang'

num_hyper_multistart = 5
##############################################
### ATO
obj_func_min = AssembleToOrderExtended(mult=-1.0)

func_name = obj_func_min.getFuncName()
# how many prev datasets should be incorporated? This determines the dim of the GP and the num of hypers
list_previous_datasets_to_load = []
indexFirstIS = 0  # id of the first IS, used to label data accordingly
separateIS0 = 0  # 1 if IS0 is not observable and hence the data supplied for IS1, IS2, ...
### in coldstart this should be 0

### Load the training data
pathToPickles = "/fs/europa/g_pf/pickles/miso"
bias_filename = "atoext_IS_1_2_1000_points_eachdiffToIS0"
init_data_pickle_filename_prefix = func_name + '_IS_0_1_2_200_points_each_repl_'  # the dataset for IS 0
replication_no = 2  # random.randint(0,2) # There are 3 pickle sets with 200 points. Pick one at random
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

data_list, bias_sq_list = createHistoricalDataForMisoEI(obj_func_min.getDim(), listPrevData, directory=pathToPickles, bias_filename=bias_filename)
###############################################

###############################################
### Begin hyper opt
hyper_result = []
for data in data_list:
    # Setup prior for MAP
    prior_mean = np.concatenate(([np.var(data.points_sampled_value)], [1.]*obj_func_min.getDim()))
    prior_sig = np.eye(obj_func_min.getDim()+1) * 100.
    prior_sig[0,0] = np.power(prior_mean[0]/5., 2.)
    prior = NormalPrior(prior_mean, prior_sig)
    hyper_bounds = [(0.1, prior_mean[i]+2.*np.sqrt(prior_sig[i,i])) for i in range(obj_func_min.getDim()+1)]
    print "hyper bound {0}".format(hyper_bounds)
    hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_bounds])
    multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(num_hyper_multistart)
    best_f = np.inf
    cov = SquareExponential(prior_mean)
    for i in range(num_hyper_multistart):
        hyper, f, output = hyper_opt(cov, data=data, init_hyper=multistart_pts[i, :],
                                     hyper_bounds=hyper_bounds, approx_grad=False, hyper_prior=prior)
        # print output
        if f < best_f:
            best_hyper = hyper
            best_f = f
    print 'best_hyper=' + str(best_hyper)
    print 'best_f= ' + str(best_f)
    print "prior mean is: {0}".format(prior_mean)
    hyper_result = np.concatenate((hyper_result, best_hyper))
sql_util.write_array_to_table("mei_hyper_{0}".format(obj_func_min.getFuncName()), hyper_result)
