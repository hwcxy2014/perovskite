import sys
import random
import math

sys.path.append("../")

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

from multifidelity_KG.model.covariance_function import MixedSquareExponential
from multifidelity_KG.model.hyperparameter_optimization import NormalPrior, hyper_opt
from multifidelity_KG.obj_functions import RosenbrockNoiseFree
from multifidelity_KG.voi.knowledge_gradient import *

from load_and_store_data import load_data_from_a_min_problem, obtainHistoricalDataForEGO, \
    create_listPrevData, createHistoricalDataForKG, match_pickle_filename

from assembleToOrderExtended import AssembleToOrderExtended

import sql_util_miso

""" Part of Matthias script that reads in data starts
"""
'''
Hyper-Estimation via MAP estimates for AssembleToOrderExtended

IS 0 = truth
IS 1 = biased
IS 2 = unbiased but noisy
'''

### AssembleToOrderVanilla
obj_func_min = AssembleToOrderExtended( mult = -1.0 )
obj_func_max = AssembleToOrderExtended( )

func_name = obj_func_min.getFuncName()
# how many prev datasets should be incorporated? This determines the dim of the GP and the num of hypers
list_previous_datasets_to_load = []
indexFirstIS = 0 # id of the first IS, used to label data accordingly
separateIS0 = 0 # 1 if IS0 is not observable and hence the data supplied for IS1, IS2, ...
### in coldstart this should be 0

### Load the training data
pathToPickles = "../pickles/miso"
init_data_pickle_filename_prefix = func_name + '_IS_1_2_3_200_points_each_repl_' # the dataset for IS 0
replication_no = 2 #random.randint(0,2) # There are 3 pickle sets with 200 points. Pick one at random
#complete_list_prev_datasets_to_load = ["{0}_repl_{1}".format(dataset_name, replication_no) for dataset_name in list_previous_datasets_to_load]
listPrevData = [] #create_listPrevData(obj_func_min, complete_list_prev_datasets_to_load, replication_no, pathToPickles, init_data_pickle_filename_prefix)

init_data_pickle_filename = init_data_pickle_filename_prefix + str(replication_no)
init_pts, init_vals = load_data_from_a_min_problem(pathToPickles, init_data_pickle_filename)
# init_pts and init_vals contain a list of lists of pts/vals, one dataset for each IS

### Load in correct ordering to compute hypers
#################################
### Load truth IS as IS 0
#################################
select_dataset = 2
corresponding_IS = 0
noise_vars = numpy.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))
select_dataset = 1
corresponding_IS = 2
noise_vars = numpy.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))
select_dataset = 0
corresponding_IS = 1
noise_vars = numpy.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))
### load them one after another
# for index in range(3):
#     select_dataset = index
#     corresponding_IS = index + 1
#     noise_vars = numpy.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
#     listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))


#################################
### create name for mysql table:
func_names_prev_datasets = []
for prev_dataset_name in list_previous_datasets_to_load:
    func_names_prev_datasets.append(prev_dataset_name.split("_")[-1])
names_used_datasets = "" if len(func_names_prev_datasets) == 0 else "_w_{0}".format("_".join(func_names_prev_datasets))
table_name = "{0}_hyper_{1}{2}".format("vkg" if len(func_names_prev_datasets) == 0 else "cskg", func_name, names_used_datasets)
print "table_name: {0}".format(table_name)

### specific for each acquisition function
hyper_bounds = [(0.01, 100) for i in range((obj_func_max.getDim()+1) * (obj_func_max.getNumIS() + separateIS0))]
# print hyper_bounds
num_hyper_multistart = 5
search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_max._search_domain])

### Gen points for hyperparam estimation
# historical_data = HistoricalData(obj_func_max._dim + 1)
# for indexIS in list_IS_to_query:
#     pts = search_domain.generate_uniform_random_points_in_domain(num_init_pts_each_IS)
#     vals = [obj_func_max.evaluate(indexIS, pt) for pt in pts]
#     IS_pts = numpy.hstack((indexIS*numpy.ones(num_init_pts_each_IS).reshape((-1,1)), pts))
#     sample_vars = [obj_func_max.noise_and_cost_func(indexIS, pt)[0] for pt in pts]
#     historical_data.append_historical_data(IS_pts, vals, sample_vars)
historical_data = createHistoricalDataForKG(obj_func_max.getDim(), listPrevData, indexFirstIS)

### Setup prior for MAP
prior_mean_IS_0 = numpy.concatenate(([numpy.var(listPrevData[0][1])], [1.]*obj_func_min.getDim()))
prior_mean_IS_1 = numpy.concatenate(([3200.0], [1.]*obj_func_min.getDim())) # mean estimated from data for AssembleToOrderExtended
prior_mean_IS_2 = numpy.concatenate(([0.0], [1.]*obj_func_min.getDim())) # mean estimated from data for AssembleToOrderExtended
# prior_mean_IS_i = numpy.concatenate(([10.], [1.]*obj_func_min._dim))
# prior_mean = numpy.concatenate((prior_mean_IS_0, numpy.tile(prior_mean_IS_i, obj_func_max.getNumIS() -1 + separateIS0 )))
prior_mean = numpy.concatenate((prior_mean_IS_0, prior_mean_IS_1, prior_mean_IS_2))
prior_sig = numpy.eye(len(prior_mean)) * 25.
# prior_sig[0, 0] = 1e6 # Jialei's original value: 5e5
for indexIS in range(obj_func_min.getNumIS()):
    pos_signal_variance = (obj_func_min.getDim() + 1) * indexIS # compute the position that corresponds to signal variance
    # we assume that for each IS the ordering is signal_var, beta_1, beta_2, ..., bet_dim
    prior_sig[pos_signal_variance,pos_signal_variance] = math.pow( (prior_mean[pos_signal_variance]/5.0) + 1e-6 , 2) # Jialei's suggestions
    hyper_bounds[pos_signal_variance] = (1., max(prior_mean[pos_signal_variance] * 2, 100))
prior = NormalPrior(prior_mean, prior_sig)
# hyper_bounds[0] = (1., prior_mean[0] * 2) ### can cause errors

""" Matthias' code ends
"""
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from moe.optimal_learning.python.python_version.log_likelihood import GaussianProcessLogMarginalLikelihood

print "prior mean\n{0}\nprior sig diag\n{1}".format(prior_mean, numpy.diag(prior_sig))
print "num_is {0}".format(obj_func_max.getNumIS() -1 + separateIS0)
hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_bounds])
print "hyper bounds\n{0}".format(hyper_bounds)
cov = MixedSquareExponential(hyperparameters=prior_mean, total_dim=obj_func_max.getDim()+1, num_is=obj_func_max.getNumIS() -1 + separateIS0 )
gp_likelihood = GaussianProcessLogMarginalLikelihood(cov, historical_data)

with PdfPages('posterior_plot_1.pdf') as pdf:
    for d in range(len(prior_mean)):
        x_list = numpy.linspace(hyper_bounds[d][0], hyper_bounds[d][1], 100)
        y_list = numpy.zeros(len(x_list))
        x = numpy.copy(prior_mean)
        for i,e in enumerate(x_list):
            print "plot {0}, {1}th pt".format(d, i)
            x[d] = e
            gp_likelihood.set_hyperparameters(x)
            y_list[i] = gp_likelihood.compute_log_likelihood() + prior.compute_log_likelihood(x)
        plt.figure()
        plt.plot(x_list, y_list, 'r-o')
        plt.title(str(d))
        pdf.savefig()
        plt.close()
