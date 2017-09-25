import sys
import random

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

from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartRosenbrock.rosenbrock_shifted import RosenbrockShifted
from coldStartRosenbrock.rosenbrock_slightshifted import RosenbrockSlightShifted

from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var1 import AssembleToOrderVar1
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4

import sql_util_cs

'''
Hyper-Estimation via MAP estimates
'''
#TODO the file name of previous datasets is not created correctly

### Rosenbrock family
# obj_func_min = RosenbrockSlightShifted( mult=1.0 )
# obj_func_max = RosenbrockSlightShifted( mult=-1.0 )                     # used by KG

### AssembleToOrderVanilla
obj_func_min = AssembleToOrderVar4( mult= -1.0 )
obj_func_max = AssembleToOrderVar4( )

func_name = obj_func_min.getFuncName()
# how many prev datasets should be incorporated? This determines the dim of the GP and the num of hypers
list_previous_datasets_to_load = [] #['data_vKG_ato_vanilla'] #['data_vKG_ato_var2'] #['data_vKG_rb_sinN'] # ['data_vKG_rb_vanN'] # []

### create list of previous datasets
pathToPickles = "../pickles/csCentered"
init_data_pickle_filename_prefix = func_name + '_IS_0_200_points_each_repl_' # the dataset for IS 0
replication_no = 0 #random.randint(0,2) # There are 3 pickle sets with 200 points. Pick one at random
complete_list_prev_datasets_to_load = ["{0}_repl_{1}".format(dataset_name, replication_no) for dataset_name in list_previous_datasets_to_load]
listPrevData = create_listPrevData(obj_func_min, complete_list_prev_datasets_to_load,
                                       replication_no, pathToPickles, init_data_pickle_filename_prefix)
number_of_previous_datasets = len(listPrevData) - 1

### create name for mysql table:
func_names_prev_datasets = []
for prev_dataset_name in list_previous_datasets_to_load:
    func_names_prev_datasets.append(prev_dataset_name.split("_")[-1])
names_used_datasets = "" if len(func_names_prev_datasets) == 0 else "_w_{0}".format("_".join(func_names_prev_datasets))
table_name = "{0}_hyper_{1}{2}".format("vkg" if len(func_names_prev_datasets) == 0 else "cskg", func_name, names_used_datasets)
print "table_name: {0}".format(table_name)

### specific for each acquisition function
hyper_bounds = [(0.01, 100) for i in range((obj_func_max._dim+1)*(number_of_previous_datasets+1))]
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
historical_data = createHistoricalDataForKG(obj_func_max._dim, listPrevData)

### Setup prior for MAP
prior_mean_IS_0 = numpy.concatenate(([numpy.var(listPrevData[0][1])], [1.]*obj_func_min._dim))
prior_mean_IS_i = numpy.concatenate(([10.], [1.]*obj_func_min._dim))
prior_mean = numpy.concatenate((prior_mean_IS_0, numpy.tile(prior_mean_IS_i, number_of_previous_datasets)))
prior_sig = numpy.eye(len(prior_mean)) * 25.
prior_sig[0, 0] = 1e6 # Jialei's original value: 5e5
prior = NormalPrior(prior_mean, prior_sig)
hyper_bounds[0] = (1., prior_mean[0] * 2)

### hyperparam opt
print "start hyperparam optimization..."
hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_bounds])
multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(num_hyper_multistart)
best_f = numpy.inf
cov = MixedSquareExponential(hyperparameters=multistart_pts[0,:], total_dim=obj_func_max._dim+1, num_is=number_of_previous_datasets)
for i in range(num_hyper_multistart):
    hyper, f, output = hyper_opt(cov, data=historical_data, init_hyper=multistart_pts[i, :],
                                 hyper_bounds=hyper_bounds, approx_grad=False, hyper_prior=prior)
    # print output
    if f < best_f:
        best_hyper = hyper
        best_f = f

print 'best_hyper=' + str(best_hyper)
print 'best_f= ' + str(best_f)
print "prior mean is: {0}".format(prior_mean)
sql_util_cs.write_array_to_table(table_name, best_hyper)
