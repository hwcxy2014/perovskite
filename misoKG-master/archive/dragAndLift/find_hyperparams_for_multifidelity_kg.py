from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

import sys
sys.path.append("../")
from multifidelity_KG.model.covariance_function import MixedSquareExponential
from multifidelity_KG.model.hyperparameter_optimization import hyper_opt
from multifidelity_KG.voi.knowledge_gradient import *
import sql_util

from dragAndLift import DragAndLift

__author__ = 'matthiaspoloczek'

'''
Estimate hyper-parameters for KG

based on Jialei's Rosenbrock code -- Many Thanks!
'''


# the next lines are dependent on the problem
func_name = 'dragAndLift'
obj_func_max = DragAndLift( mult=-1.0 ) # mult=-1.0 because dragAndLift is a minimization problem
num_pts_to_gen = 10 # numpy.repeat( 250, obj_func_max.getNumIS())


hyper_bounds = [(0.01, 100) for i in range((obj_func_max.getDim()+1)*(obj_func_max.getNumIS() +1))]
num_hyper_multistart = 5
search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_max.getSearchDomain()])

### Gen points for hyperparam estimation
data = HistoricalData(obj_func_max.getDim()+1)                     # should go into the objective func obj
for i in range(obj_func_max.getNumIS()):
    pts = search_domain.generate_uniform_random_points_in_domain(num_pts_to_gen)
    vals = [obj_func_max.evaluate(i+1, pt) for pt in pts]
    IS_pts = numpy.hstack(((i+1)*numpy.ones(num_pts_to_gen).reshape((-1,1)), pts))
    sample_vars = [obj_func_max.noise_and_cost_func(i+1, pt)[0] for pt in pts]
    data.append_historical_data(IS_pts, vals, sample_vars)

# hyperparam opt
print "start hyperparam optimization..."
hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_bounds])
multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(num_hyper_multistart)
best_f = numpy.inf
cov = MixedSquareExponential(hyperparameters=multistart_pts[0,:], total_dim=obj_func_max.getDim()+1, num_is=obj_func_max.getNumIS())
for i in range(num_hyper_multistart):
    hyper, f, output = hyper_opt(cov, data=data, init_hyper=multistart_pts[i, :], hyper_bounds=hyper_bounds, approx_grad=False)
    if f < best_f:
        best_hyper = hyper
        best_f = f
#sql_util.write_array_to_table('multifidelity_kg_hyperparam_'+func_name, best_hyper)
#TODO uncomment for real runs
