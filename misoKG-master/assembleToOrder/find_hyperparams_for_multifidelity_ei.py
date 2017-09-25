from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.covariance import SquareExponential

import sys
sys.path.append("../")
from multifidelity_KG.model.hyperparameter_optimization import hyper_opt
from multifidelity_KG.voi.knowledge_gradient import *
import sql_util

from assembleToOrder import AssembleToOrder

__author__ = 'matthiaspoloczek'

'''
Estimate hyper-parameters for the Lam et al. approach

based on Jialei's Rosenbrock code -- Many Thanks!
'''

func_name = 'assembleToOrder'
obj_func_min = AssembleToOrder(numIS=4,mult=-1.0)

hyper_bounds = [(0.01, 100) for i in range(obj_func_min.getDim() +1)]
num_hyper_multistart = 3
num_pts_to_gen = 250
search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min.getSearchDomain()])
cov = SquareExponential(numpy.ones(obj_func_min.getDim() +1))

hyper_param = numpy.zeros((obj_func_min.getNumIS(), obj_func_min.getDim()+1))
### Gen points for hyperparam estimation
for i in range(obj_func_min.getNumIS()):
    data = HistoricalData(obj_func_min.getDim())
    pts = search_domain.generate_uniform_random_points_in_domain(num_pts_to_gen)
    vals = [obj_func_min.evaluate(i+1, pt) for pt in pts]
    sample_vars = [obj_func_min.noise_and_cost_func(i+1, pt)[0] for pt in pts]
    data.append_historical_data(pts, vals, sample_vars)
    # hyperparam opt
    hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_bounds])
    multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(num_hyper_multistart)
    best_f = numpy.inf
    for k in range(num_hyper_multistart):
        hyper , f, output = hyper_opt(cov, data=data, init_hyper=multistart_pts[k, :], hyper_bounds=hyper_bounds, approx_grad=False)
        if f < best_f:
            best_hyper = numpy.copy(hyper)
            best_f = f
    hyper_param[i, :] = best_hyper
sql_util.write_array_to_table('multifidelity_ei_hyperparam_'+func_name, hyper_param.flatten())
#TODO uncomment for real runs
