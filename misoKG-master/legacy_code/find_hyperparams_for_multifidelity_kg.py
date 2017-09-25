from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

from multifidelity_KG.model.covariance_function import MixedSquareExponential
from multifidelity_KG.model.hyperparameter_optimization import hyper_opt
from multifidelity_KG.obj_functions import RosenbrockNoiseFree
from multifidelity_KG.voi.knowledge_gradient import *
import sql_util

__author__ = 'jialeiwang'

obj_func_max = RosenbrockNoiseFree(num_IS=2, mult=-1.0)
func_name = 'rosenbrock_noisefree'
hyper_bounds = [(0.01, 100) for i in range((obj_func_max._dim+1)*(obj_func_max._num_IS+1))]
num_hyper_multistart = 5
search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_max._search_domain])

### Gen points for hyperparam estimation
data = HistoricalData(obj_func_max._dim+1)
num_pts_to_gen = 250
for i in range(obj_func_max._num_IS):
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
cov = MixedSquareExponential(hyperparameters=multistart_pts[0,:], total_dim=obj_func_max._dim+1, num_is=obj_func_max._num_IS)
for i in range(num_hyper_multistart):
    hyper, f, output = hyper_opt(cov, data=data, init_hyper=multistart_pts[i, :], hyper_bounds=hyper_bounds, approx_grad=False)
    print output
    if f < best_f:
        best_hyper = hyper
        best_f = f
sql_util.write_array_to_table('multifidelity_kg_hyperparam_'+func_name, best_hyper)
