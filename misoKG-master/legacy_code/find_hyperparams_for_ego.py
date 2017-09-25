import numpy

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.covariance import SquareExponential

from multifidelity_KG.model.hyperparameter_optimization import hyper_opt
from multifidelity_KG.obj_functions import Rosenbrock
from multifidelity_KG.voi.knowledge_gradient import *
import sql_util

__author__ = 'jialeiwang'

### some constants
def noise_and_cost_func(IS, x):
    return (0.001, 1000) if IS == 1 else (0.01, 1)

obj_func_min = Rosenbrock(num_IS=2, noise_and_cost_func=noise_and_cost_func)
func_name = 'rosenbrock'
hyper_bounds = [(0.01, 100) for i in range(obj_func_min._dim+1)]
num_hyper_multistart = 3
num_pts_to_gen = 250
search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain])
cov = SquareExponential(numpy.ones(obj_func_min._dim+1))

### Gen points for hyperparam estimation
data = HistoricalData(obj_func_min._dim)
for i in range(obj_func_min._num_IS):
    pts = search_domain.generate_uniform_random_points_in_domain(num_pts_to_gen)
    vals = [obj_func_min.evaluate(i+1, pt) for pt in pts]
    sample_vars = [noise_and_cost_func(i+1, pt)[0] for pt in pts]
    data.append_historical_data(pts, vals, sample_vars)
# hyperparam opt
hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_bounds])
multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(num_hyper_multistart)
best_f = numpy.inf
for k in range(num_hyper_multistart):
    hyper , f, output = hyper_opt(cov, data=data, init_hyper=multistart_pts[k, :], hyper_bounds=hyper_bounds, approx_grad=False)
    if f < best_f:
        best_hyper = hyper
        best_f = f
sql_util.write_array_to_table('ego_hyperparam_'+func_name, best_hyper)
