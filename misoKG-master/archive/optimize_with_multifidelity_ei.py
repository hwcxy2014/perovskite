import numpy
from joblib import Parallel, delayed
import pandas
import scipy

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
from moe.optimal_learning.python.python_version.covariance import SquareExponential

from multifidelity_KG.obj_functions import Rosenbrock
import multifidelity_KG.voi.multifidelity_expected_improvement
import sql_util

__author__ = 'jialeiwang'

### some constant
func_name = 'rosenbrock'
def noise_and_cost_func(IS, x):
    return (0.001, 1000) if IS == 1 else (0.01, 1)

obj_func_min = Rosenbrock(num_IS=2, noise_and_cost_func=noise_and_cost_func, mult=1.0)
num_init_pts_all_IS = [5, 5]
num_multistart = 50
hyper_param = pandas.read_sql_table('multifidelity_ei_hyperparams_'+func_name, sql_util.sql_engine).mean(axis=0).values
search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain])
### Gen initial points
gp_list = []
data = []
for i in range(obj_func_min._num_IS):
    data.append(HistoricalData(obj_func_min._dim))
    pts = search_domain.generate_uniform_random_points_in_domain(num_init_pts_all_IS[i])
    vals = [obj_func_min.evaluate(i+1, pt) for pt in pts]
    sample_vars = [obj_func_min.noise_and_cost_func(i+1, pt)[0] for pt in pts]
    data[i].append_historical_data(pts, vals, sample_vars)
    cov_func = SquareExponential(hyper_param[i*(obj_func_min._dim+1):(i+1)*(obj_func_min._dim+1)])
    gp_list.append(GaussianProcess(cov_func, data[i]))
### VOI analysis
multifidelity_expected_improvement_evaluator = multifidelity_KG.voi.multifidelity_expected_improvement.MultifidelityExpectedImprovement(gp_list, noise_and_cost_func)

IS = 2
def negative_ei_func(x):
    return -1.0 * multifidelity_expected_improvement_evaluator.compute_expected_improvement(x)

start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)

def optimze_func(start_pt):
    result_x, result_f, output = scipy.optimize.fmin_l_bfgs_b(func=negative_ei_func, x0=start_pt, fprime=None, args=(), approx_grad=True,
                                                              bounds=obj_func_min._search_domain, m=10, factr=10.0, pgtol=1e-10,
                                                              epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, disp=0, callback=None)
    print output
    return result_x, result_f

with Parallel(n_jobs=50) as parallel:
    parallel_results = parallel(delayed(optimze_func)(pt) for pt in start_points)
min_negative_ei = numpy.inf
for i in range(len(parallel_results)):
    if min_negative_ei > parallel_results[i][1]:
        min_negative_ei = parallel_results[i][1]
        best_pt = parallel_results[i][0]
