import numpy
from joblib import Parallel, delayed
import pandas
import scipy

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess

from multifidelity_KG.model.covariance_function import MixedSquareExponential
from multifidelity_KG.obj_functions import Rosenbrock
from multifidelity_KG.voi.knowledge_gradient import *
import multifidelity_KG.voi.optimization_with_discretization as optimization_with_discretization
import sql_util
__author__ = 'jialeiwang'

### some constant
func_name = 'rosenbrock'
def noise_and_cost_func(IS, x):
    return (0.001, 1000) if IS == 1 else (0.01, 1)

obj_func_max = Rosenbrock(num_IS=2, noise_and_cost_func=noise_and_cost_func, mult=-1.0)
num_discretization = 5000
num_init_pts_all_IS = [5, 5]
num_multistart = 50
hyper_param = pandas.read_sql_table('multifidelity_kg_hyperparam_'+func_name, sql_util.sql_engine).mean(axis=0).values
search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_max._search_domain])
### Gen initial points
data = HistoricalData(obj_func_max._dim+1)
for i in range(obj_func_max._num_IS):
    pts = search_domain.generate_uniform_random_points_in_domain(num_init_pts_all_IS[i])
    vals = [obj_func_max.evaluate(i+1, pt) for pt in pts]
    IS_pts = numpy.hstack(((i+1)*numpy.ones(num_init_pts_all_IS[i]).reshape((-1,1)), pts))
    sample_vars = [obj_func_max.noise_and_cost_func(i+1, pt)[0] for pt in pts]
    data.append_historical_data(IS_pts, vals, sample_vars)
cov_func = MixedSquareExponential(hyperparameters=hyper_param, total_dim=obj_func_max._dim+1, num_is=obj_func_max._num_IS)
gp = GaussianProcess(cov_func, data)

# print "start max mu"
# num_randomization = 100000
# random_pts = search_domain.generate_uniform_random_points_in_domain(num_randomization)
# zero_random_pts = numpy.hstack((numpy.zeros((num_randomization, 1)), random_pts))
# print "random pts generated"
# mu_list = gp.compute_mean_of_points(zero_random_pts)
# print "compute mean completed"
# best_mu = numpy.amax(mu_list)
# best_pt = random_pts[numpy.argmax(mu_list), :]

def compute_negative_mu(x):
    return -1.0 * gp.compute_mean_of_points(numpy.concatenate(([0], x)).reshape((1,-1)))[0]

def optimze_func(start_pt):
    result_x, result_f, output = scipy.optimize.fmin_l_bfgs_b(func=compute_negative_mu, x0=start_pt, fprime=None, args=(), approx_grad=True,
                                                              bounds=obj_func_max._search_domain, m=10, factr=10.0, pgtol=1e-10,
                                                              epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=15000, disp=0, callback=None)
    return result_x, result_f

start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)
with Parallel(n_jobs=50) as parallel:
    parallel_results = parallel(delayed(optimze_func)(pt) for pt in start_points)
best_mu_bfgs = numpy.inf
for i in range(len(parallel_results)):
    if best_mu_bfgs > parallel_results[i][1]:
        best_mu_bfgs = parallel_results[i][1]
        best_pt_bfgs = parallel_results[i][0]

