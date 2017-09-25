import numpy
import scipy.optimize

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.expected_improvement import ExpectedImprovement
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcessNew
from moe.optimal_learning.python.cpp_wrappers.covariance import MixedSquareExponential as cppMixedSquareExponential

from multifidelity_KG.model.covariance_function import MixedSquareExponential
from multifidelity_KG.obj_functions import Rosenbrock
from multifidelity_KG.voi.knowledge_gradient import *
from multifidelity_KG.voi.optimization import *
from multifidelity_KG.result_container import BenchmarkResult
import sql_util
from assembleToOrder.assembleToOrder import AssembleToOrder
import sample_initial_points

### The following parameters must be adapted for each simulator
num_iterations = 100    # Parameters recommended by Jialei
num_threads = 64
num_multistart = 64
num_max_restart = 3
restart_threshold = 1e-5
exploitation_threshold = 1e-15
exploitation_IS = 1     # IS to use when VOI does not work
func_name = 'assembleToOrder'
numIS = 4
num_x_prime = 5000
num_discretization_before_ranking = num_x_prime * 4
### parameters ends

obj_func_max = AssembleToOrder(numIS)                        # used by KG
obj_func_min = AssembleToOrder(numIS, mult=-1.0)             # our original problems are all assumed to be minimization!
search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_max._search_domain])
noise_and_cost_func = obj_func_min.noise_and_cost_func
# pickle load initial data. Here vals are taken from the minimization problem, so need to flip sign before feeding to KG GP model!
init_pts, init_vals = sample_initial_points.load_data_from_a_min_problem("pickles", "ATO_4_IS")
# setup benchmark result container
kg_hyper_param = pandas.read_sql_table('multifidelity_kg_hyperparam_' + func_name, sql_util.sql_engine).mean(axis=0).values
kg_data = HistoricalData(obj_func_max._dim + 1)
for i in range(obj_func_max._num_IS):
    IS_pts = numpy.hstack(((i + 1) * numpy.ones(len(init_pts[i])).reshape((-1, 1)), init_pts[i]))
    vals = -1.0 * numpy.array(init_vals[i])
    noise_vars = numpy.array([noise_and_cost_func(i+1, pt)[0] for pt in init_pts[i]])
    kg_data.append_historical_data(IS_pts, vals, noise_vars)
kg_cov = MixedSquareExponential(hyperparameters=kg_hyper_param, total_dim=obj_func_max._dim + 1, num_is=obj_func_max._num_IS)
kg_cov_cpp = cppMixedSquareExponential(hyperparameters=kg_hyper_param)
kg_gp_cpp = GaussianProcessNew(kg_cov_cpp, kg_data, obj_func_max._num_IS)

sample_IS = 1
points = search_domain.generate_uniform_random_points_in_domain(5000)
zero_points = numpy.hstack((numpy.zeros(len(points)).reshape((-1,1)), points))
n_kg_cpp, n_grad_kg_cpp= negative_kg_and_grad_given_x_prime(sample_IS, zero_points, noise_and_cost_func, kg_gp_cpp)
random_pts = search_domain.generate_uniform_random_points_in_domain(1000)
for i, pt in enumerate(random_pts):
    print i
    print "cpp kg: {0}".format(-n_kg_cpp(pt))
    # print "python grad kg: {0}, cpp grad kg: {1}".format(-n_grad_kg_python(pt), -n_grad_kg_cpp(pt))
    # print "cpp grad kg {0}".format(-n_grad_kg_cpp(pt))
    # print "python grad kg {0}".format(-n_grad_kg_python(pt))
