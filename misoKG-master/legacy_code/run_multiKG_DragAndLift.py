from joblib import Parallel, delayed
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
from multifidelity_KG.voi.knowledge_gradient import *
from multifidelity_KG.voi.optimization import *
from multifidelity_KG.result_container import BenchmarkResult
import sql_util
import sample_initial_points
from assembleToOrder.assembleToOrder import AssembleToOrder
from multifidelity_KG.obj_functions import Rosenbrock
from mothers_little_helpers import process_parallel_results, load_init_points_for_all_IS, load_vals

__author__ = 'jialeiwang'

### The following parameters must be adapted for each simulator
numIS = 2
truth_IS = 0
exploitation_IS = 2     # IS to use when VOI does not work
func_name = 'rosenbrock'
init_data_pickle_filename = "rosenbrock_2_IS"
benchmark_result_table_name = "rosenbrock_multiKG_newCost_2"
obj_func_max = Rosenbrock(numIS, mult=-1.0)                        # used by KG
obj_func_min = Rosenbrock(numIS, mult=1.0)             # our original problems are all assumed to be minimization!
# less important params
exploitation_threshold = 1e-5
num_x_prime = 3000
num_discretization_before_ranking = num_x_prime * 3
num_iterations = 100
num_threads = 64
num_multistart = 64
num_candidate_start_points = 500
### end parameter

search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_max._search_domain])
noise_and_cost_func = obj_func_min.noise_and_cost_func

# Load initial data from pickle
init_pts = load_init_points_for_all_IS("pickles", init_data_pickle_filename, obj_func_min._numIS)
init_vals = load_vals("pickles", init_data_pickle_filename, obj_func_min._numIS)
#init_pts, init_vals = sample_initial_points.load_data_from_a_min_problem("pickles", init_data_pickle_filename)

# setup benchmark result container
multi_kg_result = BenchmarkResult(num_iterations, obj_func_max._dim, benchmark_result_table_name)
kg_hyper_param = pandas.read_sql_table('multifidelity_kg_hyperparam_' + func_name, sql_util.sql_engine).mean(axis=0).values
kg_data = HistoricalData(obj_func_max._dim + 1)
best_sampled_val = numpy.inf
for i in range(obj_func_max._num_IS):
    IS_pts = numpy.hstack(((i + 1) * numpy.ones(len(init_pts[i])).reshape((-1, 1)), init_pts[i]))

    # multiply all values by -1 since we assume that the training data stems from the minimization version
    # but misoKG uses the maximization version
    vals = -1.0 * numpy.array(init_vals[i])

    # obtain what used to be sample_vars
    noise_vars = numpy.array([noise_and_cost_func(i+1, pt)[0] for pt in init_pts[i]])
    kg_data.append_historical_data(IS_pts, vals, noise_vars)

    # find the best initial value
    if numpy.amin(init_vals[i]) < best_sampled_val:
        best_sampled_val = numpy.amin(init_vals[i])
        best_sampled_point = init_pts[i][numpy.argmin(init_vals[i]), :]
truth_at_best_sampled = obj_func_min.evaluate(truth_IS, best_sampled_point)

kg_cov = MixedSquareExponential(hyperparameters=kg_hyper_param, total_dim=obj_func_max._dim + 1, num_is=obj_func_max._num_IS)
kg_cov_cpp = cppMixedSquareExponential(hyperparameters=kg_hyper_param)
kg_gp_cpp = GaussianProcessNew(kg_cov_cpp, kg_data, obj_func_max._num_IS)
for kg_n in range(num_iterations):
    print "itr {0}, {1}".format(kg_n, benchmark_result_table_name)
    ### First discretize points and then only keep the good points idea
    discretization_points = search_domain.generate_uniform_random_points_in_domain(num_discretization_before_ranking)
    discretization_points = numpy.hstack((numpy.zeros((num_discretization_before_ranking,1)), discretization_points))
    all_mu = kg_gp_cpp.compute_mean_of_points(discretization_points)
    sorted_idx = numpy.argsort(all_mu)
    all_zero_x_prime = discretization_points[sorted_idx[-num_x_prime:], :]
    ### idea ends
    # all_zero_x_prime = numpy.hstack((numpy.zeros((num_x_prime,1)), search_domain.generate_uniform_random_points_in_domain(num_x_prime)))

    def min_kg_unit(start_pt, IS):
        func_to_min, grad_func = negative_kg_and_grad_given_x_prime(IS, all_zero_x_prime, noise_and_cost_func, kg_gp_cpp)
        return bfgs_optimization_grad(start_pt, func_to_min, grad_func, obj_func_max._search_domain)

    def compute_kg_unit(x, IS):
        return compute_kg_given_x_prime(IS, x, all_zero_x_prime, noise_and_cost_func(IS, x)[0], noise_and_cost_func(IS, x)[1], kg_gp_cpp)

    def find_mu_star(start_pt):
        return bfgs_optimization(start_pt, negative_mu_kg(kg_gp_cpp), obj_func_max._search_domain)

    min_negative_kg = numpy.inf
    with Parallel(n_jobs=num_threads) as parallel:
        for i in range(obj_func_max._num_IS):
            start_points_prepare = search_domain.generate_uniform_random_points_in_domain(num_candidate_start_points)
            kg_vals = parallel(delayed(compute_kg_unit)(x, i+1) for x in start_points_prepare)
            sorted_idx_kg = numpy.argsort(kg_vals)
            start_points = start_points_prepare[sorted_idx_kg[-num_multistart:], :]
            parallel_results = parallel(delayed(min_kg_unit)(pt, i+1) for pt in start_points)
            inner_min, inner_min_point = process_parallel_results(parallel_results)
            if inner_min < min_negative_kg:
                min_negative_kg = inner_min
                point_to_sample = inner_min_point
                sample_IS = i + 1
            print "IS {0}, KG {1}".format(i+1, -inner_min)
        start_points_prepare = search_domain.generate_uniform_random_points_in_domain(num_candidate_start_points)
        mu_vals = kg_gp_cpp.compute_mean_of_points(numpy.hstack((numpy.zeros((num_candidate_start_points, 1)), start_points_prepare)))
        start_points = start_points_prepare[numpy.argsort(mu_vals)[-num_multistart:], :]
        parallel_results = parallel(delayed(find_mu_star)(pt) for pt in start_points)
        negative_mu_star, mu_star_point = process_parallel_results(parallel_results)
        print "mu_star found"
    if -min_negative_kg < exploitation_threshold:
        sample_IS = exploitation_IS
        print "KG search failed, do exploitation"
        point_to_sample = mu_star_point
    sample_val = obj_func_min.evaluate(sample_IS, point_to_sample)
    predict_mean = kg_gp_cpp.compute_mean_of_points(numpy.concatenate(([0], point_to_sample)).reshape((1,-1)))[0]
    predict_var = kg_gp_cpp.compute_variance_of_points(numpy.concatenate(([0], point_to_sample)).reshape((1,-1)))[0,0]
    cost = noise_and_cost_func(sample_IS, point_to_sample)[1]
    mu_star_var = kg_gp_cpp.compute_variance_of_points(numpy.concatenate(([0], mu_star_point)).reshape((1,-1)))[0,0]
    mu_star_truth = obj_func_min.evaluate(truth_IS, mu_star_point)

    multi_kg_result.add_entry(point_to_sample, sample_IS, sample_val, best_sampled_val, truth_at_best_sampled, predict_mean, predict_var, cost, -min_negative_kg, mu_star=negative_mu_star, mu_star_var=mu_star_var, mu_star_truth=mu_star_truth, mu_star_point=mu_star_point)
    print "pt: {0} \n IS: {1} \n val: {2} \n voi: {3} \n best_sample_truth: {4} \n mu_star_point: {5} \n mu_star_truth: {6} \n total cost: {7}".format(
        point_to_sample, sample_IS, sample_val, -min_negative_kg, truth_at_best_sampled, mu_star_point, mu_star_truth, multi_kg_result._total_cost
    )
    if sample_val < best_sampled_val:
        best_sampled_val = sample_val
        best_sampled_point = point_to_sample
        truth_at_best_sampled = obj_func_min.evaluate(truth_IS, best_sampled_point)

    kg_gp_cpp.add_sampled_points([SamplePoint(numpy.concatenate(([sample_IS], point_to_sample)), -sample_val, noise_and_cost_func(sample_IS, point_to_sample)[0])])
