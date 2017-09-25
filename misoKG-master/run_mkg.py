import boto
from boto.s3.connection import S3Connection
import numpy as np
# import pickle
from joblib import Parallel, delayed
import sys

from moe.optimal_learning.python.cpp_wrappers.covariance import MixedSquareExponential as cppMixedSquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcessNew
from moe.optimal_learning.python.data_containers import SamplePoint

from constants import s3_bucket_name
from data_io import process_parallel_results, send_data_to_s3
from multifidelity_KG.voi.optimization import *
from problems.identifier import identify_problem
conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)

__author__ = 'jialeiwang'

# construct problem instance given CMD args
# format: run_mkg.py ${benchmark_name} ${repl_no} ${func_idx}
argv = sys.argv[1:]
if argv[0].find("kg") < 0:
    raise ValueError("benchmark is not mkg/kg!")
problem = identify_problem(argv, bucket)

# algorithm params
exploitation_threshold = 1e-5
num_x_prime = 3000
num_discretization_before_ranking = num_x_prime * 2
num_threads = 32
num_multistart = 32

# mkg begins
kg_cov_cpp = cppMixedSquareExponential(hyperparameters=problem.hyper_param)
kg_gp_cpp = GaussianProcessNew(kg_cov_cpp, problem.hist_data, num_IS_in=problem.num_is_in)
# data containers for pickle storage
list_best = []
list_cost = []
list_sampled_IS = []
list_sampled_points = []
list_sampled_vals = []
list_noise_var = []
list_mu_star_truth = []
list_raw_voi = []
init_best_idx = numpy.argmax(problem.hist_data._points_sampled_value[problem.hist_data._points_sampled[:, 0] == problem.truth_is])
best_sampled_val = -1.0 * problem.hist_data._points_sampled_value[init_best_idx]    # minus sign is because vals in hist_data were
                                                                                    # obtained from obj_func_max, while all values
                                                                                    # to store are from obj_func_min, for consistency
truth_at_init_best_sampled = problem.obj_func_min.evaluate(problem.truth_is, problem.hist_data.points_sampled[init_best_idx, 1:])
truth_at_best_sampled = truth_at_init_best_sampled
best_mu_star_truth = np.inf
total_cost = 0
for kg_iteration in range(problem.num_iterations):
    ### First discretize points and then only keep the good points idea
    discretization_points = problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_discretization_before_ranking)
    discretization_points = np.hstack((np.zeros((num_discretization_before_ranking,1)), discretization_points))
    all_mu = kg_gp_cpp.compute_mean_of_points(discretization_points)
    sorted_idx = np.argsort(all_mu)
    all_zero_x_prime = discretization_points[sorted_idx[-num_x_prime:], :]
    ### idea ends

    def min_kg_unit(start_pt, IS):
        func_to_min, grad_func = negative_kg_and_grad_given_x_prime(IS, all_zero_x_prime, problem.obj_func_min.noise_and_cost_func, kg_gp_cpp)
        return bfgs_optimization_grad(start_pt, func_to_min, grad_func, problem.obj_func_min._search_domain)

    def compute_kg_unit(x, IS):
        return compute_kg_given_x_prime(IS, x, all_zero_x_prime, problem.obj_func_min.noise_and_cost_func(IS, x)[0], problem.obj_func_min.noise_and_cost_func(IS, x)[1], kg_gp_cpp)

    def find_mu_star(start_pt):
        return bfgs_optimization(start_pt, negative_mu_kg(kg_gp_cpp), problem.obj_func_min._search_domain)

    min_negative_kg = np.inf
    list_raw_kg_this_itr = []
    with Parallel(n_jobs=num_threads) as parallel:
        for IS in problem.list_sample_is:
            parallel_results = parallel(delayed(min_kg_unit)(pt, IS) for pt in problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_multistart))
            inner_min, inner_min_point = process_parallel_results(parallel_results)
            list_raw_kg_this_itr.append(-inner_min * problem.obj_func_min.noise_and_cost_func(IS, inner_min_point)[1])
            if inner_min < min_negative_kg:
                min_negative_kg = inner_min
                point_to_sample = inner_min_point
                sample_is = IS
        parallel_results = parallel(delayed(find_mu_star)(pt) for pt in problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_multistart))
        negative_mu_star, mu_star_point = process_parallel_results(parallel_results)
    if -min_negative_kg * problem.obj_func_min.noise_and_cost_func(sample_is, point_to_sample)[1] < exploitation_threshold:
        print "KG search failed, do exploitation"
        point_to_sample = mu_star_point
        sample_is = problem.exploitation_is

    sample_val = problem.obj_func_min.evaluate(sample_is, point_to_sample)
    # compute mu_star_truth
    if np.array_equal(mu_star_point, point_to_sample) and problem.truth_is == sample_is:
        mu_star_truth = sample_val
    else:
        mu_star_truth = problem.obj_func_min.evaluate(problem.truth_is, mu_star_point)
    # update best_mu_star_truth
    if mu_star_truth < best_mu_star_truth:
        best_mu_star_truth = mu_star_truth
    # update best_sampled_val and truth_at_best_sampled
    if sample_val < best_sampled_val:
        best_sampled_val = sample_val
        if(problem.truth_is == sample_is):
            truth_at_best_sampled = sample_val
        else:
            truth_at_best_sampled = problem.obj_func_min.evaluate(problem.truth_is, point_to_sample)

    # NOTE: while Jialei worked everywhere with the values of the minimization problem in the computation, he used the maximization obj values for the GP.
    # That is why here sample_val is multiplied by -1
    kg_gp_cpp.add_sampled_points([SamplePoint(np.concatenate(([sample_is], point_to_sample)), -1.0 * sample_val, problem.obj_func_min.noise_and_cost_func(sample_is, point_to_sample)[0])])
    best_this_itr = min(best_mu_star_truth, truth_at_best_sampled)
    total_cost += problem.obj_func_min.noise_and_cost_func(sample_is, point_to_sample)[1]
    # save data from this iteration:
    list_best.append(best_this_itr)
    list_cost.append(total_cost)
    list_sampled_IS.append(sample_is)
    list_sampled_points.append(point_to_sample)
    list_sampled_vals.append(sample_val)
    list_noise_var.append(problem.obj_func_min.noise_and_cost_func(sample_is, point_to_sample)[0])
    list_mu_star_truth.append(mu_star_truth)
    list_raw_voi.append(list_raw_kg_this_itr)

    result_to_pickle = {
        "best": np.array(list_best),
        "cost": np.array(list_cost),
        "sampled_is": np.array(list_sampled_IS),
        "sampled_points": np.array(list_sampled_points),
        "sampled_vals": np.array(list_sampled_vals),
        "sampled_noise_var": np.array(list_noise_var),
        "mu_star_truth": np.array(list_mu_star_truth),
        "raw_voi": np.array(list_raw_voi),
        "init_best_truth": truth_at_init_best_sampled,
        }

    # write data to pickle.
    send_data_to_s3(bucket, problem.result_path, result_to_pickle)
    if problem.data_path is not None:
        data_to_pickle = {
            "points": np.array(list_sampled_points),
            "vals": np.array(list_sampled_vals),
            "noise": np.array(list_noise_var),
        }
        send_data_to_s3(bucket, problem.data_path, data_to_pickle)
