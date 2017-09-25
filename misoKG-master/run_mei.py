import boto
from boto.s3.connection import S3Connection
from joblib import Parallel, delayed
import numpy as np
import sys

from moe.optimal_learning.python.data_containers import SamplePoint
from moe.optimal_learning.python.python_version.covariance import SquareExponential

from constants import s3_bucket_name
from data_io import process_parallel_results, send_data_to_s3
from multifidelity_KG.voi.optimization import *
from problems.identifier import identify_problem
conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)

__author__ = 'jialeiwang'

# construct problem instance given CMD args
# format: refer to nbs file
argv = sys.argv[1:]
if argv[0].find("mei") < 0:
    raise ValueError("benchmark is not mei!")
problem = identify_problem(argv, bucket)

# algo params
num_threads = 8
num_multistart = 8
### MP: paper experiments were performed with both params set to 64 and 32 (lrMU)

search_domain = problem.obj_func_min.get_moe_domain()
noise_and_cost_func = problem.obj_func_min.noise_and_cost_func
best_sampled_val = numpy.inf
best_sampled_point = None
gp_dict = dict()
for IS in problem.list_sample_is:
    cov_func = SquareExponential(problem.hyper_param[IS])
    gp_dict[IS] = GaussianProcess(cov_func, problem.hist_data[IS])
    tmp_argmin = np.argmin(problem.hist_data[IS]._points_sampled_value)
    if problem.hist_data[IS]._points_sampled_value[tmp_argmin] < best_sampled_val:
        best_sampled_val = problem.hist_data[IS]._points_sampled_value[tmp_argmin]
        best_sampled_point = problem.hist_data[IS]._points_sampled[tmp_argmin, :]
init_best_idx = numpy.argmin(problem.hist_data[problem.truth_is]._points_sampled_value)
truth_at_init_best_sampled = problem.obj_func_min.evaluate(problem.truth_is, problem.hist_data[problem.truth_is]._points_sampled[init_best_idx, :])
truth_at_best_sampled = truth_at_init_best_sampled

list_best = []
list_cost = []
list_sampled_IS = []
list_sampled_points = []
list_sampled_vals = []
list_noise_var = []
list_raw_voi = []
total_cost = 0
for ei_iteration in range(problem.num_iterations):
    multifidelity_expected_improvement_evaluator = MultifidelityExpectedImprovement(gp_dict, noise_and_cost_func, problem.list_sample_is)
    min_negative_ei = numpy.inf

    def negative_ei_func(x):
        return -1.0 * multifidelity_expected_improvement_evaluator.compute_expected_improvement(x)

    def min_negative_ei_func(start_point):
        return bfgs_optimization(start_point, negative_ei_func, problem.obj_func_min._search_domain)

    with Parallel(n_jobs=num_threads) as parallel:
        # start_points_prepare = search_domain.generate_uniform_random_points_in_domain(num_candidate_start_points)
        # ei_vals = parallel(delayed(negative_ei_func)(x) for x in start_points_prepare)
        # sorted_idx_kg = numpy.argsort(ei_vals)
        # start_points = start_points_prepare[sorted_idx_kg[:num_multistart], :]
        start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)
        parallel_results = parallel(delayed(min_negative_ei_func)(pt) for pt in start_points)
    min_neg_ei, point_to_sample = process_parallel_results(parallel_results)
    sample_IS = multifidelity_expected_improvement_evaluator.choose_IS(point_to_sample)
    val = problem.obj_func_min.evaluate(sample_IS, point_to_sample)
    if val < best_sampled_val:
        best_sampled_val = val
        best_sampled_point = point_to_sample
        truth_at_best_sampled = problem.obj_func_min.evaluate(problem.truth_is, point_to_sample)
    gp_dict[sample_IS].add_sampled_points([SamplePoint(point_to_sample, val, noise_and_cost_func(sample_IS, point_to_sample)[0])])
    list_best.append(truth_at_best_sampled)
    total_cost += noise_and_cost_func(sample_IS, point_to_sample)[1]
    list_cost.append(total_cost)
    list_sampled_IS.append(sample_IS)
    list_sampled_points.append(point_to_sample)
    list_sampled_vals.append(val)
    list_noise_var.append(noise_and_cost_func(sample_IS, point_to_sample)[0])
    list_raw_voi.append(-min_neg_ei)

    result_to_pickle = {
        "best": np.array(list_best),
        "cost": np.array(list_cost),
        "sampled_is": np.array(list_sampled_IS),
        "sampled_points": np.array(list_sampled_points),
        "sampled_vals": np.array(list_sampled_vals),
        "sampled_noise_var": np.array(list_noise_var),
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
