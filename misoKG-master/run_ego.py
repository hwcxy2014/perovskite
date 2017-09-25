import boto
from boto.s3.connection import S3Connection
import numpy as np
# import pickle
from joblib import Parallel, delayed
import sys

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.expected_improvement import ExpectedImprovement

from constants import s3_bucket_name
from data_io import process_parallel_results, send_data_to_s3
from multifidelity_KG.voi.optimization import *
from problems.identifier import identify_problem
conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)

# construct problem instance given CMD args
argv = sys.argv[1:]
if argv[0].find("ego") < 0:
    raise ValueError("benchmark is not ego!")
problem = identify_problem(argv, bucket)

# algorithm params
num_threads = 32
num_multistart = 32

# ego begins
cov = SquareExponential(problem.hyper_param)
ego_gp = GaussianProcess(cov, problem.hist_data)
# data containers for pickle storage
list_best = []
list_cost = []
list_sampled_IS = []
list_sampled_points = []
list_sampled_vals = []
list_noise_var = []
list_raw_voi = []
best_sampled_val = np.amin(problem.hist_data._points_sampled_value)
truth_at_init_best_sampled = problem.obj_func_min.evaluate(problem.truth_is, problem.hist_data.points_sampled[np.argmin(problem.hist_data._points_sampled_value), :])
truth_at_best_sampled = truth_at_init_best_sampled
total_cost = 0.0
for ego_n in range(problem.num_iterations):
    expected_improvement_evaluator = ExpectedImprovement(ego_gp)
    min_negative_ei = np.inf

    def negative_ego_func(x):
        expected_improvement_evaluator.set_current_point(x.reshape((1, -1)))
        return -1.0 * expected_improvement_evaluator.compute_expected_improvement()

    def negative_ego_grad_func(x):
        expected_improvement_evaluator.set_current_point(x.reshape((1, -1)))
        return -1.0 * expected_improvement_evaluator.compute_grad_expected_improvement()[0, :]

    def min_negative_ego_func(start_point):
        return bfgs_optimization_grad(start_point, negative_ego_func, negative_ego_grad_func, problem.obj_func_min._search_domain)

    with Parallel(n_jobs=num_threads) as parallel:
        parallel_results = parallel(delayed(min_negative_ego_func)(pt) for pt in problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_multistart))
    min_neg_ei, point_to_sample = process_parallel_results(parallel_results)
    min_eval_this_itr = np.inf
    min_eval_is_this_itr = None
    for IS in problem.list_sample_is:
        list_sampled_IS.append(IS)
        val = problem.obj_func_min.evaluate(IS, point_to_sample)
        if val < min_eval_this_itr:
            min_eval_this_itr = val
            min_eval_is_this_itr = IS
        ego_gp.add_sampled_points([SamplePoint(point_to_sample, val, problem.obj_func_min.noise_and_cost_func(IS, point_to_sample)[0])])
        total_cost += problem.obj_func_min.noise_and_cost_func(IS, point_to_sample)[1]
    if min_eval_this_itr < best_sampled_val:
        best_sampled_val = min_eval_this_itr
        best_sampled_point = point_to_sample
        truth_at_best_sampled = problem.obj_func_min.evaluate(problem.truth_is, point_to_sample)
    # save data from this iteration:
    list_best.append(truth_at_best_sampled)
    list_cost.append(total_cost)
    list_sampled_points.append(point_to_sample)
    list_sampled_vals.append(min_eval_this_itr)
    list_noise_var.append(problem.obj_func_min.noise_and_cost_func(min_eval_is_this_itr, point_to_sample)[0])
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

    # write data to s3.
    send_data_to_s3(bucket, problem.result_path, result_to_pickle)
    if problem.data_path is not None:
        data_to_pickle = {
            "points": np.array(list_sampled_points),
            "vals": np.array(list_sampled_vals),
            "noise": np.array(list_noise_var),
        }
        send_data_to_s3(bucket, problem.data_path, data_to_pickle)
