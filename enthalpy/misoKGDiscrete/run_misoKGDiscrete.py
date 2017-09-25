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

'''
Weici's implementation of misoKG for a discrete search space
'''

__author__ = 'jialeiwang'
__author__ = 'matthiaspoloczek'

conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)

# construct problem instance given CMD args
# format: run_mkg.py ${benchmark_name} ${repl_no} ${func_idx}
argv = sys.argv[1:]
if argv[0].find("kg") < 0:
    raise ValueError("benchmark is not mkg/kg!")
problem = identify_problem(argv, bucket)

truth_at_init_best_sampled = problem.obj_func_min.evaluate(problem.truth_is, problem.hist_data.points_sampled[init_best_idx, 1:])
truth_at_best_sampled = truth_at_init_best_sampled
best_mu_star_truth = np.inf
total_cost = 0
for kg_iteration in range(problem.num_iterations):


    #TODO algorithm goes here

    best_this_itr = best_mu_star_truth
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

    print "step {0}, truth_at_best_sampled {1}, best_this_itr {2}, sample_is {3}".format(kg_iteration,
                                                                                         truth_at_best_sampled,
                                                                                         best_this_itr, sample_is)

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
