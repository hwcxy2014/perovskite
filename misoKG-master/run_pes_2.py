import boto
from boto.s3.connection import S3Connection
import numpy
import pickle
import sys

from joblib import Parallel, delayed
from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess as pythonGP

from constants import s3_bucket_name
from data_io import send_data_to_s3, process_parallel_results
from multifidelity_KG.voi.optimization import bfgs_optimization
from pes.covariance import ProductKernel
from pes.model import global_optimization_of_GP, scale_back, scale_forward, PESModel, optimize_entropy
from pes.entropy_search import PES
from problems.identifier import identify_problem
conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)

__author__ = 'jialeiwang'

# construct problem instance given CMD args
# format: run_pes.py ${benchmark_name} ${func_idx} ${repl_no}
argv = sys.argv[1:]
if argv[0].find("pes") < 0:
    raise ValueError("benchmark is not pes!")
problem = identify_problem(argv, bucket)
num_threads = 6
num_multistart = 6

# Transform data to (0,1)^d space
lower_bounds = problem.obj_func_min._search_domain[:, 0]
upper_bounds = problem.obj_func_min._search_domain[:, 1]
unit_domain = TensorProductDomain([ClosedInterval(0.0, 1.0) for _ in range(len(problem.obj_func_min._search_domain))])
transformed_data = HistoricalData(problem.obj_func_min.getDim() + 1)
for pt, val, var in zip(problem.hist_data.points_sampled, problem.hist_data.points_sampled_value, problem.hist_data.points_sampled_noise_variance):
    transformed_data.append_sample_points([[numpy.concatenate(([pt[0]],scale_forward(pt[1:], lower_bounds, upper_bounds))), val, var], ])

# entropy search begins
def noise_func(IS, x):
    return problem.obj_func_min.noise_and_cost_func(IS, x)[0]

def cost_func(IS, x):
    return problem.obj_func_min.noise_and_cost_func(IS, x)[1]

print problem.hyper_param
python_cov = ProductKernel(problem.hyper_param, problem.obj_func_min.getDim()+ 1, problem.num_is_in)
pes_model = PESModel(pythonGP(python_cov, transformed_data), python_cov, noise_func)
pes = PES(num_dims=problem.obj_func_min.getDim())

def find_mu_star(start_pt, gp):
    return bfgs_optimization(start_pt, lambda x: gp.compute_mean_of_points(numpy.concatenate(([0], x)).reshape((1,-1)))[0], problem.obj_func_min._search_domain)

# data containers for pickle storage
list_best = []
list_cost = []
list_sampled_IS = []
list_sampled_points = []
list_sampled_vals = []
list_noise_var = []
list_raw_voi = []
init_best_idx = numpy.argmin(problem.hist_data._points_sampled_value[problem.hist_data._points_sampled[:, 0] == problem.truth_is])
best_sampled_val = problem.hist_data._points_sampled_value[init_best_idx]
truth_at_init_best_sampled = problem.obj_func_min.evaluate(problem.truth_is, problem.hist_data.points_sampled[init_best_idx, 1:])
truth_at_best_sampled = truth_at_init_best_sampled
total_cost = 0.
best_mu_star_truth = numpy.inf
for itr in range(problem.num_iterations):
    print "{0} , itr {1}".format(argv, itr)
    with Parallel(n_jobs=num_threads) as parallel:
        parallel_results = parallel(delayed(find_mu_star)(pt, pes_model.gp_model) for pt in unit_domain.generate_uniform_random_points_in_domain(num_multistart))
        mu_star, mu_star_point = process_parallel_results(parallel_results)
    mu_star_point_org_space = scale_back(mu_star_point, lower_bounds, upper_bounds)
    mu_star_truth = problem.obj_func_min.evaluate(problem.truth_is, mu_star_point_org_space)
    best_mu_star_truth = min(best_mu_star_truth, mu_star_truth)
    pes_model.state = itr
    try:
        pt_to_sample, sample_is, acq, raw_acq = optimize_entropy(pes, pes_model, problem.obj_func_min.getDim(), num_discretization=1000, cost_func=cost_func, list_sample_is=problem.list_sample_is)
    except:
        pt_to_sample = mu_star_point
        sample_is = problem.exploitation_is
        acq = 0.0
        raw_acq = [0.0] * len(problem.list_sample_is)
    pt_to_sample_org_space = scale_back(pt_to_sample, lower_bounds, upper_bounds)
    sample_noise, sample_cost = problem.obj_func_min.noise_and_cost_func(sample_is, None)
    sample_value = problem.obj_func_min.evaluate(sample_is, pt_to_sample_org_space)
    # update best_sampled_val and truth_at_best_sampled
    if sample_value < best_sampled_val:
        best_sampled_val = sample_value
        if(problem.truth_is == sample_is):
            truth_at_best_sampled = sample_value
        else:
            truth_at_best_sampled = problem.obj_func_min.evaluate(problem.truth_is, pt_to_sample_org_space)
    best_this_itr = min(best_mu_star_truth, truth_at_best_sampled)
    pes_model.gp_model.add_sampled_points([SamplePoint(numpy.concatenate(([sample_is],pt_to_sample)), sample_value, noise_variance=sample_noise)])
    total_cost += sample_cost
    list_best.append(best_this_itr)
    list_cost.append(total_cost)
    list_sampled_IS.append(sample_is)
    list_sampled_points.append(pt_to_sample_org_space)
    list_sampled_vals.append(sample_value)
    list_noise_var.append(problem.obj_func_min.noise_and_cost_func(sample_is, None)[0])
    list_raw_voi.append(raw_acq)
    result_to_pickle = {
        "best": numpy.array(list_best),
        "cost": numpy.array(list_cost),
        "sampled_is": numpy.array(list_sampled_IS),
        "sampled_points": numpy.array(list_sampled_points),
        "sampled_vals": numpy.array(list_sampled_vals),
        "sampled_noise_var": numpy.array(list_noise_var),
        "raw_voi": numpy.array(list_raw_voi),
        "init_best_truth": truth_at_init_best_sampled,
    }
    # write data to pickle.
    send_data_to_s3(bucket, problem.result_path, result_to_pickle)
    if problem.data_path is not None:
        data_to_pickle = {"points": numpy.array(list_sampled_points),
                          "vals": numpy.array(list_sampled_vals),
                          "noise": numpy.array(list_noise_var),
                          }
        send_data_to_s3(bucket, problem.data_path, data_to_pickle)
