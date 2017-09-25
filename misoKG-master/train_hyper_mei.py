import boto
from boto.s3.connection import S3Connection
import numpy as np
import sys

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
# from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential

from constants import s3_bucket_name
from data_io import send_data_to_s3
from multifidelity_KG.model.hyperparameter_optimization import NormalPrior, hyper_opt
from problems.identifier import identify_problem

__author__ = 'jialeiwang'
conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)
# construct problem instance given CMD args
argv = sys.argv[1:]
if argv[0].find("mei") < 0: # MP: I have added "<0"
    raise ValueError("hyper is not mei!")
problem = identify_problem(argv, bucket)
hist_data = problem.hist_data
prior_mean_dict = dict()
prior_sig_dict = dict()
hyper_bounds_dict = dict()
loglikelihood_dict = dict()
data_points_sampled_dict = dict()
data_points_sampled_value_dict = dict()
hyperparam_dict = dict()
# Setup prior for MAP
num_hyper_multistart = 5
for IS in hist_data:
    prior_mean = np.concatenate(([max(0.01, np.var(hist_data[IS].points_sampled_value))], [(d[1]-d[0]) for d in problem.obj_func_min._search_domain]))
    prior_sig = np.diag(np.power(prior_mean/2., 2.0))
    prior = NormalPrior(prior_mean, prior_sig)
    hyper_bounds = [(0.001, prior_mean[i]+2.*np.sqrt(prior_sig[i,i])) for i in range(problem.obj_func_min.getDim()+1)]
    hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in hyper_bounds])
    multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(num_hyper_multistart)
    best_f = np.inf
    best_hyper = None
    cov = SquareExponential(prior_mean)
    for i in range(num_hyper_multistart):
        ### MP: Should one add [IS} to data=hist_data , in order to solve the error?
        hyper, f, output = hyper_opt(cov, data=hist_data[IS], init_hyper=multistart_pts[i, :],
                                     hyper_bounds=hyper_bounds, approx_grad=False, hyper_prior=prior)
        # print output
        if f < best_f:
            best_hyper = hyper
            best_f = f
    prior_mean_dict[IS] = prior_mean
    prior_sig_dict[IS] = np.diag(prior_sig) # MP: bugfix applied
    hyper_bounds_dict[IS] = hyper_bounds
    loglikelihood_dict[IS] = -best_f
    data_points_sampled_dict[IS] = hist_data[IS].points_sampled
    data_points_sampled_value_dict[IS] = hist_data[IS].points_sampled_value
    hyperparam_dict[IS] = best_hyper
result = {
    "prior_mean": prior_mean_dict,
    "prior_sig": prior_sig_dict,
    "hyper_bounds": hyper_bounds_dict,
    "hyperparam": hyperparam_dict,
    "loglikelihood": loglikelihood_dict,
    "data_points_sampled": data_points_sampled_dict,
    "data_points_sampled_value": data_points_sampled_value_dict,
}
send_data_to_s3(bucket, problem.hyper_path, result)
