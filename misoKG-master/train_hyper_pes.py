import boto
from boto.s3.connection import S3Connection
import numpy as np
import sys

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.log_likelihood import GaussianProcessLogMarginalLikelihood

from constants import s3_bucket_name
from data_io import send_data_to_s3
from pes.covariance import ProductKernel
from pes.model import scale_forward
from pes.slice_sampler import SliceSampler
from problems.identifier import identify_problem

__author__ = 'jialeiwang'

conn = S3Connection()
conn = boto.connect_s3()
bucket = conn.get_bucket(s3_bucket_name, validate=True)
# construct problem instance given CMD args
argv = sys.argv[1:]
if argv[0].find("pes") < 0:
    raise ValueError("hyper is not pes/seqpes!")
problem = identify_problem(argv, bucket)
hist_data = problem.hist_data

n_samples = 5
prior_alpha_mean = np.var(hist_data.points_sampled_value)
# Transform data to (0,1)^d space
lower_bounds = problem.obj_func_min._search_domain[:, 0]
upper_bounds = problem.obj_func_min._search_domain[:, 1]
transformed_data = HistoricalData(problem.obj_func_min.getDim() + 1)
for pt, val, var in zip(hist_data.points_sampled, hist_data.points_sampled_value, hist_data.points_sampled_noise_variance):
    transformed_data.append_sample_points([[np.concatenate(([pt[0]],scale_forward(pt[1:], lower_bounds, upper_bounds))), val, var], ])
################################
# slice sampling for pes begins
# construct prior for entries of Cholesky of Kt
num_is = problem.num_is_in
kt_mean = -1. * np.ones(int(num_is * (num_is + 1) / 2))  # the entries are in log space
i = 0
advance = 1
while i < len(kt_mean):
    kt_mean[i] = 0.0  # diagonal entries are set to e^(0.0) = 1
    advance += 1
    i += advance
kt_var = 100. * np.ones(len(kt_mean))
# construct prior for hypers of regular exp kernel
prior_mean_IS_0 = np.concatenate(([prior_alpha_mean], [1.] * problem.obj_func_min.getDim()))
prior_var_IS_0 = np.concatenate(([np.power(prior_alpha_mean / 2., 2.)], [25.] * problem.obj_func_min.getDim()))
prior_mean = np.concatenate((kt_mean, prior_mean_IS_0))
prior_var = np.concatenate((kt_var, prior_var_IS_0))

### setup cov and gp model
cov = ProductKernel(prior_mean, problem.obj_func_min.getDim() + 1, num_is)
gp_likelihood = GaussianProcessLogMarginalLikelihood(cov, transformed_data)
slice_sampler = SliceSampler(prior_mean, prior_var, 1, prior_mean)
hypers_mat = np.array([None] * (n_samples * len(prior_mean))).reshape((n_samples, len(prior_mean)))
for i in range(n_samples):
    print "sample {0}".format(i)
    hypers_mat[i, :] = slice_sampler.sample(gp_likelihood)
    # tmp_result = {
    #     "prior_mean": prior_mean,
    #     "prior_sig": prior_var,
    #     "hyperparam_mat": hypers_mat,
    #     "data_points_sampled": hist_data.points_sampled,
    #     "data_points_sampled_value": hist_data.points_sampled_value,
    # }
final_result = {
    "prior_mean": prior_mean,
    "prior_sig": prior_var,
    "hyperparam": np.mean(hypers_mat, axis=0),
    "hyperparam_mat": hypers_mat,
    "data_points_sampled": hist_data.points_sampled,
    "data_points_sampled_value": hist_data.points_sampled_value,
}
send_data_to_s3(bucket, problem.hyper_path, final_result)
