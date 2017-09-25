import numpy

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess

from multifidelity_KG.model.covariance_function import MixedSquareExponential
from pes.covariance import ProductKernel

__author__ = 'jialeiwang'

def get_random_gp_data(space_dim, num_is, num_data_each_is, kernel_name):
    """ Generate random gp data
    :param space_dim:
    :param num_is:
    :param num_data_each_is:
    :param kernel_name: currently it's either 'mix_exp' or 'prod_ker'
    :return:
    """
    sample_var = 0.01
    if kernel_name == "mix_exp":
        hyper_params = numpy.random.uniform(size=(num_is+1)*(space_dim+1))
        cov = MixedSquareExponential(hyper_params, space_dim+1, num_is)
    elif kernel_name == "prod_ker":
        hyper_params = numpy.random.uniform(size=(num_is+1)*(num_is+2)/2+space_dim+1)
        cov = ProductKernel(hyper_params, space_dim+1, num_is+1)
    else:
        raise NotImplementedError("invalid kernel")
    python_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in numpy.repeat([[-10., 10.]], space_dim+1, axis=0)])
    data = HistoricalData(space_dim+1)
    init_pts = python_search_domain.generate_uniform_random_points_in_domain(2)
    init_pts[:,0] = numpy.zeros(2)
    data.append_historical_data(init_pts, numpy.zeros(2), numpy.ones(2) * sample_var)
    gp = GaussianProcess(cov, data)
    points = python_search_domain.generate_uniform_random_points_in_domain(num_data_each_is)
    for pt in points:
        for i in range(num_is):
            pt[0] = i
            val = gp.sample_point_from_gp(pt, sample_var)
            data.append_sample_points([[pt, val, sample_var], ])
            gp = GaussianProcess(cov, data)
    return hyper_params, data
