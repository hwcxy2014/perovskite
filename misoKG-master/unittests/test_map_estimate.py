import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy
from unittest import TestCase

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

from multifidelity_KG.model.covariance_function import MixedSquareExponential
from multifidelity_KG.model.hyperparameter_optimization import NormalPrior, hyper_opt
from unittests.test_util import get_random_gp_data


__author__ = 'jialeiwang'

class TestMAP(object):

    def test_normal_prior(self):
        space_dim = 2
        num_IS = 2
        true_hyper, data = get_random_gp_data(space_dim, num_IS, 500)
        hyperparam_search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in numpy.repeat([[0.01, 2.]], len(true_hyper), axis=0)])
        hyper_bounds = [(0.01, 100.) for i in range(len(true_hyper))]
        multistart_pts = hyperparam_search_domain.generate_uniform_random_points_in_domain(1)
        cov = MixedSquareExponential(hyperparameters=multistart_pts[0,:], total_dim=space_dim+1, num_is=num_IS)
        test_prior = NormalPrior(5.*numpy.ones(len(true_hyper)), 25. * numpy.eye(len(true_hyper)))
        hyper_test, f, output = hyper_opt(cov, data=data, init_hyper=multistart_pts[0, :], hyper_bounds=hyper_bounds, approx_grad=False, hyper_prior=test_prior)

        good_prior = NormalPrior(true_hyper, 0.1 * numpy.eye(len(true_hyper)))
        hyper_good_prior, _, _ = hyper_opt(cov, data=data, init_hyper=multistart_pts[0, :], hyper_bounds=hyper_bounds, approx_grad=False, hyper_prior=good_prior)
        bad_prior = NormalPrior(numpy.ones(len(true_hyper)), 0.1 * numpy.eye(len(true_hyper)))
        hyper_bad_prior, _, _ = hyper_opt(cov, data=data, init_hyper=multistart_pts[0, :], hyper_bounds=hyper_bounds, approx_grad=False, hyper_prior=bad_prior)
        print "true hyper: {0}\n hyper test: {1}\n good prior: {2}\n bad prior:\n should close to one {3}".format(true_hyper, hyper_test, hyper_good_prior, hyper_bad_prior)
        print "dim {0}, num_is {1}".format(space_dim, num_IS)

if __name__ == "__main__":
    test = TestMAP()
    test.test_normal_prior()