__author__ = 'jialeiwang'

import numpy
import scipy
# from moe.optimal_learning.python.python_version.log_likelihood import GaussianProcessLogMarginalLikelihood
from moe.optimal_learning.python.cpp_wrappers.log_likelihood import GaussianProcessLogLikelihood

from covariance_function import MixedSquareExponential


class NormalPrior(object):

    def __init__(self, mu, sig):
        self._mu = mu
        self._sig_inv = numpy.linalg.inv(sig)

    def compute_log_likelihood(self, x):
        x_mu = (x - self._mu).reshape((-1, 1))
        return -0.5 * numpy.dot(x_mu.T, numpy.dot(self._sig_inv, x_mu))

    def compute_grad_log_likelihood(self, x):
        x_mu = (x - self._mu).reshape((-1, 1))
        return -0.5 * numpy.dot(self._sig_inv + self._sig_inv.T, x_mu).flatten()


def hyper_opt(cov, data, init_hyper, hyper_bounds, approx_grad, hyper_prior=None):
    """Hyperparameter optimization
    ":param cov: covariance
    :param data: instance of HistoricalData
    :param init_hyper: starting point of hyperparameter
    :param hyper_bounds: list of (lower_bound, upper_bound)
    :param approx_grad: bool
    :return: (optimial hyper, optimal marginal loglikelihood, function output)
    """
    # gp_likelihood = GaussianProcessLogMarginalLikelihood(cov, data)
    gp_likelihood = GaussianProcessLogLikelihood(cov, data)

    if hyper_prior is not None:
        def obj_func(x):
            gp_likelihood.set_hyperparameters(x)
            return -1.0 * (gp_likelihood.compute_log_likelihood() + hyper_prior.compute_log_likelihood(x))

        def grad_func(x):
            gp_likelihood.set_hyperparameters(x)
            return -1.0 * (gp_likelihood.compute_grad_log_likelihood() + hyper_prior.compute_grad_log_likelihood(x))
    else:
        def obj_func(x):
            gp_likelihood.set_hyperparameters(x)
            return -1.0 * gp_likelihood.compute_log_likelihood()

        def grad_func(x):
            gp_likelihood.set_hyperparameters(x)
            return -1.0 * gp_likelihood.compute_grad_log_likelihood()

    return scipy.optimize.fmin_l_bfgs_b(func=obj_func, x0=init_hyper, fprime=grad_func, args=(), approx_grad=approx_grad,
                                        bounds=hyper_bounds, m=10, factr=10.0, pgtol=0.01,
                                        epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=100, disp=1, callback=None)

# if __name__ == "__main__":
    # random_gp = RandomGP(dim=3, num_is=2, hyper_params=numpy.arange(1,10)*0.1)
    # print "start generating GP..."
    # data = random_gp.generate_data(100)
    # x, f, output = hyper_opt(3, 2, data, numpy.ones(9))
