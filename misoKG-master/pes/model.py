import numpy as np
import scipy


from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

__author__ = 'jialeiwang'

""" This class is created to apapt to requirements in entropy search code from Spearmint
"""

class PESModel(object):

    def __init__(self, gp_model, python_cov, noise_func):
        self.state = None   # this controls whether new EP should compute, in the same itr when model is not updated,
                            # EP does not have to re-compute, so it make sense to set state to itr no.
        self.options = {'caching': True,
                        'kernel': 'SquaredExp'}
        self.gp_model = gp_model
        self.num_dims = self.gp_model.dim-1
        self.noiseless_kernel = python_cov
        self.params = {'amp2': python_cov.alpha,
                       'ls': python_cov.length_scale}
        self.noise_func = noise_func
        self.has_data = True

    def predict(self, X, full_cov=False):
        means = self.gp_model.compute_mean_of_points(X)
        vars = self.gp_model.compute_variance_of_points(X)
        if full_cov:
            return means, vars
        else:
            return means, np.diag(vars)

    def jitter_value(self):
        return 1e-16

    def noise_value(self, which_is):
        return self.noise_func(int(which_is), None)

    @property
    def observed_values(self):
        return self.gp_model._historical_data.points_sampled_value

    @property
    def observed_inputs(self):
        return self.gp_model._historical_data.points_sampled



def global_optimization_of_GP(gp_model, bounds, num_multistart, minimization=True):
    """
    :param gp_model:
    :param bounds: list of (min, max) tuples
    :param num_multistart:
    :param minimization:
    :return: shape(space_dim+1,), best x and first entry is always zero because we assume IS0 is truth IS
    """
    sgn = 1 if minimization else -1
    fcn = lambda x: gp_model.compute_mean_of_points(np.concatenate([[0],x]).reshape((1,-1)))[0] * sgn
    grad = lambda x: gp_model.compute_grad_mean_of_points(np.concatenate([[0],x]).reshape((1, -1)), num_derivatives=1)[0, 1:] * sgn
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in bounds])
    start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)
    min_fcn = np.inf
    for start_pt in start_points:
        result_x, result_f, output = scipy.optimize.fmin_l_bfgs_b(func=fcn, x0=start_pt, fprime=grad, args=(), approx_grad=False,
                                                                  bounds=bounds, m=10, factr=10.0, pgtol=1e-10,
                                                                  epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=200, disp=0, callback=None)
        if result_f < min_fcn:
            min_fcn = result_f
            ret = result_x
    print "found GP min {0}".format(min_fcn)
    return np.concatenate([[0], ret]).reshape((1,-1))

def scale_forward(vec, lower_bounds, upper_bounds):
    return np.divide(vec-lower_bounds, upper_bounds-lower_bounds)

def scale_back(vec, lower_bounds, upper_bounds):
    return np.multiply(vec, upper_bounds-lower_bounds)+lower_bounds

def optimize_entropy(pes, pes_model, space_dim, num_discretization, cost_func, list_sample_is, bounds=None):
    if not bounds:
        bounds = [(0.,1.)]*space_dim
    # fcn = lambda x: np.mean([pes.acquisition({'obj': pes_model}, {}, np.concatenate([[which_is], x]), current_best=None, compute_grad=False)[0,0] for pes_model in pes_model_list]) * -1. / cost
    # search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in bounds])
    # start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)
    # min_fcn = np.inf
    # for start_pt in start_points:
    #     result_x, result_f, output = scipy.optimize.fmin_l_bfgs_b(func=fcn, x0=start_pt, fprime=None, args=(), approx_grad=True,
    #                                                               bounds=bounds, m=10, factr=10.0, pgtol=1e-10,
    #                                                               epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=200, disp=0, callback=None)
    #     if result_f < min_fcn:
    #         min_fcn = result_f
    #         ret = result_x
    # return np.concatenate([[which_is], ret]), -min_fcn


    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in bounds])
    points = search_domain.generate_uniform_random_points_in_domain(num_discretization)
    raw_acq = []    # for tuning costs
    best_acq = -np.inf
    for which_is in list_sample_is:
        acq_list = pes.acquisition({'obj': pes_model}, {}, np.hstack((np.ones((num_discretization,1))*which_is, points)), current_best=None, compute_grad=False) / cost_func(which_is, None)
        inner_best_idx = np.argmax(acq_list)
        raw_acq.append(acq_list[inner_best_idx] * cost_func(which_is, None))
        if acq_list[inner_best_idx] > best_acq:
            best_acq = acq_list[inner_best_idx]
            best_is = which_is
            best_idx = inner_best_idx
    return points[best_idx, :], best_is, best_acq, raw_acq
