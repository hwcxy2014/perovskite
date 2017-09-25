import numpy

from multifidelity_KG.model.covariance_function import MixedSquareExponential

__author__ = 'jialeiwang'

""" One test case is let hyperparam = [1, 2, 3, 4, 5, 6, 7, 8, 9], search space is 2d and 2 information sources.
For two points in the search space: (1,1) and (2,3)
"""
hypers = numpy.arange(1, 10)
square_exponential = MixedSquareExponential(hyperparameters=hypers, total_dim=3, num_is=2)

def grad_approx(pt_1, pt_2, delta, num_hypers):
    grad = numpy.empty(num_hypers)
    for i in range(num_hypers):
        hypers_plus = numpy.arange(1, 10)
        hypers_plus[i] += 0.5 * delta
        hypers_minus = numpy.arange(1, 10)
        hypers_minus[i] -= 0.5 * delta
        square_exponential_plus = MixedSquareExponential(hyperparameters=hypers_plus, total_dim=3, num_is=2)
        square_exponential_minus = MixedSquareExponential(hyperparameters=hypers_minus, total_dim=3, num_is=2)
        grad[i] = (square_exponential_plus.covariance(pt_1, pt_2) - square_exponential_minus.covariance(pt_1, pt_2)) / delta
    return grad

def test_cov_same_is():
    pt_1 = numpy.array([1, 1, 1])
    pt_2 = numpy.array([1, 2, 3])
    hand_compute = numpy.exp(-0.5*(0.5*0.5+numpy.power(2./3., 2.))) + 4. * numpy.exp(-0.5*(0.2*0.2+numpy.power(2./6., 2.)))
    return numpy.abs(hand_compute - square_exponential.covariance(pt_1, pt_2)) < 1e-8

def test_cov_diff_is():
    pt_1 = numpy.array([1, 1, 1])
    pt_2 = numpy.array([2, 2, 3])
    hand_compute = numpy.exp(-0.5*(0.5*0.5+numpy.power(2./3., 2.)))
    return numpy.abs(hand_compute - square_exponential.covariance(pt_1, pt_2)) < 1e-8

def test_grad_cov_same_is():
    pt_1 = numpy.array([1, 1, 1])
    pt_2 = numpy.array([1, 2, 3])
    grad_hand = numpy.zeros(9)
    zeroth_exp = square_exponential.covariance(numpy.array([1,1,1]), numpy.array([2,2,3])) / hypers[0]
    lth_exp = (square_exponential.covariance(pt_1, pt_2) - zeroth_exp) / hypers[3]
    grad_hand[0] = zeroth_exp
    grad_hand[1] = 1. / numpy.power(hypers[1], 3.) * hypers[0] * zeroth_exp
    grad_hand[2] = 4. / numpy.power(hypers[2], 3.) * hypers[0] * zeroth_exp
    grad_hand[3] = lth_exp
    grad_hand[4] = 1. / numpy.power(hypers[4], 3.) * hypers[3] * lth_exp
    grad_hand[5] = 4. / numpy.power(hypers[5], 3.) * hypers[3] * lth_exp
    return numpy.abs(grad_hand - square_exponential.hyperparameter_grad_covariance(pt_1, pt_2)).sum() < 1e-8

def test_grad_cov_diff_is():
    pt_1 = numpy.array([1, 1, 1])
    pt_2 = numpy.array([2, 2, 3])
    grad_hand = numpy.zeros(9)
    zeroth_exp = square_exponential.covariance(numpy.array([1,1,1]), numpy.array([2,2,3])) / hypers[0]
    grad_hand[0] = zeroth_exp
    grad_hand[1] = 1. / numpy.power(hypers[1], 3.) * hypers[0] * zeroth_exp
    grad_hand[2] = 4. / numpy.power(hypers[2], 3.) * hypers[0] * zeroth_exp
    return numpy.abs(grad_hand - square_exponential.hyperparameter_grad_covariance(pt_1, pt_2)).sum() < 1e-8


if __name__ == "__main__":
    print test_cov_same_is()
    print test_cov_diff_is()
    print test_grad_cov_same_is()
    print test_grad_cov_diff_is()
