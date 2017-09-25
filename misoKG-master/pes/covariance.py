import numpy
from moe.optimal_learning.python.interfaces.covariance_interface import CovarianceInterface

__author__ = 'jialeiwang'

class ProductKernel(CovarianceInterface):

    def __init__(self, hyperparameters, total_dim, num_is):
        r"""Construct a Tensor product kernel

        :param hyperparameters: hyperparameters of the covariance function; first num_is*(num_is+1)/2 elements are log
        of lower triangular entries in Cholesky of cov matrix for outputs
        :type hyperparameters: array-like of size num_is*(num_is+1)/2 + total_dim

        """
        self._dim = total_dim         # dimension of IS \times search space
        self._num_is = num_is   # Number of information sources
        self.set_hyperparameters(hyperparameters)

    @property
    def num_hyperparameters(self):
        """Return the number of hyperparameters of this covariance function."""
        return self._hyperparameters.size

    def get_hyperparameters(self):
        """Get the hyperparameters (array of float64 with shape (num_hyperparameters)) of this covariance."""
        return numpy.copy(self._hyperparameters)

    def set_hyperparameters(self, hyperparameters):
        """Set hyperparameters to the specified hyperparameters; ordering must match."""
        self._hyperparameters = numpy.array(hyperparameters, dtype=numpy.float, copy=True)
        L = numpy.zeros((self._num_is, self._num_is))
        count = 0
        for i in range(self._num_is):
            for j in range(i+1):
                L[i, j] = numpy.exp(self._hyperparameters[count])
                count += 1
        self._kt = numpy.dot(L, L.T)
        self._alpha = self._hyperparameters[count]
        self._lengths_sq = numpy.power(self._hyperparameters[(count+1):], 2.0)


    hyperparameters = property(get_hyperparameters, set_hyperparameters)

    def covariance(self, point_one, point_two):
        return self._alpha * numpy.exp(-0.5 * numpy.divide(numpy.power(point_two[1:] - point_one[1:], 2.0), self._lengths_sq).sum()) * self._kt[int(point_one[0]), int(point_two[0])]

    def grad_covariance(self, point_one, point_two):
        grad_cov = point_two[1:] - point_one[1:]
        grad_cov /= self._lengths_sq
        grad_cov *= self.covariance(point_one, point_two)
        return numpy.concatenate([[0],grad_cov])

    def hyperparameter_grad_covariance(self, point_one, point_two):
        raise NotImplementedError("hyper grad cov not implemented for ProductKernel")

    def hyperparameter_hessian_covariance(self, point_one, point_two):
        r"""Compute the hessian of self.covariance(point_one, point_two) with respect to its hyperparameters.

        TODO(GH-57): Implement Hessians in Python.

        """
        raise NotImplementedError("Python implementation does not support computing the hessian covariance wrt hyperparameters.")

    def cross_cov(self, X, x_star):
        n, d = X.shape
        l, d1 = x_star.shape
        assert d == d1, "dim in X differs from x_star"
        mat = numpy.zeros((n,l))
        for i in range(n):
            for j in range(l):
                mat[i,j] = self.covariance(X[i, :], x_star[j,:])
        return mat

    def self_var(self, x_star):
        return numpy.array([self.covariance(x,x) for x in x_star])

    def cov(self, X):
        n_data = len(X)
        cov_mat = numpy.zeros((n_data, n_data))
        for i in range(n_data):
            for j in range(i+1):
                cov_mat[i,j] = self.covariance(X[i,:], X[j,:])
                cov_mat[j,i] = cov_mat[i,j]
        return cov_mat

    @property
    def alpha(self):
        return self._alpha

    @property
    def length_scale(self):
        return numpy.sqrt(numpy.array(self._lengths_sq))
