import numpy
import random

from load_and_store_data import load_data_from_a_min_problem
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain

__author__ = 'matthiaspoloczek'

'''
The Rosenbrock function with some normal noise
'''

class RosenbrockVanilla(object):
    def __init__(self, mult=1.0):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing, and default is minimizing
        """
        self._dim = 2
        self._search_domain = numpy.repeat([[-2., 2.]], 2, axis=0)
        self._num_IS = 1
        self._mult = mult
        self._func_name = 'rbCvanN'
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG

        self._meanValuesIS = [411.7] # subtracted from evaluation to center the objective function at 0

    def computeValue(self, IS, x):
        return pow(1. - x[0], 2.0) + 100. * pow(x[1] - pow(x[0], 2.0), 2.0) - 10.0   # make the minimum -10.0 for now

    def noise_and_cost_func(self, IS, x):
        if IS == 0:
            return 0.25, 1.0
        else:
            raise RuntimeError("illegal IS")

    def ensureBoundaries(self, x):
        '''
        Numerical issues may lead to an x that is not within the search domain. Fix that

        Args:
            x:

        Returns: each value is moved to the closest boundary of that dimension
        '''
        for dim in range(len(x)):
            if x[dim] < self._search_domain[dim][0]:
                x[dim] = self._search_domain[dim][0]
            if x[dim] > self._search_domain[dim][1]:
                x[dim] = self._search_domain[dim][1]
        return x

    def evaluate(self, IS, x):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 2d numpy array
        """
        if IS == 0:
            x = self.ensureBoundaries(x) # address numerical issues arising in EGO
            return self._mult * (self.computeValue(IS, x) - self.getMeanValue(IS)) + 0.5 * numpy.random.normal()
        else:
            raise RuntimeError("illegal IS")

    def drawRandomPoint(self):
        '''

        :return: a random point in the search space
        '''
        #return [self._prg.uniform(0.0,20.0) for i in xrange(self._dim)]
        return [self._prg.uniform( self._search_domain[i][0], self._search_domain[i][1] ) for i in xrange( self._search_domain.shape[0] )]

    def estimateMeanFromPickles(self):
        '''
        Take the samples used to determine the hypers (or the training data) to estimate the mean value

        :return: the mean objective value across the samples (the sign assumes it is a minimization problem)
        '''

        sign_for_maximization_problem = -1.0 # when storing we assume the minimization version of the problem
        # actually, this should be 1.0 for Rosenbrock, but it does not matter since it is centered...

        sampled_values = []

        pathToPickles = "../pickles/csCentered"
        filename_infix = '_IS_0_200_points_each_repl_' # '_IS_0_200_points_each_repl_' # this must be un-normalized data
        for index in range(3): #  range(3)
            filename = self._func_name + filename_infix + str(index)
            print filename
            (_,vals) = load_data_from_a_min_problem(pathToPickles, filename)
            sampled_values.extend(vals[0])
        print 'Taking ' + str(len(sampled_values)) + ' sampled values into account.\n'
        # print vals
        # print sampled_values
        meanValue = sign_for_maximization_problem * numpy.mean(sampled_values)
        print 'Mean = ' + str(meanValue)

        return meanValue

    def getMeanValue(self, IS):
        if IS in self.getList_IS_to_query():
            return self._meanValuesIS[IS]
        else:
            raise RuntimeError("illegal IS")

    def getFuncName(self):
        return self._func_name

    def getTruthIS(self):
        return self._truth_IS

    def getNumIS(self):
        return self._num_IS

    def getDim(self):
        return self._dim

    def getSearchDomain(self):
        return self._search_domain

    def getList_IS_to_query(self):
        return [0]

    def get_moe_domain(self):
        return TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])


        