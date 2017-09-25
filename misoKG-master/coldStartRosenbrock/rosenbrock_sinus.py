import numpy
import random

from rosenbrock_vanilla import RosenbrockVanilla

__author__ = 'matthiaspoloczek'

'''
The Rosenbrock function with additional sin term (Lam et al)
'''

class RosenbrockSinus(RosenbrockVanilla):
    def __init__(self, mult=1.0):
        """
        :param num_IS: number of information sources
        :param noise_and_cost_func: f(IS, x) returns (noise_var, cost)
        :param mult: control whether the optimization problem is maximizing or minimizing, and default is minimizing
        """

        super(RosenbrockSinus, self).__init__(mult=1.0)

        self._dim = 2
        self._search_domain = numpy.repeat([[-2., 2.]], 2, axis=0)
        self._num_IS = 1
        self._mult = mult
        self._func_name = 'rbCsinN'
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG
        self._meanValuesIS = [423.9]

    def computeValue(self, IS, x):
        return super(RosenbrockSinus,self).computeValue(0,x) + 0.01 * numpy.sin(10.0 * x[0] + 5.0 * x[1])



