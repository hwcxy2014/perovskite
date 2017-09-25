import numpy
import random

from rosenbrock_sinus import RosenbrockSinus

__author__ = 'matthiaspoloczek'

'''
The Rosenbrock function increased everywhere by 0.01 * x[0]
'''

class RosenbrockBiased(RosenbrockSinus):
    def __init__(self, mult=1.0):
        """
        :param num_IS: number of information sources
        :param noise_and_cost_func: f(IS, x) returns (noise_var, cost)
        :param mult: control whether the optimization problem is maximizing or minimizing, and default is minimizing
        """
        self._dim = 2
        self._search_domain = numpy.repeat([[-2., 2.]], 2, axis=0)
        self._num_IS = 1
        self._mult = mult
        self._func_name = 'rbCbiasN'
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG
        self._meanValuesIS = [458.9]

    def computeValue(self, IS, x):
        return super(RosenbrockBiased, self).computeValue(0, x) + 0.01 * x[0]