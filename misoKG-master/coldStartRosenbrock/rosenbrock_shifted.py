import numpy
import random

from rosenbrock_biased import RosenbrockBiased

__author__ = 'matthiaspoloczek'

'''
The biased Rosenbrock function with x1 increased by 0.1 and x2 increased by 0.05
'''

class RosenbrockShifted(RosenbrockBiased):
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
        self._func_name = 'rbC_shiftedN'
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG
        self._meanValuesIS = [462.0]

    def computeValue(self, IS, x):
        x[0] += 0.1
        x[1] += 0.05
        return super(RosenbrockShifted,self).computeValue(0,x)