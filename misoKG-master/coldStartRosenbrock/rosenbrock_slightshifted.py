import numpy
import random

from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla



__author__ = 'matthiaspoloczek'

'''
The vanilla Rosenbrock function with x1 increased by 0.01 and x2 decreased by 0.005
'''


class RosenbrockSlightShifted(RosenbrockVanilla):
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
        self._func_name = 'rbC_slshN'
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG
        self._meanValuesIS = [456.9]

    def computeValue(self, IS, x):
        x[0] += 0.01
        x[1] += - 0.005
        return super(RosenbrockSlightShifted,self).computeValue(0,x)