import random
import sys
import numpy
import subprocess
import time					# used to measure time

sys.path.append("../")

from assembleToOrder_vanilla import AssembleToOrderVanilla


class AssembleToOrderVar1(AssembleToOrderVanilla):
    def __init__(self, mult=1.0):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing,
        and default is minimizing
        """
        self._dim = 8               # dim of the domain
        self._search_domain = numpy.repeat([[0., 20.]], self._dim, axis=0)
        self._num_IS = 1
        self._mult = mult
        self._func_name = 'atoC_var1'
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG
        self._pathToMATLABScripts = "/fs/europa/g_pf/ATO_scripts/"

        self._meanValue = 0 # subtracted from evaluation to center the objective function at 0
        #TODO this variant has not been centered yet

    def getStringAvgProdTimes(self):
        ### avg production times for items
        apt1 = .2
        apt5 = .28
        string_apt = 'apt1=' + str(apt1) + ';apt5=' + str(apt5) + ';'
        return string_apt

    def getStringItemPrices(self):
        ### prices for items
        pr1 = 1.05
        pr2 = 2
        pr3 = 3.15
        pr5 = 5.25
        string_prices = 'pr1=' + str(pr1) + ';pr2=' + str(pr2) + ';pr3=' + str(pr3) + ';pr5=' + str(pr5) + ';'
        return string_prices