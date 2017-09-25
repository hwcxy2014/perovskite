import random
import sys
import numpy
import subprocess
import time					# used to measure time

sys.path.append("../")

from assembleToOrder_vanilla import AssembleToOrderVanilla


class AssembleToOrderVar3(AssembleToOrderVanilla):
    def __init__(self, mult=1.0):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing,
        and default is minimizing
        """
        self._dim = 8               # dim of the domain
        self._search_domain = numpy.repeat([[0., 20.]], self._dim, axis=0)
        self._num_IS = 1
        self._mult = mult
        self._func_name = 'atoC_var3'
        self._list_IS_to_query = [0]
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG
        self._pathToMATLABScripts = "/fs/europa/g_pf/ATO_scripts/"

        self._meanValue = 17.242  # 0 # subtracted from evaluation to center the objective function at 0
        # from 50 samples of 20 points each: -17.1856366812

    def getStringItemPrices(self):
        ### prices for items
        pr1 = 1 + 0.02 + 0.01
        pr2 = 2 + 0.02 + 0.
        pr3 = 3 + 0.06 - 0.02 + 0.03
        pr5 = 5 + 0.05 - 0.02 + 0.04
        string_prices = 'pr1=' + str(pr1) + ';pr2=' + str(pr2) + ';pr3=' + str(pr3) + ';pr5=' + str(pr5) + ';'
        return string_prices

    def getStringArrivalRates(self):
        ### arrival rates for products control the demand
        arrivalratep1 = 3.6 - 0.1 + 0.15
        arrivalratep2 = 3 - 0.15 + 0.1
        arrivalratep3 = 2.4 + 0.05 + 0.05
        arrivalratep4 = 1.8 + 0.01
        arrivalratep5 = 1.2 + 0.
        string_arrival_rates = 'ar1=' + str(arrivalratep1) + ';ar2=' + str(arrivalratep2) + \
                               ';ar3=' + str(arrivalratep3) + ';ar4=' + str(arrivalratep4) \
                               + ';ar5=' + str(arrivalratep5) + ';'
        return string_arrival_rates

    def getStringAvgProdTimes(self):
        ### avg production times for items
        apt1 = .15 + 0.005
        apt5 = .25 + 0.005
        string_apt = 'apt1=' + str(apt1) + ';apt5=' + str(apt5) + ';'
        return string_apt


    def noise_and_cost_func(self, IS, x):
        if IS == 0:
            return 0.85, (11.422 - 7.5) # 0.85 est from [ 1.48903858  0.12032679  1.05742684  0.50622729  1.07972096]

        else:
            raise RuntimeError("illegal IS")