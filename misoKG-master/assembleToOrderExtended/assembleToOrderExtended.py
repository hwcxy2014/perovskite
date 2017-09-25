import pickle

import numpy

import subprocess
import time					# used to measure time
import random

from assembleToOrder.assembleToOrder import AssembleToOrder
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain

__author__ = 'matthiaspoloczek'

'''
The class for the Extended ATO Simulator.

This version uses both ATO simulators from Xie et al and the Hong and Nelson
'''


class AssembleToOrderExtended(AssembleToOrder):
    def __init__(self, mult=1.0):
        """
        :param numIS: number of information sources
        :param noise_and_cost_func: f(IS, x) returns (noise_var, cost)
        """
        self._dim = 8               # dim of the domain
        self._search_domain = numpy.repeat([[0., 20.]], self._dim, axis=0)
        self._num_IS = 3
        self._truth_IS = 0
        self._numAvailIS = 3        # how many IS are implemented?
        self._list_IS_to_query = [0, 1, 2]

        self._pathToMATLABScripts = "/fs/europa/g_pf/ATO_scripts/"

        # the setup estimated from the data that I used for experiments
        self._descriptionIS = numpy.array([
                                    [500, 0.056, (24.633 - 7.5)], # IS0, this is IS3 in paper, the truth IS
                                    [10, 2.944, (8.064 - 7.5)], # IS1
                                    #[50, 3., (13. - 7.5)], # etc
                                    [100, 0.332, (11.422 - 7.5)]  # IS2
                                    # [1000, 0.027, (41.327 - 7.5)],
                                    #[10000, 0.01, (400. - 7.5)]
                                   # [10, 2.485, 8.887], # for the first IS: runlength, variance,  computational cost in sec per call
                                   # #[50, 3., (13. - 7.5)], # etc
                                   # [100, 0.332, 11.422],
                                   # [500, 0.056, 24.633],
                                   # [1000, 0.027, 41.327],
                                           ])
        self._descriptionIS_index_variance = 1  # the value at index 0 is the variance
        self._descriptionIS_index_cost = 2      # the value at index 1 is the cost

        # # # corrected for constant offset for starting MATLAB:
        # self._descriptionIS = numpy.array([[-1,-1.,-1.], # corresponds to no IS
        #                             [10, 2.485, (8.887 - 7.5)], # for the first IS: runlength, variance,  computational cost in sec per call
        #                             #[50, 3., (13. - 7.5)], # etc
        #                             [100, 0.332, (11.422 - 7.5)],
        #                             [500, 0.056, (24.633 - 7.5)],
        #                             [1000, 0.027, (41.327 - 7.5)],
        #                             #[10000, 0.01, (400. - 7.5)]
        #                                    ])

        self._mult = mult           # control whether the optimization problem is maximizing or minimizing. The default (1.0) is to maximize, set to -1.0 to minimize
        self._prg = random.Random() # make sequential calls to the same PRG

        self._func_name = 'atoext'
        self._meanValuesIS = [17.049, 17.049, 17.049]
        # IS 2 and 3 are centered at 0, IS 1 may have a bias (which is not corrected here)


    def evaluate(self, IS, x):
        """
        # Run the ATO simulator
        # b_vector is currently a string, but you can adapt it to take whatever type of array you use
        # simulation_length is a positive int that gives the length of the simulation
        # random_seed should be a random int larger zero
        # return the mean, (the variance, and the elapsed time)
        :param IS: index of information source, 1, ..., M
        :param x: 8d numpy array
        :return: the obj value at x estimated by info source IS
        """

        random_seed = self._prg.randint(1,100000) 			# a random int that serves as seed for matlab
        self.ensureBoundaries(x)

        fn = -1.0
        FnVar = -1.0
        elapsed_time = 0.0

        runcmd = "b=" + self.convertXtoString(x) + ";length=" + str(self.getRunlength(IS)) + ";seed=" + str(
            random_seed)
        # print "runcmd="+runcmd

        if IS == 1:
            runcmd += ";run(\'" + self._pathToMATLABScripts + "ATOHongNelson_run.m\');exit;"
        else:
            runcmd += ";run(\'" + self._pathToMATLABScripts + "ATO_run.m\');exit;"

        try:
            start_time = time.time()
            # /usr/local/matlab/2015b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('ATO_run.m');exit;"
            # https://www.mathworks.com/matlabcentral/answers/97204-how-can-i-pass-input-parameters-when-running-matlab-in-batch-mode-in-windows
            stdout = subprocess.check_output(["/usr/local/matlab/2015b/bin/matlab", "-nodisplay", "-nojvm",
                                              "-nosplash", "-nodesktop", "-r", runcmd])
            elapsed_time = time.time() - start_time

            posfn = stdout.find("fn=") + 3
            posFnVar = stdout.find("FnVar=") + 6
            if ((posfn > 2) and (posFnVar > 5)):
                posfnEnd = stdout.find("\n",posfn)
                posFnVarEnd = stdout.find("\n",posFnVar)
                fn = stdout[posfn:posfnEnd]
                FnVar = stdout[posFnVar:posFnVarEnd]
        except subprocess.CalledProcessError, e:
            elapsed_time = time.time() - start_time

        # elapsed_time is a reasonable cost
        # fn and FnVar are the results of interest
        #print "runlength="+str(self.getRunlength(IS))+", x="+self.convertXtoString(x)+", fn="+str(fn)+" , FnVar="+str(FnVar)+" , elapsed_time="+str(elapsed_time)
        # for presentation, can be removed

        # print 'evaluate(): ' + str(self._mult * (float(fn) - self.getMeanValue(IS)))

        return self._mult * (float(fn) - self.getMeanValue(IS))   # return only the mean # self._mult * float(fn)


    # some getters for attributes
    def getFuncName(self):
        return self._func_name

    def getMeanValue(self, IS):
        if IS in self.getList_IS_to_query():
            return self._meanValuesIS[IS]
        else:
            raise RuntimeError("illegal IS")

    def getList_IS_to_query(self):
        return self._list_IS_to_query

    def get_moe_domain(self):
        return TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])


