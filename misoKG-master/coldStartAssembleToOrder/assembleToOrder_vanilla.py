import random
import sys
import numpy
import subprocess
import time					# used to measure time

sys.path.append("../")

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain

from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla


class AssembleToOrderVanilla(RosenbrockVanilla):
    def __init__(self, mult=1.0):
        """
        :param mult: control whether the optimization problem is maximizing or minimizing,
        and default is minimizing
        """
        self._dim = 8               # dim of the domain
        self._search_domain = numpy.repeat([[0., 20.]], self._dim, axis=0)
        self._num_IS = 1
        self._mult = mult
        self._func_name = 'atoC_vanilla'
        self._list_IS_to_query = [0]
        self._truth_IS = 0
        self._prg = random.Random() # make sequential calls to the same PRG
        self._pathToMATLABScripts = "/fs/europa/g_pf/ATO_scripts/"

        self._meanValue = 17.049 # 0 # subtracted from evaluation to center the objective function at 0
        # from 50 samples of 20 points each: -16.2303556018

    def noise_and_cost_func(self, IS, x):
        if IS == 0:
            return 0.332, (11.422 - 7.5) # taken from my estimation for the MISO paper
            # 0.81 from [ 1.33379224  0.75715944  0.59425036  0.39952839  0.97227596]
        else:
            raise RuntimeError("illegal IS")

    def convertXtoString(self, x):
        '''
        Matlab style string representation of x
        :param x: design point
        :return: Matlab style string representation of x
        '''
        bVector = "["
        for i in xrange(self._dim):
            bVector += str(x[i]) + " "
        bVector += "]"
        return bVector

    def computeValue(self, IS, x):
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

        random_seed = self._prg.randint(1,10000) 			# a random int that serves as seed for matlab
        num_replications = 100

        string_arrival_rates = self.getStringArrivalRates()

        string_prices = self.getStringItemPrices()

        string_hc = self.getStringHoldingCosts()

        string_apt = self.getStringAvgProdTimes()

        fn = -1.0
        FnVar = -1.0
        elapsed_time = 0.0

        try:
            start_time = time.time()
            # /usr/local/matlab/2015b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('ATO_run.m');exit;"
            # https://www.mathworks.com/matlabcentral/answers/97204-how-can-i-pass-input-parameters-when-running-matlab-in-batch-mode-in-windows

            runcmd = "b="+self.convertXtoString(x)+";length="+str(num_replications)+";seed="+str(random_seed)\
                     +";"+string_arrival_rates+string_prices+string_hc+string_apt+";run(\'" + self._pathToMATLABScripts + "csATO_run.m\');exit;"
            #print "runcmd="+runcmd
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
            print 'Error in evaluate(0,x) for x = ' + str(x)

        # elapsed_time is a reasonable cost
        # fn and FnVar are the results of interest
        #print "runlength="+str(self.getRunlength(IS))+", x="+self.convertXtoString(x)+", fn="+str(fn)+" , FnVar="+str(FnVar)+" , elapsed_time="+str(elapsed_time)
        # for presentation, can be removed

        return float(fn)   # return only the mean

    def evaluate(self, IS, x):
        """ Global optimum is 0 at (1, 1)
        :param IS: index of information source, 1, ..., M
        :param x[2]: 8D numpy array
        """
        if IS == 0:
            x = self.ensureBoundaries(x) # address numerical issues arising in EGO
            return self._mult * (self.computeValue(IS, x) - self._meanValue)
        else:
            raise RuntimeError("illegal IS")

    def getStringAvgProdTimes(self):
        ### avg production times for items
        apt1 = .15
        apt5 = .25
        string_apt = 'apt1=' + str(apt1) + ';apt5=' + str(apt5) + ';'
        return string_apt

    def getStringHoldingCosts(self):
        ### holding cost (uniform for all items)
        holding_cost = 2
        string_hc = 'hc=' + str(holding_cost) + ';'
        return string_hc

    def getStringItemPrices(self):
        ### prices for items
        pr1 = 1
        pr2 = 2
        pr3 = 3
        pr5 = 5
        string_prices = 'pr1=' + str(pr1) + ';pr2=' + str(pr2) + ';pr3=' + str(pr3) + ';pr5=' + str(pr5) + ';'
        return string_prices

    def getStringArrivalRates(self):
        ### arrival rates for products control the demand
        arrivalratep1 = 3.6
        arrivalratep2 = 3
        arrivalratep3 = 2.4
        arrivalratep4 = 1.8
        arrivalratep5 = 1.2
        string_arrival_rates = 'ar1=' + str(arrivalratep1) + ';ar2=' + str(arrivalratep2) + \
                               ';ar3=' + str(arrivalratep3) + ';ar4=' + str(arrivalratep4) \
                               + ';ar5=' + str(arrivalratep5) + ';'
        return string_arrival_rates

    def estimateVariance(self, num_random_points = 5, num_samples_per_point = 10):
        '''
        estimate variance of each information source by querying each at the
        same 5 random positions via 10 samples
        :param num_random_points: the number of random points queried for each IS
        :param num_samples_per_point: the number of samples for each point
        :return: list of vars
        '''

        measured_objvalues =  numpy.zeros(num_samples_per_point)
        variance_objvalues =  numpy.zeros( num_random_points )

        for i in xrange(num_random_points):
            x = self.drawRandomPoint()

            for it in xrange(num_samples_per_point):
                measured_objvalues[it] = self.evaluate(0,x)

            # estimate variance and cost for this point
            variance_objvalues[i] = numpy.var(measured_objvalues)

        print variance_objvalues

    def getList_IS_to_query(self):
        return self._list_IS_to_query

    def get_moe_domain(self):
        return TensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in self._search_domain])
