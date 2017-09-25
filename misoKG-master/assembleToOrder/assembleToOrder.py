import pickle

import numpy

import subprocess
import time					# used to measure time
import random

__author__ = 'matthiaspoloczek'

'''
The class for the ATO Simulator.
'''


class AssembleToOrder(object):
    def __init__(self, numIS, mult=1.0):
        """
        :param numIS: number of information sources
        :param noise_and_cost_func: f(IS, x) returns (noise_var, cost)
        """
        self._dim = 8               # dim of the domain
        self._search_domain = numpy.repeat([[0., 20.]], self._dim, axis=0)

        self._numAvailIS = 4        # how many IS are implemented?
        self._truth_IS = 4
        self._list_IS_to_query = [1, 2, 3, 4]
        self._numIS = numIS         # no information sources
        if(numIS > self._numAvailIS):
            self._numIS = self._numAvailIS

        self._pathToMATLABScripts = "/fs/europa/g_pf/ATO_scripts/"

        # the setup estimated from the data that I used for experiments
        self._descriptionIS = numpy.array([[-1,-1.,-1.], # corresponds to no IS
                                    [10, 2.485, (8.887 - 7.5)], # for the first IS: runlength, variance,  computational cost in sec per call
                                    #[50, 3., (13. - 7.5)], # etc
                                    [100, 0.332, (11.422 - 7.5)],
                                    [500, 0.056, (24.633 - 7.5)],
                                    [1000, 0.027, (41.327 - 7.5)],
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
        self._num_IS = self._numIS

    def getNumInitPtsAllIS(self):
        '''
        :return: an array with self._dim entries that gives for each dimension the number of points to sample initially
        '''
        return numpy.repeat(int(round(10*self.getDim()/self.getNumIS())), self.getDim(), axis=0) # based on Jialei's recommendation

    def getRunlength(self, IS):
        return self._descriptionIS[IS][0]

    # def getVariance(self, IS, x):
    #     return self._descriptionIS[IS][1]
    #
    # def getCost(self, IS, x):
    #     return self._descriptionIS[IS][2]

    def noise_and_cost_func(self, IS, x):
        ''' The (estimated) noise (variance) and cost of evaluating information source IS at design x
        :param IS: index of information source, 1, ..., M
        :param x: the design
        :return: the variance and the query cost
        '''
        return (self._descriptionIS[IS][1], self._descriptionIS[IS][2])

    def drawRandomPoint(self):
        '''

        :return: a random point in the search space
        '''
        #return [self._prg.uniform(0.0,20.0) for i in xrange(self._dim)]
        return [self._prg.uniform( self._search_domain[i][0], self._search_domain[i][1] ) for i in xrange( self._search_domain.shape[0] )]

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

        try:
            start_time = time.time()
            # /usr/local/matlab/2015b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('ATO_run.m');exit;"
            # https://www.mathworks.com/matlabcentral/answers/97204-how-can-i-pass-input-parameters-when-running-matlab-in-batch-mode-in-windows

            runcmd = "b="+self.convertXtoString(x)+";length="+str(self.getRunlength(IS))\
                     +";seed="+str(random_seed)+";run(\'" + self._pathToMATLABScripts + "ATO_run.m\');exit;"
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

        # elapsed_time is a reasonable cost
        # fn and FnVar are the results of interest
        #print "runlength="+str(self.getRunlength(IS))+", x="+self.convertXtoString(x)+", fn="+str(fn)+" , FnVar="+str(FnVar)+" , elapsed_time="+str(elapsed_time)
        # for presentation, can be removed

        return self._mult * float(fn)   # return only the mean


    def estimateVarianceAndCost(self, num_random_points = 5, num_samples_per_point = 10):
        '''
        estimate variance and cost of each information source by querying each at the
        same 20 random positions
        :param num_random_points: the number of random points queried for each IS
        :param num_samples_per_point: the number of samples for each point
        :return: An array with [(IS, variance, cost)] that is automatically stored in the object
        '''

        highestIndex = max(self.getList_IS_to_query())
        # print 'highestISIndex = ' + str(highestIndex)
        measured_objvalues =  numpy.zeros(num_samples_per_point)
        measured_evalcosts =  numpy.zeros(num_samples_per_point)
        variance_objvalues =  numpy.zeros( (num_random_points, highestIndex+1) )
        mean_evalcosts =  numpy.zeros( (num_random_points, highestIndex+1) )

        for i in xrange(num_random_points):
            x = self.drawRandomPoint()
            for IS in self.getList_IS_to_query(): # IS+1 is the info source
                # query each specified IS at x

                for it in xrange(num_samples_per_point):
                    start_time = time.time()
                    measured_objvalues[it] = self.evaluate(IS,x)
                    elapsed_time = time.time() - start_time
                    measured_evalcosts[it] = elapsed_time

                # estimate variance and cost for this point
                variance_objvalues[i][IS] = numpy.var(measured_objvalues)
                mean_evalcosts[i][IS] = numpy.mean(measured_evalcosts)

        for IS in self.getList_IS_to_query(): # IS+1 is the info source
            self._descriptionIS[IS][self._descriptionIS_index_variance] = numpy.mean(variance_objvalues,axis=0)[IS]
            self._descriptionIS[IS][self._descriptionIS_index_cost] = numpy.mean(mean_evalcosts,axis=0)[IS]

        print self._descriptionIS

    def ensureBoundaries(self, x):
        '''
        Ensure that every point is within the domain and limit the number of places for each coordinate to 6.
        Numerical issues may lead to an x that is not within the search domain.

        Args:
            x: the point

        Returns: each value is moved to the closest boundary of that dimension and has at most 6 places
        '''
        for dim in range(len(x)):
            x[dim] = round(x[dim], 6)
            if x[dim] < self._search_domain[dim][0]:
                x[dim] = self._search_domain[dim][0]
            if x[dim] > self._search_domain[dim][1]:
                x[dim] = self._search_domain[dim][1]
        return x


    # some getters for attributes
    def getBestInitialObjValue(self):
        return self._mult * (-numpy.inf)

    def getTruthIS(self):
        return self._truth_IS

    def getNumIS(self):
        return self._num_IS

    def getDim(self):
        return self._dim

    def getSearchDomain(self):
        return self._search_domain

    def getList_IS_to_query(self):
        return self._list_IS_to_query


# TODO: adapt code below to use abs path of .m file
class AssembleToOrderPES(object):
    def __init__(self, mult=1.0):
        """
        :param numIS: number of information sources
        :param noise_and_cost_func: f(IS, x) returns (noise_var, cost)
        """
        self._dim = 8               # dim of the domain
        self._search_domain = numpy.repeat([[0., 20.]], self._dim, axis=0)

        self._numIS = 4        # how many IS are implemented?
        self._truth_IS = 0
        self._list_IS_to_query = range(4)

        # the setup estimated from the data that I used for experiments
        self._descriptionIS = numpy.array([
                                           # [10, 2.485, (8.887 - 7.5)], # for the first IS: runlength, variance,  computational cost in sec per call
                                           # #[50, 3., (13. - 7.5)], # etc
                                           # [100, 0.332, (11.422 - 7.5)],
                                           # [500, 0.056, (24.633 - 7.5)],
                                           # [1000, 0.027, (41.327 - 7.5)],
                                           # #[10000, 0.01, (400. - 7.5)]
                                           [1000, 0.027, 41.327],
                                           [10, 2.485, 8.887], # for the first IS: runlength, variance,  computational cost in sec per call
                                           #[50, 3., (13. - 7.5)], # etc
                                           [100, 0.332, 11.422],
                                           [500, 0.056, 24.633],
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
        self._num_IS = self._numIS

    def getNumInitPtsAllIS(self):
        '''
        :return: an array with self._dim entries that gives for each dimension the number of points to sample initially
        '''
        return numpy.repeat(int(round(10*self.getDim()/self.getNumIS())), self.getDim(), axis=0) # based on Jialei's recommendation

    def getRunlength(self, IS):
        return self._descriptionIS[IS][0]

    # def getVariance(self, IS, x):
    #     return self._descriptionIS[IS][1]
    #
    # def getCost(self, IS, x):
    #     return self._descriptionIS[IS][2]

    def noise_and_cost_func(self, IS, x):
        ''' The (estimated) noise (variance) and cost of evaluating information source IS at design x
        :param IS: index of information source, 1, ..., M
        :param x: the design
        :return: the variance and the query cost
        '''
        return (self._descriptionIS[IS][1], self._descriptionIS[IS][2])

    def drawRandomPoint(self):
        '''

        :return: a random point in the search space
        '''
        #return [self._prg.uniform(0.0,20.0) for i in xrange(self._dim)]
        return [self._prg.uniform( self._search_domain[i][0], self._search_domain[i][1] ) for i in xrange( self._search_domain.shape[0] )]

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

        try:
            start_time = time.time()
            # /usr/local/matlab/2015b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('ATO_run.m');exit;"
            # https://www.mathworks.com/matlabcentral/answers/97204-how-can-i-pass-input-parameters-when-running-matlab-in-batch-mode-in-windows

            runcmd = "b="+self.convertXtoString(x)+";length="+str(self.getRunlength(IS))+";seed="+str(random_seed)+";run(\'ATO_run.m\');exit;"
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

        # elapsed_time is a reasonable cost
        # fn and FnVar are the results of interest
        #print "runlength="+str(self.getRunlength(IS))+", x="+self.convertXtoString(x)+", fn="+str(fn)+" , FnVar="+str(FnVar)+" , elapsed_time="+str(elapsed_time)
        # for presentation, can be removed

        return self._mult * float(fn)   # return only the mean


    def estimateVarianceAndCost(self, num_random_points = 5, num_samples_per_point = 10):
        '''
        estimate variance and cost of each information source by querying each at the
        same 20 random positions
        :param num_random_points: the number of random points queried for each IS
        :param num_samples_per_point: the number of samples for each point
        :return: An array with [(IS, variance, cost)] that is automatically stored in the object
        '''

        highestIndex = max(self.getList_IS_to_query())
        # print 'highestISIndex = ' + str(highestIndex)
        measured_objvalues =  numpy.zeros(num_samples_per_point)
        measured_evalcosts =  numpy.zeros(num_samples_per_point)
        variance_objvalues =  numpy.zeros( (num_random_points, highestIndex+1) )
        mean_evalcosts =  numpy.zeros( (num_random_points, highestIndex+1) )

        for i in xrange(num_random_points):
            x = self.drawRandomPoint()
            for IS in self.getList_IS_to_query(): # IS+1 is the info source
                # query each specified IS at x

                for it in xrange(num_samples_per_point):
                    start_time = time.time()
                    measured_objvalues[it] = self.evaluate(IS,x)
                    elapsed_time = time.time() - start_time
                    measured_evalcosts[it] = elapsed_time

                # estimate variance and cost for this point
                variance_objvalues[i][IS] = numpy.var(measured_objvalues)
                mean_evalcosts[i][IS] = numpy.mean(measured_evalcosts)

        for IS in self.getList_IS_to_query(): # IS+1 is the info source
            self._descriptionIS[IS][self._descriptionIS_index_variance] = numpy.mean(variance_objvalues,axis=0)[IS]
            self._descriptionIS[IS][self._descriptionIS_index_cost] = numpy.mean(mean_evalcosts,axis=0)[IS]

        print self._descriptionIS

    def ensureBoundaries(self, x):
        '''
        Ensure that every point is within the domain and limit the number of places for each coordinate to 6.
        Numerical issues may lead to an x that is not within the search domain.

        Args:
            x: the point

        Returns: each value is moved to the closest boundary of that dimension and has at most 6 places
        '''
        for dim in range(len(x)):
            x[dim] = round(x[dim], 6)
            if x[dim] < self._search_domain[dim][0]:
                x[dim] = self._search_domain[dim][0]
            if x[dim] > self._search_domain[dim][1]:
                x[dim] = self._search_domain[dim][1]
        return x


    # some getters for attributes
    def getBestInitialObjValue(self):
        return self._mult * (-numpy.inf)

    def getTruthIS(self):
        return self._truth_IS

    def getNumIS(self):
        return self._num_IS

    def getDim(self):
        return self._dim

    def getSearchDomain(self):
        return self._search_domain

    def getList_IS_to_query(self):
        return self._list_IS_to_query

    def getFuncName(self):
        return "atopes"
