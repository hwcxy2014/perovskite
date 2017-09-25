import pickle

import numpy

import time					# used to measure time
import random

from call_cfd import call_XFOIL, call_SU2
import sys
sys.path.append("/fs/europa/g_pf/Xfoil/bin/")   # This path is hardcoded for icse.cornell.edu
sys.path.append("/fs/europa/g_pf/SU2/bin")      # This path is hardcoded for icse.cornell.edu
#TODO What happens when the weird filesytem of the ICSE cluster hides /g_pf/* again?

__author__ = 'matthiaspoloczek'

'''
The class for the Drag and Lift Optimization problem, as studied by Lam et al.

Separate IS for obj func and constraints for more generality:
IS 1 drag output of XFOIL
IS 2 lift output of XFOIL
IS 3 drag output of SU2
IS 4 lift output of SU2
'''


class DragAndLift(object):
    def __init__(self, mult= 1.0):
        '''

        :param mult: The problem is a minimization problem. To turn it into a maximization problem, set mult = -1.0
        :return:
        '''
        self._dim = 2               # dim of the domain
        self._search_domain = numpy.array([[0.1, 0.5], [0.01, 8.0]])
        # Following Lam et al.: M in [0.1, 0.5] and and angle-of-attack aoa in [0.01, 8.0]

        self._numAvailIS = 4        # how many IS are implemented?
        self._num_IS = 4            # how many shall be available?
        '''
        Separate IS for obj func and constraints for more generality:
        IS 1 drag output of XFOIL
        IS 2 lift output of XFOIL
        IS 3 drag output of SU2
        IS 4 lift output of SU2
        '''

        # parameters taken from Lam et al.
        self._descriptionIS = numpy.array([[-1.,-1.], # corresponds to no IS
                                    [10., 1.], # for the first IS: variance, computational cost in sec per call
                                    [10., 1.],
                                    [1., 3600.],
                                    [1., 3600.],
                                    ])
        self._descriptionIS_index_variance = 0  # the value at index 0 is the variance
        self._descriptionIS_index_cost = 1      # the value at index 1 is the cost

        self._mult = mult           # control whether the optimization problem is maximizing or minimizing. The default (1.0) is to maximize, set to -1.0 to minimize
        self._prg = random.Random() # make sequential calls to the same PRG

        # A dictionary to store previously obtained, but not yet used drag and lift values for
        # The intention is to reduce the number of IS queries, when first the drag and then the lift (or vice versa) of some design x is requested
        self._previousQueriesXFOIL = {}
        self._previousQueriesSU2 = {}

    def getNumInitPtsAllIS(self):
        '''
        :return: an array with self._dim entries that gives for each dimension the number of points to sample initially
        '''
        return numpy.repeat(int(round(10*self._dim/self._numIS)), self._dim, axis=0)
        # based on Jialei's recommendation (based on Jones et al.)

    def noise_and_cost_func(self, IS, x):
        ''' The (estimated) noise (variance) and cost of evaluating information source IS at design x
        :param IS: index of information source, 1, ..., M
        :param x: the design
        :return: the variance and the query cost
        '''
        return (self._descriptionIS[IS][0], self._descriptionIS[IS][1])

    def drawRandomPoint(self):
        '''

        :return: a random point in the search space
        '''
        return [self._prg.uniform( self._search_domain[i][0], self._search_domain[i][1] ) for i in xrange( self._search_domain.shape[0] )]

    def evaluate(self, IS, x):
        """
        # Run the XFOIL or SU2 simulator
        :param IS: index of information source, 1, ..., M
        :param x: a point in the search_domain
        :return: the value at x estimated by info source IS
        """

        mach = x[0]
        aoa = x[1]
        fn = -1.0   # the return value
        requestedValueFound = False

        database = self._previousQueriesXFOIL
        # assume a request for XFOIL
        if(2 < IS <= 4):
            # a request for SU2
            database = self._previousQueriesSU2

        # [0] for IS 1 and IS 3, [1] for IS 2 and IS 4, thus:
        z =  (IS+1) % 2

        if( (mach,aoa) in database ):
            # this setting was request before, check if there is still an unused value
            if(len(database.get( (mach,aoa) )[z]) > 0):
                # return first value from list and remove it from list
                fn = database.get( (mach,aoa) )[z].pop(0)
                requestedValueFound = True

        if(not requestedValueFound):
            # run simulator to obtain the requested value
            if(1 <= IS <= 2):
                # run XFOIL
                drag, lift = call_XFOIL(mach,aoa)
            elif(3 <= IS <= 4):
                # run SU2
                drag, lift = call_SU2(mach,aoa)
            print "drag = " + str(drag) + ", lift = " + str(lift)

            # store newly created value
            if(z == 0):
                fn = drag
                if( (mach,aoa) in database ):
                    # append new lift to list
                    database.get( (mach,aoa) )[1].append(lift)
                else:
                    # create a new key and store new, used value
                    database[ (mach,aoa) ] = [ [] , [lift] ]
            if(z == 1):
                fn = lift
                if( (mach,aoa) in database ):
                    # append new drag to list
                    database.get( (mach,aoa) )[0].append(drag)
                else:
                    # create a new key and store new, used value
                    database[ (mach,aoa) ] = [ [drag] , [] ]

        return self._mult * fn   # return only the mean

    #TODO the IS for the constraint will have cost 0, but that does not matter!
    def estimateVarianceAndCost(self, num_random_points = 5, num_samples_per_point = 10):
        '''
        estimate variance and cost of each information source by querying each at the
        same num_random_points-many random points
        :param num_random_points: the number of random points queried for each IS
        :param num_samples_per_point: the number of samples for each point
        :return: An array with [(IS, variance, cost)] that is automatically stored in the object
        '''

        measured_objvalues =  numpy.zeros(num_samples_per_point)
        measured_evalcosts =  numpy.zeros(num_samples_per_point)
        variance_objvalues =  numpy.zeros( (num_random_points, self._numAvailIS) )
        mean_evalcosts =  numpy.zeros( (num_random_points, self._numAvailIS) )

        for i in xrange(num_random_points):
            x = self.drawRandomPoint()
            for IS in xrange(self._numAvailIS): # IS+1 is the info source
                # query each specified IS at x

                for it in xrange(num_samples_per_point):
                    #print "Invoking IS" + str(IS+1)
                    start_time = time.time()
                    measured_objvalues[it] = self.evaluate(IS+1,x)
                    elapsed_time = time.time() - start_time
                    measured_evalcosts[it] = elapsed_time
                    print "elapsed_time=" + str(elapsed_time)

                # estimate variance and cost for this point
                variance_objvalues[i][IS] = numpy.var(measured_objvalues)
                mean_evalcosts[i][IS] = numpy.mean(measured_evalcosts)

        # the calculated description is stored in the object
        for IS in xrange(self._numAvailIS): # IS+1 is the info source
            self._descriptionIS[(IS+1)][self._descriptionIS_index_variance] = numpy.mean(variance_objvalues,axis=0)[IS]
            self._descriptionIS[(IS+1)][self._descriptionIS_index_cost] = numpy.mean(mean_evalcosts,axis=0)[IS]

        # output estimated noise and cost
        print self._descriptionIS

    # some getters for attributes
    def getBestInitialObjValue(self):
        return self._mult * (-numpy.inf)

    def getNumIS(self):
        return self._num_IS

    def getDim(self):
        return self._dim

    def getSearchDomain(self):
        return self._search_domain


