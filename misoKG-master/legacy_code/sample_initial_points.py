import cPickle as pickle
import sys

import numpy
from joblib import Parallel, delayed

sys.path.append("../")

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

from assembleToOrderExtended.assembleToOrderExtended import AssembleToOrderExtended

__author__ = 'matthiaspoloczek'


'''
Evaluate points add a Latin Hypercube design and store them in a pickle file

Note that the pickle file has one array giving the list of points for each IS,
and one array giving the list of objective values for each IS.
The objective value must be for the MINIMIZATION VERSION of the problem, so choose mult accordingly when
instantiating the problem object.

This version creates several files, with the intention that there will be one for each replication.
Thus, the output of the algorithms will be reproducable.
'''



if __name__ == "__main__":
    ### Need to set the following parameters!
    #obj_func_min = RosenbrockShifted( )
    obj_func_min = AssembleToOrderExtended( mult=-1.0 )

    # list of IS that are to be queried
    list_IS_to_query = obj_func_min.getList_IS_to_query() #[1,2,3] # [0]# for coldstart # range(obj_func_min._num_IS)

    #string_list_IS_to_query = 'IS_' + '_'.join(str(element) for element in list_IS_to_query)
    # print string_list_IS_to_query
    # exit(0)

    # create initial data for runs
    num_init_pts_each_IS = 20 ###5 # for Rosenbrock # 20 # for ATO
    num_replications = 100
    # # create data for hyper opt.
    # num_init_pts_each_IS = 200
    # num_replications = 3

    allows_parallelization = True  # set to True if each simulator/IS can be queried multiple times simultaneously
    # is True for rosenbrock and ATO
    # is False for dragAndLift
    ### end
    directory = "../pickles/miso"

    for replication_no in range(num_replications):
        filename = obj_func_min.getFuncName() + '_' + 'IS_' + '_'.join(str(element) for element in list_IS_to_query) \
               + '_' + str(num_init_pts_each_IS) + "_points_each"
        if num_replications > 1:
            filename += '_repl_' + str(replication_no)
        print 'filename=' + filename

        search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain]) # this file is used below again and hence should be made available there, too
        init_points_for_all_IS = []
        init_vals_all_IS = []

        def parallel_func(IS, pt):
            return obj_func_min.evaluate(IS, pt)

        num_parallel_jobs = num_init_pts_each_IS    # Jialei's original choice
        if(('ato' in obj_func_min.getFuncName()) and (num_parallel_jobs > 10)): # do not start too many MATLAB instances
            num_parallel_jobs = 10
        if(not allows_parallelization):
            num_parallel_jobs = 1

        index_Array = 0 # which entry of the array to write into?
        with Parallel(n_jobs=num_parallel_jobs) as parallel:
            for index_IS in list_IS_to_query:
                print "{0}th IS".format(index_IS)
                points = search_domain.generate_uniform_random_points_in_domain(num_init_pts_each_IS)
                init_points_for_all_IS.append(points)
                vals = parallel(delayed(parallel_func)(index_IS, pt) for pt in init_points_for_all_IS[index_Array])
                init_vals_all_IS.append(vals)
                index_Array +=1
        print "min value: {0}".format(numpy.amin(init_vals_all_IS))
        data = {"points": init_points_for_all_IS, "vals": init_vals_all_IS}
        with open("{0}/{1}.pickle".format(directory, filename), "wb+") as file:
            pickle.dump(data, file)
        #
        # with open("{0}/{1}.pickle".format(directory, filename), "rb") as input_file:
        #     dataIN = pickle.load(input_file)
        #
        # print dataIN['vals']


