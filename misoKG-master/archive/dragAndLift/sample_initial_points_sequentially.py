import numpy
from joblib import Parallel, delayed
import cPickle as pickle
import time					# used to measure time

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

import sys
sys.path.append("../")
from mothers_little_helpers import pickle_init_points_for_all_IS, pickle_vals

from assembleToOrder.assembleToOrder import AssembleToOrder
from multifidelity_KG.obj_functions import Rosenbrock
from dragAndLift import DragAndLift


'''
Sample initial points for all IS and evaluate them

Pickle init_points_for_all_IS, vals
'''

# call_CFD requires that this file is in a subdir of multifidelity_code

def load_data_from_a_min_problem(directory, filename):
    """
    :param directory: dir of pickle file
    :param filename: name of pickle file
    :return: init_points_for_all_IS, init_vals_for_all_IS
    """
    with open("{0}/{1}.pickle".format(directory, filename), "rb") as input_file:
        data = pickle.load(input_file)
        return data['points'], data['vals']

def sample_and_pickle_initial_points(obj_func_min, allows_parallelization, init_points_for_all_IS, directory, filename):
    #global bound, search_domain, parallel, i, pt, data

    init_vals_all_IS = []

    def parallel_func(IS, pt):
        return obj_func_min.evaluate(IS, pt)

    if (allows_parallelization):
        with Parallel(n_jobs=num_init_pts_each_IS) as parallel:
            for i in range(obj_func_min.getNumIS()):
                print "{0}th IS".format(i)
                # points = search_domain.generate_uniform_random_points_in_domain(num_init_pts_each_IS)
                # init_points_for_all_IS.append(points)
                vals = parallel(delayed(parallel_func)(i + 1, pt) for pt in init_points_for_all_IS[i])
                init_vals_all_IS.append(vals)
        print "min value: {0}".format(numpy.amin(init_vals_all_IS))
        data = {"points": init_points_for_all_IS, "vals": init_vals_all_IS}
        with open("{0}/{1}.pickle".format(directory, filename), "wb+") as file:
            pickle.dump(data, file)
        #TODO there is some issue with pickling the files for some simulators

    else:
        for i in range(obj_func_min.getNumIS()):
            start_time = time.time()
            vals = [obj_func_min.evaluate(i+1, pt) for pt in init_points_for_all_IS[i]]
            elapsed_time = time.time() - start_time
            print "Simulation_cost=" + str(elapsed_time)

        func_name = filename
        pickle_init_points_for_all_IS(directory, func_name, obj_func_min.getNumIS(), init_points_for_all_IS)
        pickle_vals(directory, func_name, obj_func_min.getNumIS(), vals)

if __name__ == "__main__":
    ### Need to set the following parameters!
    obj_func_min = DragAndLift( mult=1.0 )
    directory = "pickles"
    #num_init_pts_each_IS = 10
    allows_parallelization = False  # set to True if each simulator/IS can be queried multiple times simultaneously
    # is True for rosenbrock and ATO
    # is False for dragAndLift
    ### end

    # specific to each scenario
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in
                                               obj_func_min.getSearchDomain()])
    # this file is used below again and hence should be made available there, too

    lastExistingSetId = 1   # prevent existing datasets from being overwritten
    for num_init_pts_each_IS in [10,10]: # [5, 10, 10, 5, 5]:

        init_points_for_all_IS = []
        # IS 1 and 2 at the same points
        points = search_domain.generate_uniform_random_points_in_domain(num_init_pts_each_IS)
        init_points_for_all_IS.append(points)
        init_points_for_all_IS.append(points)

        # IS 3 and 4 at the same points
        points = search_domain.generate_uniform_random_points_in_domain(num_init_pts_each_IS)
        init_points_for_all_IS.append(points)
        init_points_for_all_IS.append(points)

        lastExistingSetId += 1
        filename = "dragAndLift_4_IS_" + str(num_init_pts_each_IS) + "_points_each_id_set" + str(lastExistingSetId)
        print ("filename=" + filename)
        sample_and_pickle_initial_points(obj_func_min, allows_parallelization, init_points_for_all_IS, directory, filename)

### Original version:
# def sample_initial_points(func_name, obj_func_min, num_init_pts_all_IS):
#     search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain]) # this file is used below again and hence should be made available there, too
#     # Gen initial points for all IS
#     init_points_for_all_IS = []
#     best_initial_value = numpy.inf  # MINIMIZATION FUNCTION
#
#     for i in range(obj_func_min.getNumIS()):
#         points = search_domain.generate_uniform_random_points_in_domain(num_init_pts_all_IS[i])
#         init_points_for_all_IS.append(points)
#         vals = [obj_func_min.evaluate(i+1, pt) for pt in init_points_for_all_IS[i]]
#         sample_vars = [obj_func_min.noise_and_cost_func(i+1, pt)[0] for pt in init_points_for_all_IS[i]]
#         best_initial_value = min(best_initial_value, numpy.amin(vals))  # MINIMIZATION FUNCTION
#
#     # pickle the sample points and values
#     pickle_init_points_for_all_IS(func_name, obj_func_min.getNumIS(), init_points_for_all_IS)
#     pickle_vals(func_name, obj_func_min.getNumIS(), vals)
#     pickle_sample_vars(func_name, obj_func_min.getNumIS(), sample_vars)
#     pickle_best_initial_value(func_name, obj_func_min.getNumIS(), best_initial_value)


