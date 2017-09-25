import cPickle as pickle
import sys

import numpy
from joblib import Parallel, delayed

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

from assembleToOrder.assembleToOrder import AssembleToOrderPES
from multifidelity_KG.obj_functions import RosenbrockNoiseFreePES, RosenbrockNewNoiseFreePES
from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4



def miso_gen_data():
    """
    This script intend to do the same thing as sample_initial_points.py, with the only difference that it calls AssembleToOrderPES
    as objective, which place truth_is at IS0. This is required by Entropy Search algo and also makes sense when truth IS is accessible.
    """
    ### Need to set the following parameters!
    #obj_func_min = RosenbrockShifted( )
    obj_func_min = AssembleToOrderPES(mult=-1.0)
    # obj_func_min = RosenbrockNoiseFreePES(mult=1.0)
    # obj_func_min = RosenbrockNewNoiseFreePES(mult=1.0)

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
    directory = "/fs/europa/g_pf/pickles/miso"

    for replication_no in range(num_replications):
        filename = obj_func_min.getFuncName() + '_' + 'IS_' + '_'.join(str(element) for element in list_IS_to_query) \
                   + '_' + str(num_init_pts_each_IS) + "_points_each"
        if num_replications > 1:
            filename += '_repl_' + str(replication_no)
        print 'filename=' + filename

        search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain]) # this file is used below again and hence should be made available there, too
        init_points_for_all_IS = []
        init_vals_all_IS = []
        is_list = []

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
                is_list.append(numpy.ones(num_init_pts_each_IS)*index_IS)
                index_Array +=1
        print "min value: {0}".format(numpy.amin(init_vals_all_IS))
        data = {"points": init_points_for_all_IS, "vals": init_vals_all_IS, "IS": is_list}
        with open("{0}/{1}.pickle".format(directory, filename), "wb") as file:
            pickle.dump(data, file)

def coldstart_gen_data(obj_func_min, num_init_pts, num_replications, directory):
    """ generate initial data for experiments and store in pickle
    """
    for replication_no in range(num_replications):
        filename = "{0}/{1}_{2}_points_each_repl_{3}.pickle".format(directory, obj_func_min.getFuncName(), num_init_pts, replication_no)
        search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain]) # this file is used below again and hence should be made available there, too
        points = search_domain.generate_uniform_random_points_in_domain(num_init_pts)
        vals = [obj_func_min.evaluate(0, pt) for pt in points]
        data = {"points": points, "vals": vals, "noise": obj_func_min.noise_and_cost_func(0, None)[0] * numpy.ones(num_init_pts)}
        with open(filename, "wb") as file:
            pickle.dump(data, file)

def coldstart_gen_hyperdata(primary_obj_func_min, list_other_obj_func_min, num_pts, directory):
    """ generate data for hyperparameter optimization and store in pickle
    """
    filename = "{0}/hyper_{1}_points_{2}_{3}.pickle".format(directory, num_pts, primary_obj_func_min.getFuncName(), "_".join([func.getFuncName() for func in list_other_obj_func_min]))
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in primary_obj_func_min._search_domain]) # this file is used below again and hence should be made available there, too
    points = search_domain.generate_uniform_random_points_in_domain(num_pts)
    vals = [[primary_obj_func_min.evaluate(0, pt) for pt in points]]
    noise = [primary_obj_func_min.noise_and_cost_func(0, None)]
    for obj_func in list_other_obj_func_min:
        vals.append([obj_func.evaluate(0, pt) for pt in points])
        noise.append(obj_func.noise_and_cost_func(0, None))
    data = {"points": points, "vals": vals, "noise": noise}
    with open(filename, "wb") as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    ######
    # coldstart init data
    # obj_func_min = RosenbrockVanilla(mult=1.0)
    # num_init_pts = 1
    # num_replications = 100
    # directory = "/fs/europa/g_pf/pickles/coldstart/data"
    # coldstart_gen_data(obj_func_min, num_init_pts, num_replications, directory)
    # coldstart init data end

    ######
    # coldstart hyper data
    primary_obj_func_min = RosenbrockVanilla(mult=1.0)
    list_other_obj_func_min = [RosenbrockBiased(mult=1.0)]
    num_pts = 1000
    directory = "/fs/europa/g_pf/pickles/coldstart/data"
    coldstart_gen_hyperdata(primary_obj_func_min, list_other_obj_func_min, num_pts, directory)
    # coldstart hyper data end
