import sys
from math import isnan

from moe.optimal_learning.python.data_containers import SamplePoint
from moe.optimal_learning.python.python_version.covariance import SquareExponential

sys.path.append("../")


from multifidelity_KG.voi.optimization import *
import sql_util_cs

from load_and_store_data import obtainHistoricalDataForEGO
from mothers_little_helpers import process_parallel_results
from multifidelity_KG.obj_functions import *

from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartRosenbrock.rosenbrock_shifted import RosenbrockShifted
from coldStartRosenbrock.rosenbrock_slightshifted import RosenbrockSlightShifted
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla

from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var1 import AssembleToOrderVar1
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4

'''
Run the coldstart version of EGO on a benchmark problem.

based on Jialei's run_ego.py
'''


### The following parameters must be adapted for each simulator

### Rosenbrock
# obj_func_min = RosenbrockVanilla( )
# num_iterations = 25     # how many steps should the algo perform?
# num_init_pts_each_IS = 5

# AssembleToOrderVanilla
obj_func_min = AssembleToOrderVar2( mult=-1.0 )
num_iterations = 50     # how many steps should the algo perform?
num_init_pts_each_IS = 1 # 20

func_name = obj_func_min.getFuncName()
numIS = obj_func_min.getNumIS()
truth_IS = obj_func_min.getTruthIS()
list_IS_to_query = [0]

#obj_func_min = Rosenbrock(numIS, mult=1.0)
# numIS=1
# truth_IS = 0
# # ATO
# numIS=4
# truth_IS = 4
# func_name = 'assembleToOrder'
# init_data_pickle_filename = "ATO_4_IS"
# # benchmark_result_table_name = "rosenbrock_ego_newCost"
# best_so_far_table_name = "ATO_ego_best_so_far"
# cost_so_far_table_name = "ATO_ego_cost_so_far"
# obj_func_min = AssembleToOrder(numIS, mult=-1.0)

init_data_pickle_filename_prefix = obj_func_min.getFuncName() + '_' + 'IS_' \
                                        + '_'.join(str(element) for element in list_IS_to_query) + '_' \
                                        + str(num_init_pts_each_IS) + '_points_each_repl_'

# to determine the right pickle file
load_historical_data_from_pickle = True
pathToPickles = "../pickles/csCentered"

# less important params
num_replications = 50
num_threads = 64
num_multistart = 64
### end parameter

# perform 50 replications
for replication_no in range(num_replications):

    #init_data_pickle_filename = func_name + '_IS_0_10_points_each_repl_'
    init_data_pickle_filename = init_data_pickle_filename_prefix + str(replication_no)
    best_so_far_table_name = 'ego_' + func_name + '_best'
    cost_so_far_table_name = 'ego_' + func_name + '_cost'
    mysql_table_prefix_name = 'ego_hyper_' + func_name

    ### setup for EGO
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain])
    noise_and_cost_func = obj_func_min.noise_and_cost_func
    ego_hyper_param = pandas.read_sql_table(mysql_table_prefix_name, sql_util_cs.sql_engine).mean(axis=0).values
    cov_func = SquareExponential(ego_hyper_param)

    ### Obtain points for initial data
    data = obtainHistoricalDataForEGO(load_historical_data_from_pickle, obj_func_min, pathToPickles,
                                      list_IS_to_query, num_init_pts_each_IS, init_data_pickle_filename)

    best_sampled_val = numpy.inf

    # for i in range(obj_func_min._num_IS):
    #     if numpy.amin(init_vals[i]) < best_sampled_val:
    #         best_sampled_val = numpy.amin(init_vals[i])
    #         best_sampled_point = init_pts[i][numpy.argmin(init_vals[i]), :]

    #TODO for other problems that are not coldstart: if there is more than one IS for this problem, iterate over all to find the best value
    if numpy.amin(data._points_sampled_value) < best_sampled_val:
        best_sampled_val = numpy.amin(data._points_sampled_value)
        best_sampled_point = data._points_sampled[numpy.argmin(data._points_sampled_value)]
    # print best_sampled_val
    # print best_sampled_point
    # print data._points_sampled_value #data._points_sampled
    # exit(0)

    truth_at_best_sampled = obj_func_min.evaluate(truth_IS, best_sampled_point)

    ego_gp = GaussianProcess(cov_func, data)
    best_so_far = numpy.zeros(num_iterations)
    cost_so_far = numpy.zeros(num_iterations)

    # perform num_iterations -many steps of EI
    for current_iteration in range(num_iterations):
        print "EGO on {0}: itr {1}, {2}, {3}".format(func_name, replication_no, current_iteration, truth_at_best_sampled)
        expected_improvement_evaluator = ExpectedImprovement(ego_gp)
        min_negative_ei = numpy.inf

        def negative_ego_func(x):
            expected_improvement_evaluator.set_current_point(x.reshape((1, -1)))
            return -1.0 * expected_improvement_evaluator.compute_expected_improvement()

        def negative_ego_grad_func(x):
            expected_improvement_evaluator.set_current_point(x.reshape((1, -1)))
            return -1.0 * expected_improvement_evaluator.compute_grad_expected_improvement()[0, :]

        def min_negative_ego_func(start_point):
            return bfgs_optimization_grad(start_point, negative_ego_func, negative_ego_grad_func, obj_func_min._search_domain)

        with Parallel(n_jobs=num_threads) as parallel:
            # start_points_prepare = search_domain.generate_uniform_random_points_in_domain(num_candidate_start_points)
            # ei_vals = parallel(delayed(negative_ego_func)(x) for x in start_points_prepare)
            # sorted_idx_ei = numpy.argsort(ei_vals)
            # start_points = start_points_prepare[sorted_idx_ei[:num_multistart], :]
            start_points = search_domain.generate_uniform_random_points_in_domain(num_multistart)
            parallel_results = parallel(delayed(min_negative_ego_func)(pt) for pt in start_points)
        min_neg_ei, point_to_sample = process_parallel_results(parallel_results)
        predict_mean = ego_gp.compute_mean_of_points(point_to_sample.reshape((1,-1)))[0]
        predict_var = ego_gp.compute_variance_of_points(point_to_sample.reshape((1,-1)))[0,0]
        cost_per_iteration = 0.0

        # Speciality of EGO for MISO: evaluate the point that optimized EI for each IS
        val = obj_func_min.evaluate(truth_IS, point_to_sample)
        print 'val='+str(val)
        ego_gp.add_sampled_points([SamplePoint(point_to_sample, val, noise_and_cost_func(truth_IS, point_to_sample)[0])])
        cost_per_iteration += noise_and_cost_func(truth_IS, point_to_sample)[1]
        # for i in range(obj_func_min._num_IS):
        #     val = obj_func_min.evaluate(i + 1, point_to_sample)
        #     ego_gp.add_sampled_points([SamplePoint(point_to_sample, val, noise_and_cost_func(i + 1, point_to_sample)[0])])
        #     cost_per_iteration += noise_and_cost_func(i + 1, point_to_sample)[1]
        if val < best_sampled_val:
            best_sampled_val = val
            best_sampled_point = point_to_sample
            truth_at_best_sampled = val     # obj_func_min.evaluate(truth_IS, point_to_sample)
        best_so_far[current_iteration] = truth_at_best_sampled
        cost_so_far[current_iteration] = cost_per_iteration if current_iteration == 0 else (cost_so_far[current_iteration - 1] + cost_per_iteration)
        # print 'truth_at_best_sampled = ' + str(truth_at_best_sampled)
    best_so_far_table = pandas.DataFrame(best_so_far.reshape((1,-1)))
    best_so_far_table.to_sql(best_so_far_table_name, sql_util_cs.sql_engine, if_exists='append', index=False)
    # cost_so_far_table = pandas.DataFrame(cost_so_far.reshape((1,-1)))
    # cost_so_far_table.to_sql(cost_so_far_table_name, sql_util_cs.sql_engine, if_exists='append', index=False)


