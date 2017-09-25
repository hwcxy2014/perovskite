import os
import sys
import cPickle as pickle    # pickle is actually used

from moe.optimal_learning.python.cpp_wrappers.covariance import MixedSquareExponential as cppMixedSquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcessNew
from moe.optimal_learning.python.data_containers import SamplePoint


from multifidelity_KG.voi.optimization import *
import sql_util
from mothers_little_helpers import *
from load_and_store_data import load_data_from_a_min_problem, createHistoricalDataForKG, findBestSampledValue, \
    create_listPrevData, match_pickle_filename

# from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
# from coldStartRosenbrock.rosenbrock_shifted import RosenbrockShifted
# from coldStartRosenbrock.rosenbrock_slightshifted import RosenbrockSlightShifted
# from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
# from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
#
# from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
# from coldStartAssembleToOrder.assembleToOrder_var1 import AssembleToOrderVar1
# from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
# from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
# from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4

from assembleToOrderExtended.assembleToOrderExtended import AssembleToOrderExtended



'''
Run misoKG on a benchmark problem.

based on the coldstart version
'''

__author__ = 'matthiaspoloczek'

### MP: I have commented and refactored Jialei's code, so we should use this version to build on for future versions


rep_begin = 40
rep_end = 50
# argv = sys.argv[1:]
# replication_no = int(argv[0])
# additional_naming = "_diff"
additional_naming = "_diff"
additional_2 = "150"
# additional_2 = ""
for replication_no in range(rep_begin, rep_end):
    ### The following parameters must be adapted for each simulator

    ### Rosenbrock family
    # obj_func_min = RosenbrockVanilla( mult=1.0 )
    # obj_func_max = RosenbrockVanilla( mult=-1.0 )                     # used by KG
    # num_iterations = 25     # iterations per replication
    # num_init_pts = 5        # points training data from the current IS

    ### AssembleToOrderExtended
    obj_func_min = AssembleToOrderExtended( mult=-1.0 )
    obj_func_max = AssembleToOrderExtended( )                     # used by KG
    num_iterations = 150     # iterations per replication
    num_init_pts = 20
    list_IS_to_query = obj_func_min.getList_IS_to_query()

    func_name = obj_func_min.getFuncName()
    truth_IS = obj_func_min.getTruthIS()
    exploitation_IS = 1

    # list of strings that give the filenames of the datasets to be loaded as additional IS
    #TODO choose correct previous data for instance
    list_previous_datasets_to_load = [] #['data_vKG_rb_sinN'] #['data_vKG_rb_van'] # []
    print "previous datasets: {0}".format(list_previous_datasets_to_load)

    path_to_pickle_data = "/fs/europa/g_pf/pickles/miso"
    path_to_pickle_result = "/fs/europa/g_pf/pickles/miso/result"
    # init_data_pickle_filename_prefix = func_name + '_IS_0_' + str(num_init_pts) + '_points_each_repl_'
    init_data_pickle_filename_prefix = obj_func_min.getFuncName() + '_' + 'IS_' \
                                            + '_'.join(str(element) for element in list_IS_to_query) + '_' \
                                            + str(num_init_pts) + '_points_each_repl_'


    # less important params
    exploitation_threshold = 1e-5
    num_x_prime = 3000
    num_discretization_before_ranking = num_x_prime * 3
    num_threads = 32
    num_multistart = 32
    num_candidate_start_points = 500
    ### end parameter

    # perform num_replications replications, each replication performs num_iterations many steps

    # create list of previous datasets
    all_filenames = os.listdir(path_to_pickle_data)
    list_previous_dataset_names_for_this_replication = []
    for name_prefix in list_previous_datasets_to_load:
        list_previous_dataset_names_for_this_replication.append(match_pickle_filename(all_filenames, name_prefix+"_repl", replication_no))
    listPrevData = create_listPrevData(obj_func_min, list_previous_dataset_names_for_this_replication,
                                       replication_no, path_to_pickle_data, init_data_pickle_filename_prefix)

    # Select the best IS or use only IS 0
    # run_acquisition_on_IS_0_only = True
    number_of_previous_datasets = len(listPrevData) - 1
    #print 'number_of_previous_datasets = ' + str(number_of_previous_datasets)

    func_names_prev_datasets = []
    for prev_dataset_name in list_previous_dataset_names_for_this_replication:
        func_names_prev_datasets.append(prev_dataset_name.split("_")[-3])
    names_used_datasets = "" if len(func_names_prev_datasets) == 0 else "_w_{0}".format("_".join(func_names_prev_datasets))
    filename_to_pickle = "data_misoKG_{0}{1}{2}{3}_repl_{4}".format(func_name, names_used_datasets, additional_naming, additional_2, replication_no)

    # name the tables
    mysql_hypers_table_name = "{0}_hyper_{1}{2}{3}".format("vkg" if len(func_names_prev_datasets) == 0 else "cskg", func_name, names_used_datasets, additional_naming)
    best_so_far_table_name = "{0}_{1}{2}{3}{4}_best".format("vkg" if len(func_names_prev_datasets) == 0 else "cskg", func_name, names_used_datasets, additional_naming, additional_2)
    cost_so_far_table_name = "{0}_{1}{2}{3}{4}_cost".format("vkg" if len(func_names_prev_datasets) == 0 else "cskg", func_name, names_used_datasets, additional_naming, additional_2)
    print mysql_hypers_table_name
    print best_so_far_table_name
    print filename_to_pickle

    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_max._search_domain])
    noise_and_cost_func = obj_func_min.noise_and_cost_func

    # setup benchmark result container
    #kg_hyper_param = [100.0, 1, 1, 1, 1, 1, 1, 1, 1] # test hypers
    kg_hyper_param = pandas.read_sql_table(mysql_hypers_table_name, sql_util.sql_engine).mean(axis=0).values
    print 'kg_hyper_param' +  str(kg_hyper_param)

    kg_data = createHistoricalDataForKG(obj_func_min.getDim(), listPrevData)
    ### only consider the initial data of this IS for the best value
    best_sampled_point, best_sampled_val, truth_at_best_sampled = findBestSampledValue(obj_func_min, [listPrevData[0]], truth_IS)
    print 'truth_at_best_sampled = ' + str(truth_at_best_sampled)
    ### Looking at all the previous data sets and sampling the point, that was best for its resp. IS, at the current IS
    #best_sampled_point, best_sampled_val, truth_at_best_sampled = findBestSampledValue(obj_func_min, listPrevData, truth_IS)

    kg_cov_cpp = cppMixedSquareExponential(hyperparameters=kg_hyper_param)
    #TODO We should come up with a solution to set num_IS_in so that it is valid in any setting
    num_IS_in = max(max(obj_func_min.getList_IS_to_query()), number_of_previous_datasets)
    # Jialei said that num_IS_in must be 2 if the IS are 0,1,2
    # print 'num_IS_in = ' + str(num_IS_in)
    kg_gp_cpp = GaussianProcessNew(kg_cov_cpp, kg_data, num_IS_in)
    #here: third arg obj_func_max._num_IS was replaced by number_of_previous_datasets

    # collect all samples from a single replication: (IS, point, sampled_val, noise_var)
    list_best = []
    list_cost = []
    list_sampled_IS = []
    list_sampled_points = []
    list_sampled_vals = []
    list_noise_variance_at_sample = []
    list_mu_star_truth = []
    list_raw_voi = []
    init_best_truth = obj_func_min.evaluate(obj_func_min.getTruthIS(), kg_data.points_sampled[numpy.argmax(kg_data._points_sampled_value), 1:])

    # MP: How many steps does the acquisition function perform? This value is num_iterations
    total_cost = 0.
    best_mu_star_truth = numpy.inf
    for kg_iteration in range(num_iterations):
        raw_voi = []
        print "benchmark {0}, repl {1}, step {2}".format(mysql_hypers_table_name, replication_no, kg_iteration)
        ### First discretize points and then only keep the good points idea
        discretization_points = search_domain.generate_uniform_random_points_in_domain(num_discretization_before_ranking)
        discretization_points = numpy.hstack((numpy.zeros((num_discretization_before_ranking,1)), discretization_points))
        all_mu = kg_gp_cpp.compute_mean_of_points(discretization_points)
        sorted_idx = numpy.argsort(all_mu)
        all_zero_x_prime = discretization_points[sorted_idx[-num_x_prime:], :]
        ### idea ends
        # all_zero_x_prime = numpy.hstack((numpy.zeros((num_x_prime,1)), search_domain.generate_uniform_random_points_in_domain(num_x_prime)))

        def min_kg_unit(start_pt, IS):
            func_to_min, grad_func = negative_kg_and_grad_given_x_prime(IS, all_zero_x_prime, noise_and_cost_func, kg_gp_cpp)
            return bfgs_optimization_grad(start_pt, func_to_min, grad_func, obj_func_max._search_domain)

        def compute_kg_unit(x, IS):
            return compute_kg_given_x_prime(IS, x, all_zero_x_prime, noise_and_cost_func(IS, x)[0], noise_and_cost_func(IS, x)[1], kg_gp_cpp)

        def find_mu_star(start_pt):
            return bfgs_optimization(start_pt, negative_mu_kg(kg_gp_cpp), obj_func_max._search_domain)

        min_negative_kg = numpy.inf
        with Parallel(n_jobs=num_threads) as parallel:
            for id_IS in obj_func_min.getList_IS_to_query():
                # the id of the id_IS-th IS is id_IS
                start_points_prepare = search_domain.generate_uniform_random_points_in_domain(num_candidate_start_points)
                kg_vals = parallel(delayed(compute_kg_unit)(x, id_IS) for x in start_points_prepare)
                sorted_idx_kg = numpy.argsort(kg_vals)
                start_points = start_points_prepare[sorted_idx_kg[-num_multistart:], :]
                parallel_results = parallel(delayed(min_kg_unit)(pt, id_IS) for pt in start_points)
                inner_min, inner_min_point = process_parallel_results(parallel_results)
                raw_voi.append(-inner_min * obj_func_min.noise_and_cost_func(id_IS, None)[1])
                if inner_min < min_negative_kg:
                    min_negative_kg = inner_min
                    point_to_sample = inner_min_point
                    sample_IS = id_IS
                print "IS {0}, KG {1}".format(id_IS, -inner_min)
            start_points_prepare = search_domain.generate_uniform_random_points_in_domain(num_candidate_start_points)
            mu_vals = kg_gp_cpp.compute_mean_of_points(numpy.hstack((numpy.zeros((num_candidate_start_points, 1)), start_points_prepare)))
            start_points = start_points_prepare[numpy.argsort(mu_vals)[-num_multistart:], :]
            parallel_results = parallel(delayed(find_mu_star)(pt) for pt in start_points)
            negative_mu_star, mu_star_point = process_parallel_results(parallel_results)
        if -min_negative_kg < exploitation_threshold:
            sample_IS = exploitation_IS
            print "KG search failed, do exploitation"
            point_to_sample = mu_star_point

        predict_mean = kg_gp_cpp.compute_mean_of_points(numpy.concatenate(([0], point_to_sample)).reshape((1,-1)))[0]
        predict_var = kg_gp_cpp.compute_variance_of_points(numpy.concatenate(([0], point_to_sample)).reshape((1,-1)))[0,0]
        cost = noise_and_cost_func(sample_IS, point_to_sample)[1]
        mu_star_var = kg_gp_cpp.compute_variance_of_points(numpy.concatenate(([0], mu_star_point)).reshape((1,-1)))[0,0]

        ### OPTIMIZATION for csKG, where truth_IS = exploitation_IS = sample_IS
        sample_val = obj_func_min.evaluate(sample_IS, point_to_sample)
        if ((point_to_sample == mu_star_point).all and (truth_IS == sample_IS)):
            mu_star_truth = sample_val
        else:
            mu_star_truth = obj_func_min.evaluate(truth_IS, mu_star_point)
        if mu_star_truth < best_mu_star_truth:
            best_mu_star_truth = mu_star_truth

        if sample_val < best_sampled_val:
            best_sampled_val = sample_val
            best_sampled_point = point_to_sample
            if(truth_IS == sample_IS):
                truth_at_best_sampled = sample_val
            else:
                truth_at_best_sampled = obj_func_min.evaluate(truth_IS, best_sampled_point)

        # NOTE: while Jialei worked everywhere with the values of the minimization problem in the computation, he used the maximization obj values for the GP.
        # That is why here sample_val is multiplied by -1
        kg_gp_cpp.add_sampled_points([SamplePoint(numpy.concatenate(([sample_IS], point_to_sample)), -sample_val, noise_and_cost_func(sample_IS, point_to_sample)[0])])
        total_cost += cost
        best_this_itr = min(best_mu_star_truth, truth_at_best_sampled)
        print "sample IS {0}, value {1}, best sampled val {2}".format(sample_IS, sample_val, best_sampled_val)
        print "mu star truth {0}".format(mu_star_truth)
        print "truth_at_best_sampled {0}".format(truth_at_best_sampled)
        print "best so far {0}".format(best_this_itr)
        print "total cost {0}".format(total_cost)
        list_best.append(best_this_itr)
        list_cost.append(total_cost)

        # save data from this iteration:
        list_sampled_IS.append(sample_IS)
        list_sampled_points.append(point_to_sample)
        list_noise_variance_at_sample.append(noise_and_cost_func(sample_IS, point_to_sample)[0])
        # NOTE: while Jialei worked everywhere with the values of the minimization problem in the computation, he used the maximization obj values for the GP.
        # but here we store the value of the min problem
        list_sampled_vals.append(sample_val)
        list_raw_voi.append(raw_voi)
        list_mu_star_truth.append(mu_star_truth)

    # # write results to MySQL table
    # best_so_far_table = pandas.DataFrame(numpy.array(list_best).reshape((1,-1)))
    # best_so_far_table.to_sql(best_so_far_table_name, sql_util.sql_engine, if_exists='append', index=False)
    # cost_so_far_table = pandas.DataFrame(numpy.array(list_cost).reshape((1,-1)))
    # cost_so_far_table.to_sql(cost_so_far_table_name, sql_util.sql_engine, if_exists='append', index=False)

    # store data from this replication as dictionary, and append it to list
        data_to_pickle = {"points": list_sampled_points,
                          "vals": list_sampled_vals,
                          "noise_variance": list_noise_variance_at_sample,
                          "sampledIS": list_sampled_IS,
                          "best": list_best,
                          "cost": list_cost,
                          "raw_voi": list_raw_voi,
                          "mu_star_truth": list_mu_star_truth,
                          "init_best_truth": init_best_truth,
                          }


        # write data to pickle.
        with open("{0}/{1}.pickle".format(path_to_pickle_result, filename_to_pickle), "wb") as output_file:
            pickle.dump(data_to_pickle, output_file)
