import os
import sys
import cPickle as pickle    # pickle is actually used

from moe.optimal_learning.python.cpp_wrappers.covariance import MixedSquareExponential as cppMixedSquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcessNew
from moe.optimal_learning.python.data_containers import SamplePoint

sys.path.append("../")

from multifidelity_KG.voi.optimization import *
import sql_util_cs
from mothers_little_helpers import *
from load_and_store_data import load_data_from_a_min_problem, createHistoricalDataForKG, findBestSampledValue, \
    create_listPrevData, match_pickle_filename

from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartRosenbrock.rosenbrock_shifted import RosenbrockShifted
from coldStartRosenbrock.rosenbrock_slightshifted import RosenbrockSlightShifted

from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var1 import AssembleToOrderVar1
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4



'''
Run the coldstart version of misoKG on a benchmark problem.

based on Jialei's run_multiKG.py
'''

__author__ = 'matthiaspoloczek'

### MP: I have commented and refactored Jialei's code, so we should use this version to build on for future versions


### The following parameters must be adapted for each simulator

### Rosenbrock family
# obj_func_min = RosenbrockVanilla( mult=1.0 )
# obj_func_max = RosenbrockVanilla( mult=-1.0 )                     # used by KG
# num_iterations = 25     # iterations per replication
# num_init_pts = 5        # points training data from the current IS

### AssembleToOrderVanilla
obj_func_min = AssembleToOrderVar4( mult=-1.0 )
obj_func_max = AssembleToOrderVar4( )                     # used by KG
num_iterations = 50     # iterations per replication
num_init_pts = 1

func_name = obj_func_min.getFuncName()
numIS = obj_func_min.getNumIS()
truth_IS = obj_func_min.getTruthIS()
num_replications = 50    # how many replications

# list of strings that give the filenames of the datasets to be loaded as additional IS
#TODO choose correct previous data for instance
list_previous_datasets_to_load = [] #['data_vKG_rb_sinN'] #['data_vKG_rb_van'] # []
print "previous datasets: {0}".format(list_previous_datasets_to_load)

pathToPickles = "../pickles/csCentered"
init_data_pickle_filename_prefix = func_name + '_IS_0_' + str(num_init_pts) + '_points_each_repl_'



# less important params
exploitation_threshold = 1e-5
num_x_prime = 3000
num_discretization_before_ranking = num_x_prime * 3
num_threads = 64
num_multistart = 64
num_candidate_start_points = 500
### end parameter

# perform num_replications replications, each replication performs num_iterations many steps
for replication_no in range(num_replications):

    # create list of previous datasets
    all_filenames = os.listdir(pathToPickles)
    list_previous_dataset_names_for_this_replication = []
    for name_prefix in list_previous_datasets_to_load:
        list_previous_dataset_names_for_this_replication.append(match_pickle_filename(all_filenames, name_prefix+"_repl", replication_no))
    listPrevData = create_listPrevData(obj_func_min, list_previous_dataset_names_for_this_replication,
                                       replication_no, pathToPickles, init_data_pickle_filename_prefix)

    # Select the best IS or use only IS 0
    run_acquisition_on_IS_0_only = True
    exploitation_IS = truth_IS
    number_of_previous_datasets = len(listPrevData) - 1
    #print 'number_of_previous_datasets = ' + str(number_of_previous_datasets)

    func_names_prev_datasets = []
    for prev_dataset_name in list_previous_dataset_names_for_this_replication:
        func_names_prev_datasets.append(prev_dataset_name.split("_")[-3])
    names_used_datasets = "" if len(func_names_prev_datasets) == 0 else "_w_{0}".format("_".join(func_names_prev_datasets))
    filename_to_pickle = "data_vKG_{0}{1}_repl_{2}".format(func_name, names_used_datasets, replication_no)

    # name the tables
    mysql_hypers_table_name = "{0}_hyper_{1}{2}".format("vkg" if len(func_names_prev_datasets) == 0 else "cskg", func_name, names_used_datasets)
    best_so_far_table_name = "{0}_{1}{2}_best".format("vkg" if len(func_names_prev_datasets) == 0 else "cskg", func_name, names_used_datasets)
    cost_so_far_table_name = "{0}_{1}{2}_cost".format("vkg" if len(func_names_prev_datasets) == 0 else "cskg", func_name, names_used_datasets)

    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_max._search_domain])
    noise_and_cost_func = obj_func_min.noise_and_cost_func

    # setup benchmark result container
    #kg_hyper_param = [100.0, 1, 1, 1, 1, 1, 1, 1, 1] # test hypers
    kg_hyper_param = pandas.read_sql_table(mysql_hypers_table_name, sql_util_cs.sql_engine).mean(axis=0).values
    print 'kg_hyper_param' +  str(kg_hyper_param)

    kg_data = createHistoricalDataForKG(obj_func_min._dim, listPrevData)
    ### only consider the initial data of this IS for the best value
    best_sampled_point, best_sampled_val, truth_at_best_sampled = findBestSampledValue(obj_func_min, [listPrevData[0]], truth_IS)
    print 'truth_at_best_sampled = ' + str(truth_at_best_sampled)
    ### Looking at all the previous data sets and sampling the point, that was best for its resp. IS, at the current IS
    #best_sampled_point, best_sampled_val, truth_at_best_sampled = findBestSampledValue(obj_func_min, listPrevData, truth_IS)

    kg_cov_cpp = cppMixedSquareExponential(hyperparameters=kg_hyper_param)
    kg_gp_cpp = GaussianProcessNew(kg_cov_cpp, kg_data, number_of_previous_datasets)
    #here: third arg obj_func_max._num_IS was replaced by number_of_previous_datasets

    # MP: best_so_far and cost_so_far give the best truth seen (resp., the total cost) up to iteration i
    best_so_far = numpy.zeros(num_iterations)
    cost_so_far = numpy.zeros(num_iterations)

    # collect all samples from a single replication: (IS, point, sampled_val, noise_var)
    list_sampled_IS = []
    list_sampled_points = []
    list_sampled_vals = []
    list_noise_variance_at_sample = []

    # MP: How many steps does the acquisition function perform? This value is num_iterations
    for kg_iteration in range(num_iterations):
        print "benchmark {0}, repl {1}, step {2}, best_truth {3}".format(mysql_hypers_table_name, replication_no,
                                                                         kg_iteration, truth_at_best_sampled)
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

        # create a list of IS for which the best KG point is computed. Then the next sample is the best among these optima.
        # Typically, IS 0 is none of these.
        list_IS = range(1,obj_func_max._num_IS + 1)
        if(run_acquisition_on_IS_0_only):
            # for cold start we may wish to only consider IS 0
            list_IS = [0]

        with Parallel(n_jobs=num_threads) as parallel:
            for id_IS in list_IS:
                # the id of the id_IS-th IS is id_IS
                start_points_prepare = search_domain.generate_uniform_random_points_in_domain(num_candidate_start_points)
                kg_vals = parallel(delayed(compute_kg_unit)(x, id_IS) for x in start_points_prepare)
                sorted_idx_kg = numpy.argsort(kg_vals)
                start_points = start_points_prepare[sorted_idx_kg[-num_multistart:], :]
                parallel_results = parallel(delayed(min_kg_unit)(pt, id_IS) for pt in start_points)
                inner_min, inner_min_point = process_parallel_results(parallel_results)
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
            print "mu_star found"
        if -min_negative_kg < exploitation_threshold:
            sample_IS = exploitation_IS
            print "KG search failed, do exploitation"
            point_to_sample = mu_star_point

        predict_mean = kg_gp_cpp.compute_mean_of_points(numpy.concatenate(([0], point_to_sample)).reshape((1,-1)))[0]
        predict_var = kg_gp_cpp.compute_variance_of_points(numpy.concatenate(([0], point_to_sample)).reshape((1,-1)))[0,0]
        cost = noise_and_cost_func(sample_IS, point_to_sample)[1]
        mu_star_var = kg_gp_cpp.compute_variance_of_points(numpy.concatenate(([0], mu_star_point)).reshape((1,-1)))[0,0]

        #TODO evaluate() was called 3 times, but is it 3 times on the same value (if there is only one IS?)
        ### OPTIMIZATION for csKG, where truth_IS = exploitation_IS = sample_IS
        sample_val = obj_func_min.evaluate(sample_IS, point_to_sample)
        if ((point_to_sample == mu_star_point).all and (truth_IS == sample_IS)):
            mu_star_truth = sample_val
        else:
            mu_star_truth = obj_func_min.evaluate(truth_IS, mu_star_point)

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
        best_so_far[kg_iteration] = min(mu_star_truth, truth_at_best_sampled)
        cost_so_far[kg_iteration] = cost if kg_iteration == 0 else (cost + cost_so_far[kg_iteration - 1])

        # save data from this iteration:
        list_sampled_IS.append(sample_IS)
        list_sampled_points.append(point_to_sample)
        list_noise_variance_at_sample.append(noise_and_cost_func(sample_IS, point_to_sample)[0])
        # NOTE: while Jialei worked everywhere with the values of the minimization problem in the computation, he used the maximization obj values for the GP.
        # but here we store the value of the min problem
        list_sampled_vals.append(sample_val)

    # write results to MySQL table
    best_so_far_table = pandas.DataFrame(best_so_far.reshape((1,-1)))
    best_so_far_table.to_sql(best_so_far_table_name, sql_util_cs.sql_engine, if_exists='append', index=False)
    # cost_so_far_table = pandas.DataFrame(cost_so_far.reshape((1,-1)))
    # cost_so_far_table.to_sql(cost_so_far_table_name, sql_util_cs.sql_engine, if_exists='append', index=False)

    # store data from this replication as dictionary, and append it to list
    data_to_pickle = {"points": list_sampled_points, "vals": list_sampled_vals,
                      "noise_variance": list_noise_variance_at_sample,
                      "sampledIS": list_sampled_IS }


    # write data to pickle.
    with open("{0}/{1}.pickle".format(pathToPickles, filename_to_pickle), "wb") as output_file:
        list_data_replications = pickle.dump(data_to_pickle, output_file)
