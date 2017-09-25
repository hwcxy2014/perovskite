import cPickle as pickle

import numpy

__author__ = 'matthiaspoloczek'

'''
Mother's Little Helpers collects function that will be useful for all parts of the project.
'''

# there is a separate method for each object, since we might want to load some independent of others

def pickle_init_points_for_all_IS(directory, func_name, numIS, init_points_for_all_IS ):
    with open(directory+'/'+ func_name+'_'+str(numIS)+'_init_points_for_all_IS.pickle', "wb+") as output_file:
        pickle.dump( init_points_for_all_IS, output_file )

def pickle_vals(directory, func_name, numIS, vals ):
    with open(directory+'/'+ func_name+'_'+str(numIS)+'_vals.pickle', "wb+") as output_file:
        pickle.dump( vals, output_file )

def pickle_sample_vars(directory, func_name, numIS, sample_vars ):
    with open(directory+'/'+ func_name+'_'+str(numIS)+'_sample_vars.pickle', "wb+") as output_file:
        pickle.dump( sample_vars, output_file )

def pickle_best_initial_value(directory, func_name, numIS, best_initial_value ):
    with open(directory+'/'+ func_name+'_'+str(numIS)+'_best_initial_value.pickle', "wb+") as output_file:
        pickle.dump( best_initial_value, output_file )

def load_init_points_for_all_IS(directory, func_name, numIS):
    with open(directory+'/'+ func_name+'_'+str(numIS)+'_init_points_for_all_IS.pickle', "rb") as input_file:
        init_points_for_all_IS = pickle.load(input_file)
    return init_points_for_all_IS

def load_vals(directory, func_name, numIS):
    with open(directory+'/'+ func_name+'_'+str(numIS)+'_vals.pickle', "rb") as input_file:
        vals = pickle.load(input_file)
    return vals

def load_sample_vars(directory, func_name, numIS):
    with open(directory+'/'+ func_name+'_'+str(numIS)+'_sample_vars.pickle', "rb") as input_file:
        sample_vars = pickle.load(input_file)
    return sample_vars

def load_best_initial_value(directory, func_name, numIS):
    with open(directory+'/'+ func_name+'_'+str(numIS)+'_best_initial_value.pickle', "rb") as input_file:
        best_initial_value = pickle.load(input_file)
    return best_initial_value

def process_parallel_results(parallel_results):
    inner_min = numpy.inf
    for result in parallel_results:
        if inner_min > result[1]:
            inner_min = result[1]
            inner_min_point = result[0]
    return inner_min, inner_min_point


