import os
import cPickle as pickle
import pandas as pd

import numpy

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain


__author__ = 'matthiaspoloczek'

'''
Methods to load and store previous datasets. And to create HistoricalData Objects for the different acquisition functions
'''


def load_data_from_a_min_problem(directory, filename):
    """
    :param directory: dir of pickle file
    :param filename: name of pickle file
    :return: init_points_for_all_IS, init_vals_for_all_IS
    """
    with open("{0}/{1}.pickle".format(directory, filename), "rb") as input_file:
        data = pickle.load(input_file)
        return data['points'], data['vals']


def obtainHistoricalDataForEGO(load_historical_data_from_pickle, obj_func_min, directoryToPickles,
                               list_IS_to_query, num_init_pts_each_IS, init_data_pickle_filename=''):
    '''
    Create Historical Data object for EGO that contains initial data.
    If truthIS is among the IS, then load only the data from that one
    Args:
        load_historical_data_from_pickle: if True load from pickle otherwise do a random Latin hypercube design
        obj_func_min: the problem
        directoryToPickles: path to the directory that contains the pickle files
        list_IS_to_query: list of the IS that should be queried, e.g. [0, 1, 2]
        num_init_pts_each_IS: how many points for each IS - is either used to find right pickle or to determine the number of points to sample
        init_data_pickle_filename: optional parameter that gives the filename of the pickle to load

    Returns: HistoricalData object

    '''
    historical_data = HistoricalData(obj_func_min._dim)
    if (load_historical_data_from_pickle):
        # To load the pickled data, do:
        if(init_data_pickle_filename == ''):
            init_data_pickle_filename = obj_func_min.getFuncName() + '_' + 'IS_' \
                                        + '_'.join(str(element) for element in list_IS_to_query) + '_' \
                                        + str(num_init_pts_each_IS) + "_points_each"
        init_pts_array, init_vals_array = load_data_from_a_min_problem(directoryToPickles, init_data_pickle_filename)

        # if truthIS is among the sampled, then load only that one:
        if obj_func_min.getTruthIS() in list_IS_to_query:
            indexArray = list_IS_to_query.index( obj_func_min.getTruthIS() )
            sample_vars = [obj_func_min.noise_and_cost_func(obj_func_min.getTruthIS(), pt)[0] for pt in init_pts_array[indexArray]]
            historical_data.append_historical_data(init_pts_array[indexArray], init_vals_array[indexArray], sample_vars)
        else:
            # load data for all IS
            indexArray = 0
            for index_IS in list_IS_to_query:
                sample_vars = [obj_func_min.noise_and_cost_func(index_IS, pt)[0] for pt in init_pts_array[indexArray]]
                historical_data.append_historical_data(init_pts_array[indexArray], init_vals_array[indexArray], sample_vars)
                indexArray += 1
    else:
        # generate initial data from querying random points for each IS
        for index_IS in list_IS_to_query:
            if (obj_func_min.getTruthIS() in list_IS_to_query) and (index_IS != obj_func_min.getTruthIS()):
                continue # the truthIS is observed but this is another IS: skip!

            search_domain = pythonTensorProductDomain(
                [ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain])
            pts = search_domain.generate_uniform_random_points_in_domain(num_init_pts_each_IS)
            vals = [obj_func_min.evaluate(index_IS, pt) for pt in pts]
            sample_vars = [obj_func_min.noise_and_cost_func(index_IS, pt)[0] for pt in pts]
            historical_data.append_historical_data(pts, vals, sample_vars)

    return historical_data


def createHistoricalDataGeneral(dim_obj_func_min, listPrevData, mult, indexFirstIS = 0):
    '''

    Args:
        dim_obj_func_min: dim of the obj function, as given in obj_func_min._dim
        listPrevData: list of tuples (data, vals, noise)
        indexFirstIS: what is the number of the first IS given in listPrevData. Others are numbered consecutively

    Returns: HistoricalData object for KG (with additional first column that gives the IS the data corresponds to

    '''
    data = HistoricalData(dim_obj_func_min + 1)
    indexIS = indexFirstIS # this is the number that corresponds to the IS-dimension in the GP
    for dataset in listPrevData:
        # add first column that gives the IS the data corresponds to
        IS_pts = numpy.hstack((indexIS * numpy.ones(len(dataset[0])).reshape((-1, 1)), dataset[0]))

        # multiply all values by -1 since we assume that the training data stems from the minimization version
        # but misoKG uses the maximization version
        vals = mult * numpy.array(dataset[1])
        data.append_historical_data(IS_pts, vals, dataset[2])
        indexIS +=1
    return data

def createHistoricalDataForKG(dim_obj_func_min, listPrevData, indexFirstIS = 0):
    return createHistoricalDataGeneral(dim_obj_func_min, listPrevData, mult=-1.0, indexFirstIS=indexFirstIS)

def createHistoricalDataForPES(dim_obj_func_min, listPrevData, indexFirstIS = 0):
    return createHistoricalDataGeneral(dim_obj_func_min, listPrevData, mult=1.0, indexFirstIS=indexFirstIS)

def createHistoricalDataForMisoEI(dim_obj_func_min, listPrevData, directory, bias_filename):
    """ Note: since misoEI uses notion of fidelity variance, I set it to noise_var + bias^2, where bias is estimated
    from biasData
    :param dim_obj_func_min:
    :param listPrevData:
    :return:
    """
    with open("{0}/{1}.pickle".format(directory, bias_filename), "rb") as input_file:
        bias_data = pickle.load(input_file)
    bias_sq_list = numpy.power(numpy.concatenate(([0.], [numpy.mean(bias_data['vals'][i]) for i in range(len(listPrevData)-1)])), 2.0)
    data_list = []
    for i, dataset in enumerate(listPrevData):
        data = HistoricalData(dim_obj_func_min)
        data.append_historical_data(dataset[0], dataset[1], numpy.array(dataset[2]) + bias_sq_list[i])
        data_list.append(data)
    return data_list, bias_sq_list

def createHistoricalDataForMisoKGDiff(dim_obj_func_min, listPrevData, directory, bias_filename, mult=-1.0):
    """ This data is only used to train mKG hyperparams, and suppose listPrevData[0] is unbiased IS
    :param dim_obj_func_min:
    :param listPrevData:
    :param directory:
    :param bias_filename:
    :return:
    """
    with open("{0}/{1}.pickle".format(directory, bias_filename), "rb") as input_file:
        bias_data = pickle.load(input_file)
    data_IS0 = HistoricalData(dim_obj_func_min)
    data_IS0.append_historical_data(listPrevData[0][0], mult * numpy.array(listPrevData[0][1]), numpy.array(listPrevData[0][2]))
    data_list = [data_IS0]
    for i in range(len(listPrevData)-1):
        data = HistoricalData(dim_obj_func_min)
        data.append_historical_data(bias_data['points'][i][:200,:], mult * numpy.array(bias_data['vals'][i][:200]), numpy.ones(len(bias_data['vals'][i][:200])) * (numpy.mean(listPrevData[0][2]) + numpy.mean(listPrevData[i+1][2])))
        data_list.append(data)
    return data_list

def findBestSampledValue(obj_func_min, listPrevData, truth_IS):
    '''
    Find the best objective value (for the truth IS) in all the previous data sets.
    Note that this should only be called on data sets that stem from this IS.
    Args:
        obj_func_min: the simulator object
        listPrevData: list of tuples (data, vals, noise)
        truth_IS: number of the truth IS for the object obj_func_min, typically 0

    Returns: best_sampled_point, best_sampled_val, truth_at_best_sampled

    '''
    best_sampled_val = numpy.inf

    for dataset in listPrevData:
        # find the best initial value
        if numpy.amin(dataset[1]) < best_sampled_val:
            best_sampled_val = numpy.amin(dataset[1])
            best_sampled_point = dataset[0][numpy.argmin(dataset[1])]

    truth_at_best_sampled = obj_func_min.evaluate(truth_IS, best_sampled_point)
    return best_sampled_point, best_sampled_val, truth_at_best_sampled



def findBestSampledValueFromHistoricalData(obj_func_min, historicaldata):
    '''
    Locate the best sampled value in a historical data object

    Args:
        obj_func_min: the simulator object
        historicaldata: the historicaldata object
        truth_IS: number of the truth IS for the object obj_func_min

    Returns: best_sampled_point, best_sampled_val, truth_at_best_sampled
    '''

    best_sampled_val = numpy.inf
    best_sampled_point = []
    for point in historicaldata.to_list_of_sample_points():
        # print "point['point'] = " + str(point[0])
        # print "point['value'] = " + str(point[1])
        if(point[1] < best_sampled_val):
            best_sampled_val = point[1]
            best_sampled_point = point[0]
            # print best_sampled_point

    truth_at_best_sampled = obj_func_min.evaluate(obj_func_min.getTruthIS(), best_sampled_point)
    # print best_sampled_point
    # print best_sampled_val
    # print truth_at_best_sampled
    return best_sampled_point, best_sampled_val, truth_at_best_sampled




def create_listPrevData(obj_func_min, list_previous_datasets_to_load, replication_no, pathToPickles, init_data_pickle_filename_prefix):
    '''

    Args:
        obj_func_min:
        list_previous_datasets_to_load:
        replication_no:
        pathToPickles:
        init_data_pickle_filename_prefix:

    Returns:

    '''
    listPrevData = []
    # append data for IS0 (initial data for source to be queried, IS1, IS2, ...
    # Load initial data from pickle
    index_IS = 0
    init_data_pickle_filename = init_data_pickle_filename_prefix + str(replication_no)
    init_pts, init_vals = load_data_from_a_min_problem(pathToPickles, init_data_pickle_filename)
    noise_vars = numpy.array([obj_func_min.noise_and_cost_func(index_IS, pt)[0] for pt in init_pts[index_IS]])
    listPrevData.append((init_pts[index_IS], init_vals[index_IS], noise_vars))
    # print 'init_pts = ' + str(init_pts)
    # print 'init_vals = ' + str(init_vals)
    # print 'noise_vars = ' + str(noise_vars)
    # print listPrevData
    # exit()
    # Load data from previous runs on related instances. This will be additional IS for the GP
    for filename_previous_dataset in list_previous_datasets_to_load:
        with open("{0}/{1}.pickle".format(pathToPickles, filename_previous_dataset), "rb") as input_file:
            list_data_replications = pickle.load(input_file)

            # TODO filter loaded data to imitate stopping the evaluation when the resp. optimum is reached
            listPrevData.append(
                (list_data_replications['points'],
                 list_data_replications['vals'],
                 list_data_replications['noise_variance'])
            )
        '''
        list_data_replications gives a list of dictionaries, one for each replication:
        {"points": list_sampled_points, "vals": list_sampled_vals,
                                                     "noise_variance": list_noise_variance_at_sample,
                                                     "sampledIS": list_sampled_IS }
        '''

    return listPrevData

def match_pickle_filename(all_filenames, target_prefix, target_idx):
    """ Given target filename prefix and idx, whose format is prefix_idx.pickle, return the target pickle filename
    if exists, otherwise return the most recently generated one, i.e., index is biggest s.t < target_idx
    :param all_filenames:
    :param target_prefix:
    :param target_idx:
    :return:
    """
    all_valid_names = []
    for filename in all_filenames:
        splits = filename.split('.')
        if len(splits) == 2 and splits[1] == "pickle":
            all_valid_names.append(splits[0])
    if len(all_valid_names) == 0:
        raise RuntimeError("no pickle files in the directory!")
    idx_list = []
    for name in all_valid_names:
        if name.find('_repl_') >= 0:
            idx = int(name.split('_')[-1])
            prefix = "_".join(name.split('_')[:-1])
            if prefix == target_prefix:
                if idx == target_idx:
                    return "{0}_{1}".format(target_prefix, target_idx)
                else:
                    idx_list.append(idx)
    idx_list = numpy.array(idx_list, dtype=numpy.int64)
    trunc_ordered_idx_list = numpy.sort(idx_list[idx_list < target_idx])
    if len(trunc_ordered_idx_list) == 0:
        raise RuntimeError("no matching pickle files in the directory!")
    return "{0}_{1}".format(target_prefix, trunc_ordered_idx_list[-1])

def create_summary(dir, target_prefix):
    """ Return pd.DataFrame that summarizes the target experiment
    :param dir: directory
    :param target_prefix: the filenames should match target_prefix_*.pickle where * is the repl no.
    :return: pd.DataFrame
    """
    pickle_filenames = []
    data = None
    for filename in os.listdir(dir):
        if filename.find(target_prefix) >= 0 and filename.find(".pickle") >= 0:
            pickle_filenames.append(filename)
    for filename in pickle_filenames:
        repl_no = int(filename.split("_")[-1][:-7])
        print repl_no
        with open(dir+'/'+filename, 'rb') as f:
            d = pickle.load(f)
            if data is None:
                data = pd.DataFrame(columns=d.keys())
            for col_name in d.keys():
                print col_name
                data.loc[repl_no, col_name] = d[col_name]
    return data
