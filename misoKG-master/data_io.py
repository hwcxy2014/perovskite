import numpy as np
import os
import pickle

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain

__author__ = 'jialeiwang'

def get_data_from_s3(bucket, key):
    s3_key = bucket.get_key(key)
    if s3_key is None:
        raise ValueError("key not found")
    data = pickle.loads(s3_key.get_contents_as_string())
    return data

def send_data_to_s3(bucket, key, data):
    s3_key = bucket.get_key(key)
    if s3_key is None:
        s3_key = bucket.new_key(key)
    s3_key.set_contents_from_string(pickle.dumps(data))

def gen_data_to_pickle(directory, obj_func_min, num_pts, which_IS, filename):
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain])
    points = search_domain.generate_uniform_random_points_in_domain(num_pts)
    vals = [obj_func_min.evaluate(which_IS, pt) for pt in points]
    noise = obj_func_min.noise_and_cost_func(which_IS, None) * np.ones(num_pts)
    data = {"points": points, "vals": vals, "noise": noise}
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def construct_hist_data_from_pickle(dim, directory, IS_filename_dict, combine_IS, sign, take_diff=False, primary_key=None):
    """
    :param dim: space dimension of the problem
    :type dim: int
    :param directory: dir of the pickle files
    :type directory: str
    :param IS_filename_dict: {IS: filename} hashtable which provides name of the pickle file for the corresponding IS
    :type IS_filename_dict: dict
    :param combine_IS: whether construct a single HistoricalData on the space IS \times space, or a dict of HistoricalData
    objects, with each corresponds to each IS
    :type combine_IS: bool
    :param sign: sign = 1.0 means minimization problem, otherwise is maximization
    :type sign: float
    :param take_diff: whether take diff between IS_i and primary_IS, this is enabled for one approach of estimating mKG
    hyperparameters
    :type take_diff: bool
    :param primary_key: if take_diff = True, this is used to specify primary IS
    :type primary_key: int
    :return: if combine_IS = True, return a HistoricalData object, otherwise return a dict of {IS: HistoricalData}
    :rtype: HistoricalData or dict
    """
    points_dict = {}
    vals_dict = {}
    noise_dict = {}
    if take_diff:
        with open("{0}/{1}.pickle".format(directory, IS_filename_dict[primary_key]), "rb") as f:
            data = pickle.load(f)
            points_dict[primary_key] = np.array(data['points'])
            vals_dict[primary_key] = sign * np.array(data['vals'])
            noise_dict[primary_key] = np.array(data['noise'])
    for key in IS_filename_dict:
        if take_diff and key != primary_key:
            with open("{0}/{1}.pickle".format(directory, IS_filename_dict[key]), "rb") as f:
                data = pickle.load(f)
                assert np.array_equal(data['points'], points_dict[primary_key]), "inconsistent points, cannot take diff!"
                points_dict[key] = np.array(data['points'])
                vals_dict[key] = sign * np.array(data['vals']) - vals_dict[primary_key]
                noise_dict[key] = np.array(data['noise']) + noise_dict[primary_key]
        elif not take_diff:
            with open("{0}/{1}.pickle".format(directory, IS_filename_dict[key]), "rb") as f:
                data = pickle.load(f)
                points_dict[key] = np.array(data['points'])
                vals_dict[key] = sign * np.array(data['vals'])
                noise_dict[key] = np.array(data['noise'])
    if combine_IS:
        to_return = HistoricalData(dim=dim+1)
        for key in points_dict:
            num_data = len(vals_dict[key])
            to_return.append_historical_data(np.hstack((key*np.ones(num_data).reshape((-1,1)), points_dict[key])), vals_dict[key], noise_dict[key])
    else:
        to_return = {}
        for key in points_dict:
            to_return[key] = HistoricalData(dim=dim)
            to_return[key].append_historical_data(points_dict[key], vals_dict[key], noise_dict[key])
    return to_return

def gen_data_to_s3(bucket, obj_func_min, num_pts, which_IS, key):
    search_domain = pythonTensorProductDomain([ClosedInterval(bound[0], bound[1]) for bound in obj_func_min._search_domain])
    points = search_domain.generate_uniform_random_points_in_domain(num_pts)
    vals = [obj_func_min.evaluate(which_IS, pt) for pt in points]
    noise = obj_func_min.noise_and_cost_func(which_IS, None) * np.ones(num_pts)
    data = {"points": points, "vals": vals, "noise": noise}
    send_data_to_s3(bucket, key, data)

def construct_hist_data_from_s3(bucket, dim, IS_key_dict, combine_IS, sign, take_diff=False, primary_IS=None):
    """
    :param bucket: amazon s3 bucket object
    :param dim: space dimension of the problem
    :type dim: int
    :param IS_key_dict: {IS: key} hashtable which provides key of the data for the corresponding IS
    :type IS_key_dict: dict
    :param combine_IS: whether construct a single HistoricalData on the space IS \times space, or a dict of HistoricalData
    objects, with each corresponds to each IS
    :type combine_IS: bool
    :param sign: sign = 1.0 means minimization problem, otherwise is maximization
    :type sign: float
    :param take_diff: whether take diff between IS_i and primary_IS, this is enabled for one approach of estimating mKG
    hyperparameters
    :type take_diff: bool
    :param primary_key: if take_diff = True, this is used to specify primary IS
    :type primary_key: int
    :return: if combine_IS = True, return a HistoricalData object, otherwise return a dict of {IS: HistoricalData}
    :rtype: HistoricalData or dict
    """
    points_dict = {}
    vals_dict = {}
    noise_dict = {}
    if take_diff:
        data = get_data_from_s3(bucket, IS_key_dict[primary_IS])
        points_dict[primary_IS] = np.array(data['points'])
        vals_dict[primary_IS] = sign * np.array(data['vals'])
        noise_dict[primary_IS] = np.array(data['noise'])
    for IS in IS_key_dict:
        if take_diff and IS != primary_IS:
            data = get_data_from_s3(bucket, IS_key_dict[IS])
            assert np.array_equal(data['points'], points_dict[primary_IS]), "inconsistent points, cannot take diff!"
            points_dict[IS] = np.array(data['points'])
            vals_dict[IS] = sign * np.array(data['vals']) - vals_dict[primary_IS]
            noise_dict[IS] = np.array(data['noise']) + noise_dict[primary_IS]
        elif not take_diff:
            data = get_data_from_s3(bucket, IS_key_dict[IS])
            points_dict[IS] = np.array(data['points'])
            vals_dict[IS] = sign * np.array(data['vals'])
            noise_dict[IS] = np.array(data['noise'])
    if combine_IS:
        to_return = HistoricalData(dim=dim+1)
        for IS in points_dict:
            num_data = len(vals_dict[IS])
            to_return.append_historical_data(np.hstack((IS*np.ones(num_data).reshape((-1,1)), points_dict[IS])), vals_dict[IS], noise_dict[IS])
    else:
        to_return = {}
        for IS in points_dict:
            to_return[IS] = HistoricalData(dim=dim)
            to_return[IS].append_historical_data(points_dict[IS], vals_dict[IS], noise_dict[IS])
    return to_return

def match_data_filename(dir, target_filename_prefix, target_idx, check_legitimate, **kwargs):
    """ If the file ``target_filename_prefix_idx`` is broken (does not have enough data because previous run was problematic,
    in particular, it happens for entropy search where you might encounter ill conditioned cov matrix
    :param dir:
    :param target_filename_prefix:
    :param target_idx:
    :param check_legitimate: the function that returns True if the file path is legitimate in some sense (defined by this checking function)
    :return:
    """
    all_filenames = os.listdir(dir)
    if "{0}/{1}_{2}.pickle".format(dir, target_filename_prefix, target_idx) in all_filenames and check_legitimate("{0}/{1}_{2}.pickle".format(dir, target_filename_prefix, target_idx), **kwargs):
        return "{0}_{1}".format(target_filename_prefix, target_idx)
    file_seq = np.random.permutation(range(len(all_filenames)))
    for seq in file_seq:
        if all_filenames[seq].find(target_filename_prefix) >= 0 and check_legitimate("{0}/{1}".format(dir, all_filenames[seq]), **kwargs):
            return all_filenames[seq][:-7]

def check_file_legitimate(file_path, num_data):
    with open(file_path, 'rb') as f:
        d = pickle.load(f)
        return len(d['vals']) == num_data

def process_parallel_results(parallel_results):
    inner_min = np.inf
    for result in parallel_results:
        if inner_min > result[1]:
            inner_min = result[1]
            inner_min_point = result[0]
    return inner_min, inner_min_point
