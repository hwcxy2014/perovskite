import sys
import numpy as np
import pandas as pd

sys.path.append("../")

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pythonTensorProductDomain
from moe.optimal_learning.python.python_version.log_likelihood import GaussianProcessLogMarginalLikelihood
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess as pythonGP

from assembleToOrderExtended import AssembleToOrderExtended
from load_and_store_data import load_data_from_a_min_problem, obtainHistoricalDataForEGO, \
    create_listPrevData, createHistoricalDataForPES, match_pickle_filename
from pes.covariance import ProductKernel
from pes.model import global_optimization_of_GP, scale_back, scale_forward, PESModel, optimize_entropy
from pes.entropy_search import PES
import sql_util_miso

__author__ = 'jialeiwang'

""" For this algorithm, Predicted Entropy Search from the paper Multi-task bayesian optimization, we agree that all IS
are accessible. Unlike misoKG, which assumes IS0 is hidden, here IS0 is no different than the other IS.
"""
"""
IS 0 = truth
IS 1 = biased
IS 2 = unbiased but noisy
"""
### AssembleToOrderVanilla
replication_no = 1
num_itr = 50
truth_is = 0    # extra cautious if you have to modify this, since many places in the code assume truth_is = 0
obj_func_min = AssembleToOrderExtended(mult=-1.0)
obj_func_max = AssembleToOrderExtended()

func_name = obj_func_min.getFuncName()
# how many prev datasets should be incorporated? This determines the dim of the GP and the num of hypers
list_previous_datasets_to_load = []
indexFirstIS = 0  # id of the first IS, used to label data accordingly
separateIS0 = 0  # 1 if IS0 is not observable and hence the data supplied for IS1, IS2, ...
### in coldstart this should be 0

### Load the training data
pathToPickles = "../pickles/miso"
init_data_pickle_filename_prefix = func_name + '_IS_0_1_2_20_points_each_repl_'  # the dataset for IS 0
print init_data_pickle_filename_prefix
# complete_list_prev_datasets_to_load = ["{0}_repl_{1}".format(dataset_name, replication_no) for dataset_name in list_previous_datasets_to_load]
listPrevData = []  # create_listPrevData(obj_func_min, complete_list_prev_datasets_to_load, replication_no, pathToPickles, init_data_pickle_filename_prefix)

init_data_pickle_filename = init_data_pickle_filename_prefix + str(replication_no)
init_pts, init_vals = load_data_from_a_min_problem(pathToPickles, init_data_pickle_filename)
# init_pts and init_vals contain a list of lists of pts/vals, one dataset for each IS

### Load in correct ordering to compute hypers
#################################
### Load truth IS as IS 0
#################################
select_dataset = 2
corresponding_IS = 0
noise_vars = np.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))
select_dataset = 1
corresponding_IS = 2
noise_vars = np.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))
select_dataset = 0
corresponding_IS = 1
noise_vars = np.array([obj_func_min.noise_and_cost_func(corresponding_IS, pt)[0] for pt in init_pts[select_dataset]])
listPrevData.append((init_pts[select_dataset], init_vals[select_dataset], noise_vars))
data = createHistoricalDataForPES(obj_func_max.getDim(), listPrevData, indexFirstIS)


#################################
### mysql table:
hyper_params = pd.read_sql_table('pes_hyper_{0}'.format(obj_func_min.getFuncName()), sql_util_miso.sql_engine)
# hyper_params.iloc[:,6] = np.ones(len(hyper_params.index)) * 1.
# table_name = "pes_{0}_best".format(func_name)
# print "table_name: {0}".format(table_name)

peak_idx = 47
data_val = data._points_sampled_value[peak_idx]
pt = data._points_sampled[peak_idx,:]
min_val = obj_func_min.evaluate(int(pt[0]), pt[1:])
max_val = obj_func_max.evaluate(int(pt[0]), pt[1:])
print "peak idx {0}\npt {1}\npickle data {2}\n obj min {3}\n obj max {4}".format(peak_idx, pt, data_val, min_val, max_val)