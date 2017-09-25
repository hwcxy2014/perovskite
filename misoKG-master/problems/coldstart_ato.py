from abc import ABCMeta, abstractproperty
import pickle

from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4
from data_io import construct_hist_data_from_pickle, match_data_filename, check_file_legitimate, get_data_from_s3, send_data_to_s3, construct_hist_data_from_s3, construct_hist_data_from_s3

__author__ = 'jialeiwang'

class ColdstartAtoHyper(object):
    """ Base class for all coldstart ATO training hyperparameters
    """
    __metaclass__ = ABCMeta

    def __init__(self, method_name):
        self.method_name = method_name
        # self.data_dir = "/fs/europa/g_pf/pickles/coldstart/data"
        self._hist_data = None

    @property
    def hist_data(self):
        return self._hist_data

    @property
    def obj_func_min(self):
        return AssembleToOrderVanilla(mult=-1.0)

    @property
    def hyper_path(self):
        # return "/fs/europa/g_pf/pickles/coldstart/hyper/{0}_ato.pickle".format(self.method_name)
        return "coldstart/hyper/{0}_ato".format(self.method_name)

class ColdstartAtoHyperEgo(ColdstartAtoHyper):

    def __init__(self, bucket):
        super(ColdstartAtoHyperEgo, self).__init__("ego")
        # self._hist_data = construct_hist_data_from_pickle(dim=8, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_atoC_vanilla"}, combine_IS=False, sign=1.0)[0]
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"coldstart/data/hyper_1000_points_atoC_vanilla"}, combine_IS=False, sign=1.0)[0]

class ColdstartAtoHyperKg(ColdstartAtoHyper):

    def __init__(self, bucket):
        super(ColdstartAtoHyperKg, self).__init__("kg")
        # self._hist_data = construct_hist_data_from_pickle(dim=8, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_atoC_vanilla"}, combine_IS=False, sign=-1.0)[0]
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"coldstart/data/hyper_1000_points_atoC_vanilla"}, combine_IS=False, sign=-1.0)[0]

class ColdstartAtoHyperMkg(ColdstartAtoHyper):

    def __init__(self, bucket):
        super(ColdstartAtoHyperMkg, self).__init__("mkg")
        # self._hist_data = construct_hist_data_from_pickle(dim=8, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_atoC_vanilla", 1:"hyper_1000_points_atoC_var3", 2:"hyper_1000_points_atoC_var4"}, combine_IS=False, sign=-1.0, take_diff=True, primary_key=0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8,
                                                      IS_key_dict={0:"coldstart/data/hyper_1000_points_atoC_vanilla", 1:"coldstart/data/hyper_1000_points_atoC_var3", 2:"coldstart/data/hyper_1000_points_atoC_var4"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class ColdstartAtoHyperSeqpes(ColdstartAtoHyper):

    def __init__(self, bucket):
        super(ColdstartAtoHyperSeqpes, self).__init__("seqpes")
        # self._hist_data = construct_hist_data_from_pickle(dim=8, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_atoC_vanilla"}, combine_IS=True, sign=1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"coldstart/data/hyper_1000_points_atoC_vanilla"}, combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 1

class ColdstartAtoHyperPes(ColdstartAtoHyper):

    def __init__(self, bucket):
        super(ColdstartAtoHyperPes, self).__init__("pes")
        # self._hist_data = construct_hist_data_from_pickle(dim=8, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_atoC_vanilla", 1:"hyper_1000_points_atoC_var3", 2:"hyper_1000_points_atoC_var4"}, combine_IS=True, sign=1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"coldstart/data/hyper_1000_points_atoC_vanilla", 1:"coldstart/data/hyper_1000_points_atoC_var3", 2:"coldstart/data/hyper_1000_points_atoC_var4"}, combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 3

class ColdstartAtoBenchmark(object):
    """ Base class for all coldstart rosenbrock benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, obj_func_idx, replication_no, method_name, bucket):
        self.func_list = [AssembleToOrderVanilla(mult=-1.0), AssembleToOrderVar2(mult=-1.0), AssembleToOrderVar3(mult=-1.0), AssembleToOrderVar4(mult=-1.0)]
        # self.data_dir = "/fs/europa/g_pf/pickles/coldstart/data"
        # self.hyper_dir = "/fs/europa/g_pf/pickles/coldstart/hyper"
        self.obj_func_idx = obj_func_idx
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket

    @abstractproperty
    def num_is_in(self):
        pass

    @property
    def hist_data(self):
        return self._hist_data

    @property
    def hyper_param(self):
        data = get_data_from_s3(self._bucket, "coldstart/hyper/{0}_ato".format(self.method_name))
        return data['hyperparam']

    @property
    def obj_func_min(self):
        return self.func_list[self.obj_func_idx]

    @property
    def result_path(self):
        return "coldstart/result/{0}_{1}_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)

    @property
    def data_path(self):
        return "coldstart/data/{0}_{1}_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)

    @property
    def num_iterations(self):
        return 50

    @property
    def truth_is(self):
        return 0

    @property
    def exploitation_is(self):
        return 0

    @property
    def list_sample_is(self):
        return [0]

class ColdstartAtoBenchmarkEgo(ColdstartAtoBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartAtoBenchmarkEgo, self).__init__(obj_func_idx, replication_no, "ego", bucket)
        # self._hist_data = construct_hist_data_from_pickle(dim=8, directory=self.data_dir, IS_filename_dict={0:"{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=False, sign=1.0)[0]
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=False, sign=1.0)[0]

    @property
    def num_is_in(self):
        return None

class ColdstartAtoBenchmarkKg(ColdstartAtoBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartAtoBenchmarkKg, self).__init__(obj_func_idx, replication_no, "kg", bucket)
        # self._hist_data = construct_hist_data_from_pickle(dim=8, directory=self.data_dir, IS_filename_dict={0:"{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=True, sign=-1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 0

class ColdstartAtoBenchmarkMkg(ColdstartAtoBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartAtoBenchmarkMkg, self).__init__(obj_func_idx, replication_no, "mkg", bucket)
        prev_idx_1 = self.obj_func_idx - 2
        prev_idx_2 = self.obj_func_idx - 1
        prev_data_filename_1 = "coldstart/data/kg_{0}_repl_{1}".format(self.func_list[prev_idx_1 if prev_idx_1 >= 0 else prev_idx_1+4].getFuncName(), self.replication_no)
        prev_data_filename_2 = "coldstart/data/kg_{0}_repl_{1}".format(self.func_list[prev_idx_2 if prev_idx_2 >= 0 else prev_idx_2+4].getFuncName(), self.replication_no)
        # self._hist_data = construct_hist_data_from_pickle(dim=8, directory=self.data_dir, IS_filename_dict={0:"{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no), 1:prev_data_filename_1, 2:prev_data_filename_2}, combine_IS=True, sign=-1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no), 1:prev_data_filename_1, 2:prev_data_filename_2}, combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 2

class ColdstartAtoBenchmarkSeqpes(ColdstartAtoBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartAtoBenchmarkSeqpes, self).__init__(obj_func_idx, replication_no, "seqpes", bucket)
        # self._hist_data = construct_hist_data_from_pickle(dim=8, directory=self.data_dir, IS_filename_dict={0:"{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=True, sign=1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 1

class ColdstartAtoBenchmarkPes(ColdstartAtoBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartAtoBenchmarkPes, self).__init__(obj_func_idx, replication_no, "pes", bucket)
        prev_idx_1 = self.obj_func_idx - 2
        prev_idx_2 = self.obj_func_idx - 1
        prev_data_filename_1 = "coldstart/data/seqpes_{0}_repl_{1}".format(self.func_list[prev_idx_1 if prev_idx_1 >= 0 else prev_idx_1+4].getFuncName(), self.replication_no)
        prev_data_filename_2 = "coldstart/data/seqpes_{0}_repl_{1}".format(self.func_list[prev_idx_2 if prev_idx_2 >= 0 else prev_idx_2+4].getFuncName(), self.replication_no)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no), 1:prev_data_filename_1, 2:prev_data_filename_2}, combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 3

class_collection = {
    "coldstart_ato_hyper_ego": ColdstartAtoHyperEgo,
    "coldstart_ato_hyper_kg": ColdstartAtoHyperKg,
    "coldstart_ato_hyper_mkg": ColdstartAtoHyperMkg,
    "coldstart_ato_hyper_seqpes": ColdstartAtoHyperSeqpes,
    "coldstart_ato_hyper_pes": ColdstartAtoHyperPes,
    "coldstart_ato_benchmark_ego": ColdstartAtoBenchmarkEgo,
    "coldstart_ato_benchmark_kg": ColdstartAtoBenchmarkKg,
    "coldstart_ato_benchmark_mkg": ColdstartAtoBenchmarkMkg,
    "coldstart_ato_benchmark_seqpes": ColdstartAtoBenchmarkSeqpes,
    "coldstart_ato_benchmark_pes": ColdstartAtoBenchmarkPes,
}
