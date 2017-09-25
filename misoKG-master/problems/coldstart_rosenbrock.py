from abc import ABCMeta, abstractproperty
import pickle

from coldStartRosenbrock.rosenbrock_vanilla import RosenbrockVanilla
from coldStartRosenbrock.rosenbrock_sinus import RosenbrockSinus
from coldStartRosenbrock.rosenbrock_biased import RosenbrockBiased
from coldStartRosenbrock.rosenbrock_slightshifted import RosenbrockSlightShifted
from data_io import construct_hist_data_from_pickle, match_data_filename, check_file_legitimate, construct_hist_data_from_s3, get_data_from_s3

__author__ = 'jialeiwang'

class ColdstartRosenbrockHyper(object):
    """ Base class for all coldstart rosenbrock training hyperparameters
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
        return RosenbrockVanilla(mult=1.0)

    @property
    def hyper_path(self):
        # return "/fs/europa/g_pf/pickles/coldstart/hyper/{0}_rb.pickle".format(self.method_name)
        return "coldstart/hyper/{0}_rb.pickle".format(self.method_name)

class ColdstartRosenbrockHyperEgo(ColdstartRosenbrockHyper):

    def __init__(self, bucket):
        super(ColdstartRosenbrockHyperEgo, self).__init__("ego")
        # self._hist_data = construct_hist_data_from_pickle(dim=2, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_rbCvanN"}, combine_IS=False, sign=1.0)[0]
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/hyper_1000_points_rbCvanN"}, combine_IS=False, sign=1.0)[0]

class ColdstartRosenbrockHyperKg(ColdstartRosenbrockHyper):

    def __init__(self, bucket):
        super(ColdstartRosenbrockHyperKg, self).__init__("kg")
        # self._hist_data = construct_hist_data_from_pickle(dim=2, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_rbCvanN"}, combine_IS=False, sign=-1.0)[0]
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/hyper_1000_points_rbCvanN"}, combine_IS=False, sign=-1.0)[0]

class ColdstartRosenbrockHyperMkg(ColdstartRosenbrockHyper):

    def __init__(self, bucket):
        super(ColdstartRosenbrockHyperMkg, self).__init__("mkg")
        # self._hist_data = construct_hist_data_from_pickle(dim=2, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_rbCvanN", 1:"hyper_1000_points_rbCbiasN"}, combine_IS=False, sign=-1.0, take_diff=True, primary_key=0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/hyper_1000_points_rbCvanN", 1:"coldstart/data/hyper_1000_points_rbCbiasN"}, combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class ColdstartRosenbrockHyperSeqpes(ColdstartRosenbrockHyper):

    def __init__(self, bucket):
        super(ColdstartRosenbrockHyperSeqpes, self).__init__("seqpes")
        # self._hist_data = construct_hist_data_from_pickle(dim=2, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_rbCvanN"}, combine_IS=True, sign=1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/hyper_1000_points_rbCvanN"}, combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 1

class ColdstartRosenbrockHyperPes(ColdstartRosenbrockHyper):

    def __init__(self, bucket):
        super(ColdstartRosenbrockHyperPes, self).__init__("pes")
        # self._hist_data = construct_hist_data_from_pickle(dim=2, directory=self.data_dir, IS_filename_dict={0:"hyper_1000_points_rbCvanN", 1:"hyper_1000_points_rbCbiasN"}, combine_IS=True, sign=1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/hyper_1000_points_rbCvanN", 1:"coldstart/data/hyper_1000_points_rbCbiasN"}, combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class ColdstartRosenbrockBenchmark(object):
    """ Base class for all coldstart rosenbrock benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, obj_func_idx, replication_no, method_name, bucket):
        self.func_list = [RosenbrockVanilla(mult=1.0), RosenbrockSinus(mult=1.0), RosenbrockBiased(mult=1.0), RosenbrockSlightShifted(mult=1.0)]
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
        # with open("{0}/{1}_rb.pickle".format(self.hyper_dir, self.method_name), 'rb') as f:
        #     data = pickle.load(f)
        data = get_data_from_s3(self._bucket, "coldstart/hyper/{0}_rb".format(self.method_name))
        return data['hyperparam']

    @property
    def obj_func_min(self):
        return self.func_list[self.obj_func_idx]

    @property
    def result_path(self):
        # return "/fs/europa/g_pf/pickles/coldstart/result/{0}_{1}_repl_{2}.pickle".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)
        return "coldstart/result/{0}_{1}_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)

    @property
    def data_path(self):
        # return "/fs/europa/g_pf/pickles/coldstart/data/{0}_{1}_repl_{2}.pickle".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)
        return "coldstart/data/{0}_{1}_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)

    @property
    def num_iterations(self):
        return 25

    @property
    def truth_is(self):
        return 0

    @property
    def exploitation_is(self):
        return 0

    @property
    def list_sample_is(self):
        return [0]

class ColdstartRosenbrockBenchmarkEgo(ColdstartRosenbrockBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartRosenbrockBenchmarkEgo, self).__init__(obj_func_idx, replication_no, "ego", bucket)
        # self._hist_data = construct_hist_data_from_pickle(dim=2, directory=self.data_dir, IS_filename_dict={0:"{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=False, sign=1.0)[0]
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=False, sign=1.0)[0]

    @property
    def num_is_in(self):
        return None

class ColdstartRosenbrockBenchmarkKg(ColdstartRosenbrockBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartRosenbrockBenchmarkKg, self).__init__(obj_func_idx, replication_no, "kg", bucket)
        # self._hist_data = construct_hist_data_from_pickle(dim=2, directory=self.data_dir, IS_filename_dict={0:"{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=True, sign=-1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 0

class ColdstartRosenbrockBenchmarkMkg(ColdstartRosenbrockBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartRosenbrockBenchmarkMkg, self).__init__(obj_func_idx, replication_no, "mkg", bucket)
        prev_idx = 2 if self.obj_func_idx == 0 else (self.obj_func_idx - 1)
        if obj_func_idx == 3:   # for slsh, prev is van
            prev_idx = 0
        prev_data_filename = "coldstart/data/kg_{0}_repl_{1}".format(self.func_list[prev_idx].getFuncName(), self.replication_no)
        # self._hist_data = construct_hist_data_from_pickle(dim=2, directory=self.data_dir, IS_filename_dict={0:"{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no), 1:prev_data_filename}, combine_IS=True, sign=-1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no), 1:prev_data_filename}, combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1

class ColdstartRosenbrockBenchmarkSeqpes(ColdstartRosenbrockBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartRosenbrockBenchmarkSeqpes, self).__init__(obj_func_idx, replication_no, "seqpes", bucket)
        # self._hist_data = construct_hist_data_from_pickle(dim=2, directory=self.data_dir, IS_filename_dict={0:"{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=True, sign=1.0)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no)}, combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 1

class ColdstartRosenbrockBenchmarkPes(ColdstartRosenbrockBenchmark):

    def __init__(self, obj_func_idx, replication_no, bucket):
        super(ColdstartRosenbrockBenchmarkPes, self).__init__(obj_func_idx, replication_no, "pes", bucket)
        prev_idx = 2 if self.obj_func_idx == 0 else (self.obj_func_idx - 1)
        if obj_func_idx == 3:   # for slsh, prev is van
            prev_idx = 0
        prev_data_filename = "coldstart/data/seqpes_{0}_repl_{1}".format(self.func_list[prev_idx].getFuncName(), self.replication_no)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"coldstart/data/{0}_1_points_each_repl_{1}".format(self.obj_func_min.getFuncName(), self.replication_no), 1:prev_data_filename}, combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class_collection = {
    "coldstart_rb_hyper_ego": ColdstartRosenbrockHyperEgo,
    "coldstart_rb_hyper_kg": ColdstartRosenbrockHyperKg,
    "coldstart_rb_hyper_mkg": ColdstartRosenbrockHyperMkg,
    "coldstart_rb_hyper_seqpes": ColdstartRosenbrockHyperSeqpes,
    "coldstart_rb_hyper_pes": ColdstartRosenbrockHyperPes,
    "coldstart_rb_benchmark_ego": ColdstartRosenbrockBenchmarkEgo,
    "coldstart_rb_benchmark_kg": ColdstartRosenbrockBenchmarkKg,
    "coldstart_rb_benchmark_mkg": ColdstartRosenbrockBenchmarkMkg,
    "coldstart_rb_benchmark_seqpes": ColdstartRosenbrockBenchmarkSeqpes,
    "coldstart_rb_benchmark_pes": ColdstartRosenbrockBenchmarkPes,
}