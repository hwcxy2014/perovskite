from abc import ABCMeta, abstractproperty

from misoRosenbrock.rosenbrock import RosenbrockNew, RosenbrockRemi
from data_io import get_data_from_s3, send_data_to_s3, construct_hist_data_from_s3

__author__ = 'jialeiwang'

class MisoRosenbrockHyper(object):
    """ Base class for all miso rb training hyperparameters
    """
    __metaclass__ = ABCMeta

    def __init__(self, method_name, obj_func_idx):
        self.method_name = method_name
        self._hist_data = None
        self._obj_func = [RosenbrockRemi(mult=1.0), RosenbrockNew(mult=1.0)]
        self._obj_func_idx = obj_func_idx

    @property
    def hist_data(self):
        return self._hist_data

    @property
    def obj_func_min(self):
        return self._obj_func[self._obj_func_idx]

    @property
    def hyper_path(self):
        return "miso/hyper/{0}_{1}".format(self.method_name, self.obj_func_min.getFuncName())

class MisoRosenbrockHyperEgo(MisoRosenbrockHyper):

    def __init__(self, obj_func_idx, bucket):
        super(MisoRosenbrockHyperEgo, self).__init__("ego", obj_func_idx)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"miso/data/hyper_{0}_IS_0_1000_points".format(self.obj_func_min.getFuncName())}, combine_IS=False, sign=1.0)[0]

class MisoRosenbrockHyperMkg(MisoRosenbrockHyper):

    def __init__(self, obj_func_idx, bucket):
        super(MisoRosenbrockHyperMkg, self).__init__("mkg", obj_func_idx)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2,
                                                      IS_key_dict={0:"miso/data/hyper_{0}_IS_0_1000_points".format(self.obj_func_min.getFuncName()), 1:"miso/data/hyper_{0}_IS_1_1000_points".format(self.obj_func_min.getFuncName())},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisoRosenbrockHyperPes(MisoRosenbrockHyper):

    def __init__(self, obj_func_idx, bucket):
        super(MisoRosenbrockHyperPes, self).__init__("pes", obj_func_idx)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2,
                                                      IS_key_dict={0:"miso/data/hyper_{0}_IS_0_1000_points".format(self.obj_func_min.getFuncName()), 1:"miso/data/hyper_{0}_IS_1_1000_points".format(self.obj_func_min.getFuncName())},
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisoRosenbrockHyperMei(MisoRosenbrockHyper):

    def __init__(self, obj_func_idx, bucket):
        super(MisoRosenbrockHyperMei, self).__init__("mei", obj_func_idx)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2,
                                                      IS_key_dict={0:"miso/data/hyper_{0}_IS_0_1000_points".format(self.obj_func_min.getFuncName()), 1:"miso/data/hyper_{0}_IS_1_1000_points".format(self.obj_func_min.getFuncName())},
                                                      combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisoRosenbrockBenchmark(object):
    """ Base class for all miso atoext benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, obj_func_idx, bucket):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func = [RosenbrockRemi(mult=1.0), RosenbrockNew(mult=1.0)]
        self._obj_func_idx = obj_func_idx

    @abstractproperty
    def num_is_in(self):
        pass

    @property
    def hist_data(self):
        return self._hist_data

    @property
    def hyper_param(self):
        data = get_data_from_s3(self._bucket, "miso/hyper/{0}_{1}".format(self.method_name, self._obj_func[self._obj_func_idx].getFuncName()))
        return data['hyperparam']

    @property
    def obj_func_min(self):
        return self._obj_func[self._obj_func_idx]

    @property
    def result_path(self):
        return "miso/result/{0}_{1}_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)

    @property
    def data_path(self):
        return None

    @property
    def num_iterations(self):
        return 25

    @property
    def truth_is(self):
        return 0

    @property
    def exploitation_is(self):
        return 1

    @property
    def list_sample_is(self):
        return range(2)

class MisoRosenbrockBenchmarkEgo(MisoRosenbrockBenchmark):

    def __init__(self, replication_no, obj_func_idx, bucket):
        super(MisoRosenbrockBenchmarkEgo, self).__init__(replication_no, "ego", obj_func_idx, bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2, IS_key_dict={0:"miso/data/{0}_IS_0_5_points_repl_{1}".format(self.obj_func_min.getFuncName(), replication_no)}, combine_IS=False, sign=1.0)[0]

    @property
    def num_is_in(self):
        return None

    @property
    def list_sample_is(self):
        return [0]

class MisoRosenbrockBenchmarkMkg(MisoRosenbrockBenchmark):

    def __init__(self, replication_no, obj_func_idx, bucket):
        super(MisoRosenbrockBenchmarkMkg, self).__init__(replication_no, "mkg", obj_func_idx, bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2,
                                                      IS_key_dict={0:"miso/data/{0}_IS_0_5_points_repl_{1}".format(self.obj_func_min.getFuncName(), replication_no), 1:"miso/data/{0}_IS_1_5_points_repl_{1}".format(self.obj_func_min.getFuncName(), replication_no)},
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 2

class MisoRosenbrockBenchmarkPes(MisoRosenbrockBenchmark):

    def __init__(self, replication_no, obj_func_idx, bucket):
        super(MisoRosenbrockBenchmarkPes, self).__init__(replication_no, "pes", obj_func_idx, bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2,
                                                      IS_key_dict={0:"miso/data/{0}_IS_0_5_points_repl_{1}".format(self.obj_func_min.getFuncName(), replication_no), 1:"miso/data/{0}_IS_1_5_points_repl_{1}".format(self.obj_func_min.getFuncName(), replication_no)},
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2    # This should be the number of IS

    @property
    def result_path(self):
        return "miso/result/{0}_2_{1}_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)
    #FIXME this property prevents pes2 from overwriting the results of pes. Eventually, it should be removed for new benchmarks. It adds the _2_

class MisoRosenbrockBenchmarkMei(MisoRosenbrockBenchmark):

    def __init__(self, replication_no, obj_func_idx, bucket):
        super(MisoRosenbrockBenchmarkMei, self).__init__(replication_no, "mei", obj_func_idx, bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=2,
                                                      IS_key_dict={0:"miso/data/{0}_IS_0_5_points_repl_{1}".format(self.obj_func_min.getFuncName(), replication_no), 1:"miso/data/{0}_IS_1_5_points_repl_{1}".format(self.obj_func_min.getFuncName(), replication_no)},
                                                      combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 2    # This should be the number of IS

class_collection = {
    "miso_rb_hyper_ego": MisoRosenbrockHyperEgo,
    "miso_rb_hyper_mkg": MisoRosenbrockHyperMkg,
    "miso_rb_hyper_pes": MisoRosenbrockHyperPes,
    "miso_rb_hyper_mei": MisoRosenbrockHyperMei,
    "miso_rb_benchmark_ego": MisoRosenbrockBenchmarkEgo,
    "miso_rb_benchmark_mkg": MisoRosenbrockBenchmarkMkg,
    "miso_rb_benchmark_pes": MisoRosenbrockBenchmarkPes,
    "miso_rb_benchmark_mei": MisoRosenbrockBenchmarkMei,
}
