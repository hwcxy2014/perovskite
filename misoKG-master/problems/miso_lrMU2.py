from abc import ABCMeta, abstractproperty

from misoLogRegMnistUsps.logreg_mnistusps import LogRegMNISTUSPS2
from data_io import get_data_from_s3, construct_hist_data_from_s3

__author__ = 'matthiaspoloczek'

class MisolrMU2Hyper(object):
    """ Base class for all misolrMU2 training hyperparameters
    """
    __metaclass__ = ABCMeta

    def __init__(self, method_name):
        self.method_name = method_name
        self._hist_data = None

    @property
    def hist_data(self):
        return self._hist_data

    @property
    def obj_func_min(self):
        return LogRegMNISTUSPS2(mult=1.0)

    @property
    def hyper_path(self):
        return "miso/hyper/{0}_lrMU2".format(self.method_name)

class MisolrMU2HyperEgo(MisolrMU2Hyper):

    def __init__(self, bucket):
        super(MisolrMU2HyperEgo, self).__init__("ego")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points"},
                                                      combine_IS=False, sign=1.0)[0]

class MisolrMU2HyperMkg(MisolrMU2Hyper):

    def __init__(self, bucket):
        super(MisolrMU2HyperMkg, self).__init__("mkg")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisolrMU2HyperMkgCandPts(MisolrMU2Hyper):

    def __init__(self, bucket):
        super(MisolrMU2HyperMkgCandPts, self).__init__("mkgcandpts")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisolrMU2HyperPes(MisolrMU2Hyper):

    def __init__(self, bucket):
        super(MisolrMU2HyperPes, self).__init__("pes")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"},
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisolrMU2HyperMei(MisolrMU2Hyper):

    def __init__(self, bucket):
        super(MisolrMU2HyperMei, self).__init__("mei")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"
                                                                   }, combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisolrMU2Benchmark(object):
    """ Base class for all misolrMU2 benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, bucket):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func_min = LogRegMNISTUSPS2(mult=1.0)

    @abstractproperty
    def num_is_in(self):
        pass

    @property
    def hist_data(self):
        return self._hist_data

    @property
    def hyper_param(self):
        data = get_data_from_s3(self._bucket, "miso/hyper/{0}_lrMU2".format(self.method_name))
        return data['hyperparam']

    @property
    def obj_func_min(self):
        return self._obj_func_min

    @property
    def result_path(self):
        return "miso/result/{0}_{1}_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)

    @property
    def data_path(self):
        return None

    @property
    def num_iterations(self):
        return 50

    @property
    def truth_is(self):
        return 0

    @property
    def exploitation_is(self):
        return 1

    @property
    def list_sample_is(self):
        return range(2)

class MisolrMU2BenchmarkEgo(MisolrMU2Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU2BenchmarkEgo, self).__init__(replication_no, "ego", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no)},
                                                      combine_IS=False, sign=1.0)[0]

    @property
    def num_is_in(self):
        return None

    @property
    def list_sample_is(self):
        #EGO should only query the truthIS since it is available in this benchmark
        return [self.truth_is]

class MisolrMU2BenchmarkMkg(MisolrMU2Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU2BenchmarkMkg, self).__init__(replication_no, "mkg", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU2

    # TODO: this should delete if you don't want to see convergence behavior
    # @property
    # def num_iterations(self):
    #     return 150

class MisolrMU2BenchmarkMkgCandPts(MisolrMU2Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU2BenchmarkMkgCandPts, self).__init__(replication_no, "mkgcandpts", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU2

    # TODO: this should delete if you don't want to see convergence behavior
    # @property
    # def num_iterations(self):
    #     return 150

class MisolrMU2BenchmarkPes(MisolrMU2Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU2BenchmarkPes, self).__init__(replication_no, "pes", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2    # This should be the number of IS

class MisolrMU2BenchmarkMei(MisolrMU2Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU2BenchmarkMei, self).__init__(replication_no, "mei", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 3    # This should be the number of IS

class_collection = {
    "miso_lrMU2_hyper_ego": MisolrMU2HyperEgo,
    "miso_lrMU2_hyper_mkg": MisolrMU2HyperMkg,
    "miso_lrMU2_hyper_mkgcandpts": MisolrMU2HyperMkgCandPts,
    "miso_lrMU2_hyper_pes": MisolrMU2HyperPes,
    "miso_lrMU2_hyper_mei": MisolrMU2HyperMei,
    "miso_lrMU2_benchmark_ego": MisolrMU2BenchmarkEgo,
    "miso_lrMU2_benchmark_mkg": MisolrMU2BenchmarkMkg,
    "miso_lrMU2_benchmark_mkgcandpts": MisolrMU2BenchmarkMkgCandPts,
    "miso_lrMU2_benchmark_pes": MisolrMU2BenchmarkPes,
    "miso_lrMU2_benchmark_mei": MisolrMU2BenchmarkMei,
}


