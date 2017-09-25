from abc import ABCMeta, abstractproperty

from misoLogRegMnistUsps.logreg_mnistusps import LogRegMNISTUSPS3
from data_io import get_data_from_s3, construct_hist_data_from_s3

__author__ = 'matthiaspoloczek'

class MisolrMU3Hyper(object):
    """ Base class for all misolrMU3 training hyperparameters
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
        return LogRegMNISTUSPS3(mult=1.0)

    @property
    def hyper_path(self):
        return "miso/hyper/{0}_lrMU2".format(self.method_name)

class MisolrMU3HyperEgo(MisolrMU3Hyper):

    def __init__(self, bucket):
        super(MisolrMU3HyperEgo, self).__init__("ego")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points"},
                                                      combine_IS=False, sign=1.0)[0]

class MisolrMU3HyperMkg(MisolrMU3Hyper):

    def __init__(self, bucket):
        super(MisolrMU3HyperMkg, self).__init__("mkg")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisolrMU3HyperMkgCandPts(MisolrMU3Hyper):

    def __init__(self, bucket):
        super(MisolrMU3HyperMkgCandPts, self).__init__("mkgcandpts")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisolrMU3HyperPes(MisolrMU3Hyper):

    def __init__(self, bucket):
        super(MisolrMU3HyperPes, self).__init__("pes")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"},
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisolrMU3HyperMei(MisolrMU3Hyper):

    def __init__(self, bucket):
        super(MisolrMU3HyperMei, self).__init__("mei")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"
                                                                   }, combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisolrMU3Benchmark(object):
    """ Base class for all misolrMU3 benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, bucket):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func_min = LogRegMNISTUSPS3(mult=1.0)

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

class MisolrMU3BenchmarkEgo(MisolrMU3Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU3BenchmarkEgo, self).__init__(replication_no, "ego", bucket)
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

class MisolrMU3BenchmarkMkg(MisolrMU3Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU3BenchmarkMkg, self).__init__(replication_no, "mkg", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU3

    # TODO: this should delete if you don't want to see convergence behavior
    # @property
    # def num_iterations(self):
    #     return 150

class MisolrMU3BenchmarkMkgCandPts(MisolrMU3Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU3BenchmarkMkgCandPts, self).__init__(replication_no, "mkgcandpts", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU3

    # TODO: this should delete if you don't want to see convergence behavior
    # @property
    # def num_iterations(self):
    #     return 150

class MisolrMU3BenchmarkPes(MisolrMU3Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU3BenchmarkPes, self).__init__(replication_no, "pes", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2    # This should be the number of IS

class MisolrMU3BenchmarkMei(MisolrMU3Benchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU3BenchmarkMei, self).__init__(replication_no, "mei", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 3    # This should be the number of IS

class_collection = {
    "miso_lrMU3_hyper_ego": MisolrMU3HyperEgo,
    "miso_lrMU3_hyper_mkg": MisolrMU3HyperMkg,
    "miso_lrMU3_hyper_mkgcandpts": MisolrMU3HyperMkgCandPts,
    "miso_lrMU3_hyper_pes": MisolrMU3HyperPes,
    "miso_lrMU3_hyper_mei": MisolrMU3HyperMei,
    "miso_lrMU3_benchmark_ego": MisolrMU3BenchmarkEgo,
    "miso_lrMU3_benchmark_mkg": MisolrMU3BenchmarkMkg,
    "miso_lrMU3_benchmark_mkgcandpts": MisolrMU3BenchmarkMkgCandPts,
    "miso_lrMU3_benchmark_pes": MisolrMU3BenchmarkPes,
    "miso_lrMU3_benchmark_mei": MisolrMU3BenchmarkMei,
}


