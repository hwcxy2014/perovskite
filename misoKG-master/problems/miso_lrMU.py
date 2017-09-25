from abc import ABCMeta, abstractproperty

from misoLogRegMnistUsps.logreg_mnistusps import LogRegMNISTUSPS
from data_io import get_data_from_s3, construct_hist_data_from_s3

__author__ = 'matthiaspoloczek'

class MisoLrMUHyper(object):
    """ Base class for all misoLRMU training hyperparameters
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
        return LogRegMNISTUSPS(mult=1.0)

    @property
    def hyper_path(self):
        return "miso/hyper/{0}_lrMU".format(self.method_name)

class MisoLrMUHyperEgo(MisoLrMUHyper):

    def __init__(self, bucket):
        super(MisoLrMUHyperEgo, self).__init__("ego")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU_IS_0_1000_points"},
                                                      combine_IS=False, sign=1.0)[0]

class MisoLrMUHyperMkg(MisoLrMUHyper):

    def __init__(self, bucket):
        super(MisoLrMUHyperMkg, self).__init__("mkg")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU_IS_1_1000_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisoLrMUHyperMkgCandPts(MisoLrMUHyper):

    def __init__(self, bucket):
        super(MisoLrMUHyperMkgCandPts, self).__init__("mkgcandpts")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU_IS_1_1000_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisoLrMUHyperPes(MisoLrMUHyper):

    def __init__(self, bucket):
        super(MisoLrMUHyperPes, self).__init__("pes")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU_IS_1_1000_points"},
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisoLrMUHyperMei(MisoLrMUHyper):

    def __init__(self, bucket):
        super(MisoLrMUHyperMei, self).__init__("mei")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU_IS_1_1000_points"
                                                                   }, combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisoLrMUBenchmark(object):
    """ Base class for all misolrMU benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, bucket):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func_min = LogRegMNISTUSPS(mult=1.0)

    @abstractproperty
    def num_is_in(self):
        pass

    @property
    def hist_data(self):
        return self._hist_data

    @property
    def hyper_param(self):
        data = get_data_from_s3(self._bucket, "miso/hyper/{0}_lrMU".format(self.method_name))
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

class MisoLrMUBenchmarkEgo(MisoLrMUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoLrMUBenchmarkEgo, self).__init__(replication_no, "ego", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU_IS_0_10_points_repl_{0}".format(replication_no)},
                                                      combine_IS=False, sign=1.0)[0]

    @property
    def num_is_in(self):
        return None

    @property
    def list_sample_is(self):
        #EGO should only query the truthIS since it is available in this benchmark
        return [self.truth_is]

class MisoLrMUBenchmarkMkg(MisoLrMUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoLrMUBenchmarkMkg, self).__init__(replication_no, "mkg", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU

    # TODO: this should delete if you don't want to see convergence behavior
    # @property
    # def num_iterations(self):
    #     return 150

class MisoLrMUBenchmarkMkgCandPts(MisoLrMUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoLrMUBenchmarkMkgCandPts, self).__init__(replication_no, "mkgcandpts", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU

    # TODO: this should delete if you don't want to see convergence behavior
    # @property
    # def num_iterations(self):
    #     return 150

class MisoLrMUBenchmarkPes(MisoLrMUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoLrMUBenchmarkPes, self).__init__(replication_no, "pes", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2    # This should be the number of IS

class MisoLrMUBenchmarkMei(MisoLrMUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoLrMUBenchmarkMei, self).__init__(replication_no, "mei", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 3    # This should be the number of IS

class_collection = {
    "miso_lrMU_hyper_ego": MisoLrMUHyperEgo,
    "miso_lrMU_hyper_mkg": MisoLrMUHyperMkg,
    "miso_lrMU_hyper_mkgcandpts": MisoLrMUHyperMkgCandPts,
    "miso_lrMU_hyper_pes": MisoLrMUHyperPes,
    "miso_lrMU_hyper_mei": MisoLrMUHyperMei,
    "miso_lrMU_benchmark_ego": MisoLrMUBenchmarkEgo,
    "miso_lrMU_benchmark_mkg": MisoLrMUBenchmarkMkg,
    "miso_lrMU_benchmark_mkgcandpts": MisoLrMUBenchmarkMkgCandPts,
    "miso_lrMU_benchmark_pes": MisoLrMUBenchmarkPes,
    "miso_lrMU_benchmark_mei": MisoLrMUBenchmarkMei,
}


