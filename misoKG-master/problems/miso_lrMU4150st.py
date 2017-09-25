from abc import ABCMeta, abstractproperty

from misoLogRegMnistUsps.logreg_mnistusps import LogRegMNISTUSPS4
from data_io import get_data_from_s3, construct_hist_data_from_s3

__author__ = 'matthiaspoloczek'

class MisolrMU4150stHyper(object):
    """ Base class for all misolrMU4150st training hyperparameters
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
        return LogRegMNISTUSPS4(mult=1.0)

    @property
    def hyper_path(self):
        return "miso/hyper/{0}_lrMU2".format(self.method_name)

class MisolrMU4150stHyperEgo(MisolrMU4150stHyper):

    def __init__(self, bucket):
        super(MisolrMU4150stHyperEgo, self).__init__("ego")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points"},
                                                      combine_IS=False, sign=1.0)[0]

class MisolrMU4150stHyperMkg(MisolrMU4150stHyper):

    def __init__(self, bucket):
        super(MisolrMU4150stHyperMkg, self).__init__("mkg")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisolrMU4150stHyperMkgCandPts(MisolrMU4150stHyper):

    def __init__(self, bucket):
        super(MisolrMU4150stHyperMkgCandPts, self).__init__("mkgcandpts")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisolrMU4150stHyperPes(MisolrMU4150stHyper):

    def __init__(self, bucket):
        super(MisolrMU4150stHyperPes, self).__init__("pes")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"},
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisolrMU4150stHyperMei(MisolrMU4150stHyper):

    def __init__(self, bucket):
        super(MisolrMU4150stHyperMei, self).__init__("mei")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/hyper_lrMU2_IS_0_1000_points",
                                                                   1:"miso/data/hyper_lrMU2_IS_1_1000_points"
                                                                   }, combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisolrMU4150stBenchmark(object):
    """ Base class for all misolrMU4150st benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, bucket):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func_min = LogRegMNISTUSPS4(mult=1.0)

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
        return "miso/result/{0}_{1}150st_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)

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

class MisolrMU4150stBenchmarkEgo(MisolrMU4150stBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU4150stBenchmarkEgo, self).__init__(replication_no, "ego", bucket)
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

class MisolrMU4150stBenchmarkMkg(MisolrMU4150stBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU4150stBenchmarkMkg, self).__init__(replication_no, "mkg", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU4150st

    # TODO: this should delete if you don't want to see convergence behavior
    @property
    def num_iterations(self):
        return 150

class MisolrMU4150stBenchmarkMkgCandPts(MisolrMU4150stBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU4150stBenchmarkMkgCandPts, self).__init__(replication_no, "mkgcandpts", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU4150st

    # TODO: this should delete if you don't want to see convergence behavior
    @property
    def num_iterations(self):
        return 150

class MisolrMU4150stBenchmarkPes(MisolrMU4150stBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU4150stBenchmarkPes, self).__init__(replication_no, "pes", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2    # This should be the number of IS

    @property
    def num_iterations(self):
        return 150

    @property
    def result_path(self):
        return "miso/result/{0}2_{1}_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)
    #FIXME this property prevents pes2 from overwriting the results of pes. Eventually, it should be removed for new benchmarks. It adds the _2_

class MisolrMU4150stBenchmarkMei(MisolrMU4150stBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisolrMU4150stBenchmarkMei, self).__init__(replication_no, "mei", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=4,
                                                      IS_key_dict={0:"miso/data/lrMU2_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/lrMU2_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 3    # This should be the number of IS

    @property
    def num_iterations(self):
        return 150

class_collection = {
    "miso_lrMU4150st_hyper_ego": MisolrMU4150stHyperEgo,
    "miso_lrMU4150st_hyper_mkg": MisolrMU4150stHyperMkg,
    "miso_lrMU4150st_hyper_mkgcandpts": MisolrMU4150stHyperMkgCandPts,
    "miso_lrMU4150st_hyper_pes": MisolrMU4150stHyperPes,
    "miso_lrMU4150st_hyper_mei": MisolrMU4150stHyperMei,
    "miso_lrMU4150st_benchmark_ego": MisolrMU4150stBenchmarkEgo,
    "miso_lrMU4150st_benchmark_mkg": MisolrMU4150stBenchmarkMkg,
    "miso_lrMU4150st_benchmark_mkgcandpts": MisolrMU4150stBenchmarkMkgCandPts,
    "miso_lrMU4150st_benchmark_pes": MisolrMU4150stBenchmarkPes,
    "miso_lrMU4150st_benchmark_mei": MisolrMU4150stBenchmarkMei,
}


