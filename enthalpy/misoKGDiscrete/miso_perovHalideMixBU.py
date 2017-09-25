xfrom abc import ABCMeta, abstractproperty

from misoPerovskitesMixedHalides.misoPerovskites_mixedhalides import MisoPerovskites_mixedhalides
from data_io import get_data_from_s3, construct_hist_data_from_s3

__author__ = 'matthiaspoloczek'

class MisoperovHalideMixBUHyper(object):
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
        return MisoPerovskites_mixedhalides(mult=1.0)

    @property
    def hyper_path(self):
        return "miso/hyper/{0}_perovHalideMixBU".format(self.method_name)

class MisoperovHalideMixBUHyperEgo(MisoperovHalideMixBUHyper):

    def __init__(self, bucket):
        super(MisoperovHalideMixBUHyperEgo, self).__init__("ego")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/hyper_perovHalideMixBU_IS_0_10_points"},
                                                      combine_IS=False, sign=1.0)[0]

class MisoperovHalideMixBUHyperMkg(MisoperovHalideMixBUHyper):

    def __init__(self, bucket):
        super(MisoperovHalideMixBUHyperMkg, self).__init__("mkg")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/hyper_perovHalideMixBU_IS_0_10_points",
                                                                   1:"miso/data/hyper_perovHalideMixBU_IS_1_10_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisoperovHalideMixBUHyperMkgCandPts(MisoperovHalideMixBUHyper):

    def __init__(self, bucket):
        super(MisoperovHalideMixBUHyperMkgCandPts, self).__init__("mkgcandpts")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/hyper_perovHalideMixBU_IS_0_10_points",
                                                                   1:"miso/data/hyper_perovHalideMixBU_IS_1_10_points"},
                                                      combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisoperovHalideMixBUHyperPes(MisoperovHalideMixBUHyper):

    def __init__(self, bucket):
        super(MisoperovHalideMixBUHyperPes, self).__init__("pes")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/hyper_perovHalideMixBU_IS_0_10_points",
                                                                   1:"miso/data/hyper_perovHalideMixBU_IS_1_10_points"},
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisoperovHalideMixBUHyperMei(MisoperovHalideMixBUHyper):

    def __init__(self, bucket):
        super(MisoperovHalideMixBUHyperMei, self).__init__("mei")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/hyper_perovHalideMixBU_IS_0_10_points",
                                                                   1:"miso/data/hyper_perovHalideMixBU_IS_1_10_points"
                                                                   }, combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 2

class MisoperovHalideMixBUBenchmark(object):
    """ Base class for all misolrMU4150st benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, bucket):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func_min = MisoPerovskites_mixedhalides(mult=1.0)

    @abstractproperty
    def num_is_in(self):
        pass

    @property
    def hist_data(self):
        return self._hist_data

    @property
    def hyper_param(self):
        data = get_data_from_s3(self._bucket, "miso/hyper/{0}_perovHalideMixBU".format(self.method_name))
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
        return 100

    @property
    def truth_is(self):
        return 0

    @property
    def exploitation_is(self):
        return 1

    @property
    def list_sample_is(self):
        return range(2)

class MisoperovHalideMixBUBenchmarkEgo(MisoperovHalideMixBUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoperovHalideMixBUBenchmarkEgo, self).__init__(replication_no, "ego", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/perovHalideMixBU_IS_0_10_points_repl_{0}".format(replication_no)},
                                                      combine_IS=False, sign=1.0)[0]

    @property
    def num_is_in(self):
        return None

    @property
    def list_sample_is(self):
        #EGO should only query the truthIS since it is available in this benchmark
        return [self.truth_is]

class MisoperovHalideMixBUBenchmarkMkg(MisoperovHalideMixBUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoperovHalideMixBUBenchmarkMkg, self).__init__(replication_no, "mkg", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/perovHalideMixBU_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/perovHalideMixBU_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU4150st

    # TODO: this should delete if you don't want to see convergence behavior
    @property
    def num_iterations(self):
        return 100

class MisoperovHalideMixBUBenchmarkMkgCandPts(MisoperovHalideMixBUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoperovHalideMixBUBenchmarkMkgCandPts, self).__init__(replication_no, "mkgcandpts", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/perovHalideMixBU_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/perovHalideMixBU_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 1    # This should be idx of the last IS, in this case, is 1 for lrMU4150st

    # TODO: this should delete if you don't want to see convergence behavior
    @property
    def num_iterations(self):
        return 100

class MisoperovHalideMixBUBenchmarkPes(MisoperovHalideMixBUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoperovHalideMixBUBenchmarkPes, self).__init__(replication_no, "pes", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/perovHalideMixBU_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/perovHalideMixBU_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 2    # This should be the number of IS

    @property
    def num_iterations(self):
        return 100

    @property
    def result_path(self):
        return "miso/result/{0}2_{1}_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)
    #FIXME this property prevents pes2 from overwriting the results of pes. Eventually, it should be removed for new benchmarks. It adds the _2_

class MisoperovHalideMixBUBenchmarkMei(MisoperovHalideMixBUBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoperovHalideMixBUBenchmarkMei, self).__init__(replication_no, "mei", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=3,
                                                      IS_key_dict={0:"miso/data/perovHalideMixBU_IS_0_10_points_repl_{0}".format(replication_no),
                                                                   1:"miso/data/perovHalideMixBU_IS_1_10_points_repl_{0}".format(replication_no)
                                                                   },
                                                      combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 2    # This should be the number of IS

    @property
    def num_iterations(self):
        return 100

class_collection = {
    "miso_perovHalideMixBU_hyper_ego": MisoperovHalideMixBUHyperEgo,
    "miso_perovHalideMixBU_hyper_mkg": MisoperovHalideMixBUHyperMkg,
    "miso_perovHalideMixBU_hyper_mkgcandpts": MisoperovHalideMixBUHyperMkgCandPts,
    "miso_perovHalideMixBU_hyper_pes": MisoperovHalideMixBUHyperPes,
    "miso_perovHalideMixBU_hyper_mei": MisoperovHalideMixBUHyperMei,
    "miso_perovHalideMixBU_benchmark_ego": MisoperovHalideMixBUBenchmarkEgo,
    "miso_perovHalideMixBU_benchmark_mkg": MisoperovHalideMixBUBenchmarkMkg,
    "miso_perovHalideMixBU_benchmark_mkgcandpts": MisoperovHalideMixBUBenchmarkMkgCandPts,
    "miso_perovHalideMixBU_benchmark_pes": MisoperovHalideMixBUBenchmarkPes,
    "miso_perovHalideMixBU_benchmark_mei": MisoperovHalideMixBUBenchmarkMei,
}


