from abc import ABCMeta, abstractproperty

from assembleToOrderExtended.assembleToOrderExtended import AssembleToOrderExtended
from data_io import construct_hist_data_from_pickle, match_data_filename, check_file_legitimate, get_data_from_s3, send_data_to_s3, construct_hist_data_from_s3

__author__ = 'jialeiwang'

class MisoAtoextHyper(object):
    """ Base class for all miso ATO training hyperparameters
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
        return AssembleToOrderExtended(mult=-1.0)

    @property
    def hyper_path(self):
        return "miso/hyper/{0}_atoext".format(self.method_name)

class MisoAtoextHyperEgo(MisoAtoextHyper):

    def __init__(self, bucket):
        super(MisoAtoextHyperEgo, self).__init__("ego")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"miso/data/hyper_atoext_IS_0_1000_points"}, combine_IS=False, sign=1.0)[0]

class MisoAtoextHyperMkg(MisoAtoextHyper):

    def __init__(self, bucket):
        super(MisoAtoextHyperMkg, self).__init__("mkg")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"miso/data/hyper_atoext_IS_0_1000_points", 1:"miso/data/hyper_atoext_IS_1_1000_points", 2:"miso/data/hyper_atoext_IS_2_1000_points"}, combine_IS=False, sign=-1.0, take_diff=True, primary_IS=0)

class MisoAtoextHyperPes(MisoAtoextHyper):

    def __init__(self, bucket):
        super(MisoAtoextHyperPes, self).__init__("pes")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"miso/data/hyper_atoext_IS_0_1000_points", 1:"miso/data/hyper_atoext_IS_1_1000_points", 2:"miso/data/hyper_atoext_IS_2_1000_points"}, combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 3

class MisoAtoextHyperMei(MisoAtoextHyper):

    def __init__(self, bucket):
        super(MisoAtoextHyperMei, self).__init__("mei")
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"miso/data/hyper_atoext_IS_0_1000_points", 1:"miso/data/hyper_atoext_IS_1_1000_points", 2:"miso/data/hyper_atoext_IS_2_1000_points"}, combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 3

class MisoAtoextBenchmark(object):
    """ Base class for all miso atoext benchmark problems.
    """
    __metaclass__ = ABCMeta

    def __init__(self, replication_no, method_name, bucket):
        self.replication_no = replication_no
        self.method_name = method_name
        self._hist_data = None
        self._bucket = bucket
        self._obj_func_min = AssembleToOrderExtended(mult=-1.0)

    @abstractproperty
    def num_is_in(self):
        pass

    @property
    def hist_data(self):
        return self._hist_data

    @property
    def hyper_param(self):
        data = get_data_from_s3(self._bucket, "miso/hyper/{0}_atoext".format(self.method_name))
        return data['hyperparam']

    @property
    def obj_func_min(self):
        return self._obj_func_min

    @property
    def result_path(self):
        return "miso/result/{0}_{1}_150steps_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)

    @property
    def data_path(self):
        return None

    @property
    def num_iterations(self):
        return 150

    @property
    def truth_is(self):
        return 0

    @property
    def exploitation_is(self):
        return 2

    @property
    def list_sample_is(self):
        return range(3)

class MisoAtoextBenchmarkEgo(MisoAtoextBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoAtoextBenchmarkEgo, self).__init__(replication_no, "ego", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8, IS_key_dict={0:"miso/data/atoext_IS_0_20_points_repl_{0}".format(replication_no)}, combine_IS=False, sign=1.0)[0]

    @property
    def num_is_in(self):
        return None

    @property
    def list_sample_is(self):
        return [0]

class MisoAtoextBenchmarkMkg(MisoAtoextBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoAtoextBenchmarkMkg, self).__init__(replication_no, "mkg", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8,
                                                      IS_key_dict={0:"miso/data/atoext_IS_0_20_points_repl_{0}".format(replication_no), 1:"miso/data/atoext_IS_1_20_points_repl_{0}".format(replication_no), 2:"miso/data/atoext_IS_2_20_points_repl_{0}".format(replication_no)},
                                                      combine_IS=True, sign=-1.0)

    @property
    def num_is_in(self):
        return 2    # This should be idx of the last IS, in this case, is 2

class MisoAtoextBenchmarkPes(MisoAtoextBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoAtoextBenchmarkPes, self).__init__(replication_no, "pes", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8,
                                                      IS_key_dict={0:"miso/data/atoext_IS_0_20_points_repl_{0}".format(replication_no), 1:"miso/data/atoext_IS_1_20_points_repl_{0}".format(replication_no), 2:"miso/data/atoext_IS_2_20_points_repl_{0}".format(replication_no)},
                                                      combine_IS=True, sign=1.0)

    @property
    def num_is_in(self):
        return 3    # This should be the number of IS

    @property
    def result_path(self):
        return "miso/result/{0}_2_{1}_150steps_repl_{2}".format(self.method_name, self.obj_func_min.getFuncName(), self.replication_no)
    #FIXME this property prevents pes2 from overwriting the results of pes. Eventually, it should be removed for new benchmarks. It adds the _2_

class MisoAtoextBenchmarkMei(MisoAtoextBenchmark):

    def __init__(self, replication_no, bucket):
        super(MisoAtoextBenchmarkMei, self).__init__(replication_no, "mei", bucket)
        self._hist_data = construct_hist_data_from_s3(bucket=bucket, dim=8,
                                                      IS_key_dict={0:"miso/data/atoext_IS_0_20_points_repl_{0}".format(replication_no), 1:"miso/data/atoext_IS_1_20_points_repl_{0}".format(replication_no), 2:"miso/data/atoext_IS_2_20_points_repl_{0}".format(replication_no)},
                                                      combine_IS=False, sign=1.0)

    @property
    def num_is_in(self):
        return 3    # This should be the number of IS

class_collection = {
    "miso_atoext_hyper_ego": MisoAtoextHyperEgo,
    "miso_atoext_hyper_mkg": MisoAtoextHyperMkg,
    "miso_atoext_hyper_pes": MisoAtoextHyperPes,
    "miso_atoext_hyper_mei": MisoAtoextHyperMei,
    "miso_atoext_benchmark_ego": MisoAtoextBenchmarkEgo,
    "miso_atoext_benchmark_mkg": MisoAtoextBenchmarkMkg,
    "miso_atoext_benchmark_pes": MisoAtoextBenchmarkPes,
    "miso_atoext_benchmark_mei": MisoAtoextBenchmarkMei,
}
