from problems.coldstart_rosenbrock import *

__author__ = 'jialeiwang'

def tests_hyper():
    for method_name in ["ego", "kg", "mkg", "seqpes", "pes"]:
        yield check_hyper, method_name

def check_hyper(method_name):
    obj_dict = {
        "ego": ColdstartRosenbrockHyperEgo(),
        "kg": ColdstartRosenbrockHyperKg(),
        "mkg": ColdstartRosenbrockHyperMkg(),
        "seqpes": ColdstartRosenbrockHyperSeqpes(),
        "pes": ColdstartRosenbrockHyperPes(),
    }
    hyper = obj_dict[method_name]
    assert hyper.method_name == method_name
    assert hyper.obj_func_min.getFuncName() == "rbCvanN"
    assert hyper.hyper_path == "/fs/europa/g_pf/pickles/coldstart/hyper/{0}_rb.pickle".format(method_name)
    if method_name == "mkg":
        assert hyper.hist_data[0].points_sampled.shape == (1000,2)
        assert hyper.hist_data[1].points_sampled.shape == (1000,2)
    elif method_name == "pes":
        assert hyper.hist_data.points_sampled.shape == (2000,3)
    elif method_name == "seqpes":
        assert hyper.hist_data.points_sampled.shape == (1000,3)
    else:
        assert hyper.hist_data.points_sampled.shape == (1000,2)

def tests_benchmark():
    for method_name in ["ego", "kg", "mkg", "seqpes", "pes"]:
        for func_idx in range(3):
            for repl_no in [0, 15, 99]:
                yield check_benchmark, func_idx, repl_no, method_name

def check_benchmark(obj_func_idx, repl_no, method_name):
    func_list = [RosenbrockVanilla(mult=1.0), RosenbrockSinus(mult=1.0), RosenbrockBiased(mult=1.0)]
    obj_dict = {
        "ego": ColdstartRosenbrockBenchmarkEgo(obj_func_idx, repl_no),
        "kg": ColdstartRosenbrockBenchmarkKg(obj_func_idx, repl_no),
        "mkg": ColdstartRosenbrockBenchmarkMkg(obj_func_idx, repl_no),
        "seqpes": ColdstartRosenbrockBenchmarkSeqpes(obj_func_idx, repl_no),
        "pes": ColdstartRosenbrockBenchmarkPes(obj_func_idx, repl_no),
    }
    benchmark = obj_dict[method_name]
    print benchmark.hyper_param
    if method_name == "ego":
        assert benchmark.hist_data.points_sampled.shape == (1,2)
    elif method_name == "kg" or method_name == "seqpes":
        assert benchmark.hist_data.points_sampled.shape == (1,3)
    else:
        assert benchmark.hist_data.points_sampled.shape == (26,3)
    assert benchmark.obj_func_min.getFuncName() == func_list[obj_func_idx].getFuncName()
    assert benchmark.result_path == "/fs/europa/g_pf/pickles/coldstart/result/{0}_{1}_repl_{2}.pickle".format(method_name, func_list[obj_func_idx].getFuncName(), repl_no)
    assert benchmark.data_path == "/fs/europa/g_pf/pickles/coldstart/data/{0}_{1}_repl_{2}.pickle".format(method_name, func_list[obj_func_idx].getFuncName(), repl_no)
    assert benchmark.num_iterations == 25
    assert benchmark.truth_is == 0
    assert benchmark.exploitation_is == 0
    assert benchmark.list_sample_is == [0]
    if method_name == "kg":
        assert benchmark.num_is_in == 0
    if method_name == "mkg":
        assert benchmark.num_is_in == 1
    if method_name == "pes":
        assert benchmark.num_is_in == 2
    if method_name == "seqpes":
        assert benchmark.num_is_in == 1
