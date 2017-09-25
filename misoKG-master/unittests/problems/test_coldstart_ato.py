from coldStartAssembleToOrder.assembleToOrder_vanilla import AssembleToOrderVanilla
from coldStartAssembleToOrder.assembleToOrder_var2 import AssembleToOrderVar2
from coldStartAssembleToOrder.assembleToOrder_var3 import AssembleToOrderVar3
from coldStartAssembleToOrder.assembleToOrder_var4 import AssembleToOrderVar4
from problems.coldstart_ato import *

__author__ = 'jialeiwang'

def tests_hyper():
    for method_name in ["ego", "kg", "mkg", "seqpes", "pes"]:
        yield check_hyper, method_name

def check_hyper(method_name):
    obj_dict = {
        "ego": ColdstartAtoHyperEgo(),
        "kg": ColdstartAtoHyperKg(),
        "mkg": ColdstartAtoHyperMkg(),
        "seqpes": ColdstartAtoHyperSeqpes(),
        "pes": ColdstartAtoHyperPes(),
    }
    hyper = obj_dict[method_name]
    assert hyper.method_name == method_name
    assert hyper.obj_func_min.getFuncName() == "atoC_vanilla"
    assert hyper.hyper_path == "/fs/europa/g_pf/pickles/coldstart/hyper/{0}_ato.pickle".format(method_name)
    if method_name == "mkg":
        assert hyper.hist_data[0].points_sampled.shape == (1000,8)
        assert hyper.hist_data[1].points_sampled.shape == (1000,8)
        assert hyper.hist_data[2].points_sampled.shape == (1000,8)
    elif method_name == "pes":
        assert hyper.hist_data.points_sampled.shape == (3000,9)
    elif method_name == "seqpes":
        assert hyper.hist_data.points_sampled.shape == (1000,9)
    else:
        assert hyper.hist_data.points_sampled.shape == (1000,8)

def tests_benchmark():
    for method_name in ["ego", "kg", "mkg", "seqpes", "pes"]:
        for func_idx in range(4):
            for repl_no in [0, 15, 99]:
                yield check_benchmark, func_idx, repl_no, method_name

def check_benchmark(obj_func_idx, repl_no, method_name):
    func_list = [AssembleToOrderVanilla(mult=-1.0), AssembleToOrderVar2(mult=-1.0), AssembleToOrderVar3(mult=-1.0), AssembleToOrderVar4(mult=-1.0)]
    obj_dict = {
        "ego": ColdstartAtoBenchmarkEgo(obj_func_idx, repl_no),
        "kg": ColdstartAtoBenchmarkKg(obj_func_idx, repl_no),
        "mkg": ColdstartAtoBenchmarkMkg(obj_func_idx, repl_no),
        "seqpes": ColdstartAtoBenchmarkSeqpes(obj_func_idx, repl_no),
        "pes": ColdstartAtoBenchmarkPes(obj_func_idx, repl_no),
    }
    benchmark = obj_dict[method_name]
    print benchmark.hyper_param
    if method_name == "ego":
        assert benchmark.hist_data.points_sampled.shape == (1,8)
    elif method_name == "kg" or method_name == "seqpes":
        assert benchmark.hist_data.points_sampled.shape == (1,9)
    else:
        assert benchmark.hist_data.points_sampled.shape == (101,9)
    assert benchmark.obj_func_min.getFuncName() == func_list[obj_func_idx].getFuncName()
    assert benchmark.result_path == "/fs/europa/g_pf/pickles/coldstart/result/{0}_{1}_repl_{2}.pickle".format(method_name, func_list[obj_func_idx].getFuncName(), repl_no)
    assert benchmark.data_path == "/fs/europa/g_pf/pickles/coldstart/data/{0}_{1}_repl_{2}.pickle".format(method_name, func_list[obj_func_idx].getFuncName(), repl_no)
    assert benchmark.num_iterations == 50
    assert benchmark.truth_is == 0
    assert benchmark.exploitation_is == 0
    assert benchmark.list_sample_is == [0]
    if method_name == "kg":
        assert benchmark.num_is_in == 0
    if method_name == "mkg":
        assert benchmark.num_is_in == 2
    if method_name == "pes":
        assert benchmark.num_is_in == 3
    if method_name == "seqpes":
        assert benchmark.num_is_in == 1
