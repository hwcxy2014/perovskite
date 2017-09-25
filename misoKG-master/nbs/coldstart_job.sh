#!/bin/sh

#### Benchmark
idx_begin=0
idx_end=99
benchmark_name="coldstart_rb_benchmark_mkg"
obj_idx=3
for i in $(seq $idx_begin $idx_end)
do
  echo ${benchmark_name}
  echo ${obj_idx}
  jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/coldstart_benchmark.nbs ${benchmark_name} $i ${obj_idx}" -mfail -email charleswang304@gmail.com -nproc 4 8 -q medium -xhost sys_pf
#  jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/coldstart_benchmark.nbs ${benchmark_name} $i ${obj_idx}" -mfail -email charleswang304@gmail.com
done

#### Hyper training
#benchmark_name="coldstart_ato_hyper_pes"
#jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/coldstart_hyper.nbs ${benchmark_name}" -mfail -email charleswang304@gmail.com
##-nproc 4 8 -q long

#### Gen benchmark data
#func_name="rb_slsh" # for func_name, refer to coldstart_gen_benchmark_data.py
#idx_begin=0
#idx_end=99
#for i in $(seq $idx_begin $idx_end)
#do
#  jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/coldstart_gen_benchmark_data.nbs ${func_name} $i" -mfail -email charleswang304@gmail.com
#done
