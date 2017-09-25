#!/bin/sh

#### RB Benchmark
#idx_begin=0
#idx_end=99
#benchmark_name="miso_rb_benchmark_pes"
#which_rb=1
#for i in $(seq $idx_begin $idx_end)
#do
#  echo ${benchmark_name}
#  echo ${which_rb}
#  jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/miso_rb_benchmark.nbs ${benchmark_name} ${which_rb} $i" -mfail -email charleswang304@gmail.com -nproc 4 8 -q medium -xhost sys_pf
##  jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/miso_rb_benchmark.nbs ${benchmark_name} ${which_rb} $i" -mfail -email charleswang304@gmail.com -nproc 1 4 -xhost sys_pf
#done

#### RB Hyper training
#benchmark_name="miso_rb_hyper_pes"
#which_rb=0
#jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/miso_rb_hyper.nbs ${benchmark_name} ${which_rb}" -mfail -email charleswang304@gmail.com
##-nproc 4 8 -q long

#### atoext Benchmark
idx_begin=0
idx_end=99
benchmark_name="miso_atoext_benchmark_mei"
for i in $(seq $idx_begin $idx_end)
do
  echo ${benchmark_name} $i
#  jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/miso_atoext_benchmark.nbs ${benchmark_name} $i" -mfail -email charleswang304@gmail.com -nproc 4 24 -q long -xhost sys_pf
  jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/miso_atoext_benchmark.nbs ${benchmark_name} $i" -mfail -email charleswang304@gmail.com -nproc 4 8 -q medium -xhost sys_pf
done

#### atoext Hyper training
#benchmark_name="miso_atoext_hyper_pes"
#jsub "/fs/home/jw865/remote_deployment/multifidelity/nbs/miso_atoext_hyper.nbs ${benchmark_name}" -mfail -email charleswang304@gmail.com
##-nproc 4 8 -q long
