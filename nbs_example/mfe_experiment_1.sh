#!/bin/sh

for mu in 1.0 2.0 3.0 4.0 5.0 6.0 8.0 10.0 12.0 15.0
do
dir='/fs/home/py75/projects/crowdsourcing-models/mfe-computation/experiments/experiment_4/mu_'
dir=$dir$mu
for C in 0.05 0.15 0.25 0.5 0.75 1.0 
do 
for n0 in 0.0 2.0 5.0 10.0 15.0 20.0 30.0
do
for n1 in 0.0 2.0 5.0 8.0 12.0 18.0 24.0 30.0 40.0 60.0
do
    # submit a job with given parameters to the NBS system
    jsub "mfe_experiment_1.nbs $n0 $n1 $C $dir $mu"
done
done
done 
done
