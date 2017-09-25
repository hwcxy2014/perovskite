#!/bin/bash

#echo "Activating virt env"
#v mupenv
#TODO does not work, start env manually

#echo "Running find_hyperparams_for_multifidelity_kg.py"
#python find_hyperparams_for_multifidelity_kg.py

echo "Waiting 9h"
# the hyper opt runs 11h in total, 2 are over already
sleep 9h

echo "Running run_multiKG_newLS.py"
python run_multiKG_newLS2.py