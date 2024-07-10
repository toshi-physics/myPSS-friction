#!/bin/bash

source ~/miniconda3/bin/activate pss-env

rhoseed_start=0.5
rhoseed_end=4.5
drhoseed=0.5
rgamma=2.0
alpha=0.1
chi=5

rhoseed=$(python3 -c "print('{:f}'.format($rhoseed_start))")
while (( $(bc <<< "$rhoseed <= $rhoseed_end") ))
    do
        echo "Creating files for rhoseed = $rhoseed"
        ./run_model_new_drho.s model_Q_v_rho_h_CH_both_noise $rgamma $alpha $chi $rhoseed
        rhoseed=$(python3 -c "print('{:f}'.format($rhoseed+$drhoseed))")
    done