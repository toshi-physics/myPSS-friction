#!/bin/bash

source ~/miniconda3/bin/activate pss-env

run_start=2
run_end=10
drun=1

p0=1.0
alpha=1.0
D=1
chi=5
rhoseed=0.5

run=$(python3 -c "print('{:d}'.format($run_start))")
while (( $(bc <<< "$run <= $run_end") ))
    do
        echo "Creating files for run = $run"
        ./run_model_runs.s model_Q_v_rho_heaviside $p0 $alpha $D $chi $rhoseed $run
        run=$(python3 -c "print('{:d}'.format($run+$drun))")
    done