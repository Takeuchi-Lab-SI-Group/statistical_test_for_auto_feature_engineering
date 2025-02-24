#!/bin/bash


for signal in 0.2 0.4 0.6 0.8; do
    for noise_type in iid corr; do
        python experiment_tpr.py --signal $signal --noise_type $noise_type
    done
done
