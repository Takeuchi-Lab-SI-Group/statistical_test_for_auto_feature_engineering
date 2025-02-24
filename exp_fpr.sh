#!/bin/bash

for n in 100 150 200; do
    for noise_type in iid corr; do
        python experiment_fpr.py --n $n --noise_type $noise_type
    done
done
