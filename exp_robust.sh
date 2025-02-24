#!/bin/bash

for n in 100 150 200; do
    python experiment_robust_estimated.py --n $n
done

for robust_type in exponential gennormflat t;do
    for distance in 0.01 0.02 0.03 0.04;do
        python experiment_robust_non_gaussian.py --robust_type $robust_type --distance $distance
    done
done
