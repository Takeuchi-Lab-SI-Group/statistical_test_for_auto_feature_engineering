#!/bin/bash

file="experiment_fpr.py"


for n in 100 150 200;do
    for d in 4;do
        for noise_type in iid corr;do
            for option in naive DS oc parametric;do
                python experiment/experiment_fpr.py \
                    --n $n \
                    --d $d \
                    --noise_type $noise_type \
                    --option $option
            done
        done
    done
done
