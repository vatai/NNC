#!/usr/bin/bash
#SBATCH
for exp in $(seq 11 19); do
    for norm in True False; do
        python3 src/get_dense_weight_size.py with proc_args.norm=$norm proc_args.epsilon=1e-0$exp
    done
done
