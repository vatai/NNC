#!/bin/sh
#SBATCH -p p
#SBATCH --exclusive
#SBATCH --gres=gpu:1


for exp in 15 25 35; do
    for norm in True False; do
        if [ $norm = True ]; then
            name=norm
        else
            name=nonorm
        fi
        srun sh -c "singularity exec --nv ./singularity/tf-11-mod \
		python ./src/compare_with_compression.py with \
			gen_args.batch_size=64 \
			eval_args.max_queue_size=14 \
			eval_args.workers=14 \
			eval_args.use_multiprocessing=True \
			proc_args.norm=$norm \
			proc_args.epsilon=0.0$exp \
			json_name=$name-0.0$exp"
    done
done
