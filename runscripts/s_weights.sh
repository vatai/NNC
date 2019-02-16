#!/bin/sh

#SBATCH -p p
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

for eps in 4 5; do
  norm=$SLURM_ARRAY_TASK_ID
  srun sh -c "singularity exec --nv ./singularity/tf-11-mod \
    python src/get_dense_weight_size.py with \
      proc_args.norm=$norm \
      proc_args.epsilon=0.0$eps \
      proc_args.smoothing=0 \
    "
done
