#!/bin/sh

#SBATCH -p p
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

for eps in 9; do
  if [ $SLURM_ARRAY_TASK_ID = 0 ]; then
    norm=True
  else
    norm=False
  fi
  srun sh -c "singularity exec --nv ./singularity/tf-11-mod \
    python src/get_dense_weight_size.py with \
      proc_args.norm=$norm \
      proc_args.epsilon=0.0$eps"
done
