#!/bin/sh
#SBATCH -p p
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=0-1


for exp in 4 5; do
  norm=$SLURM_ARRAY_TASK_ID
  srun sh -c "singularity exec --nv ./singularity/tf-11-mod \
    python ./src/compare_with_compression.py with \
      gen_args.batch_size=64 \
      eval_args.max_queue_size=14 \
      eval_args.workers=14 \
      eval_args.use_multiprocessing=True \
      proc_args.norm=$norm \
      proc_args.smoothing=0 \
      proc_args.epsilon=0.0$exp \
      "
done
