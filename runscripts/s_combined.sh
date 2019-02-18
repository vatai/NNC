#!/bin/sh
#SBATCH -p p
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --time=2:00:00

norm=$SLURM_ARRAY_TASK_ID
eps=$1
dsmooth=$2
csmooth=$3
srun sh -c "singularity exec --nv ./singularity/tf-11-mod \
  python ./src/combined.py with \
    gen_args.batch_size=64 \
    eval_args.max_queue_size=14 \
    eval_args.workers=14 \
    eval_args.use_multiprocessing=True \
    proc_args.norm=$norm \
    proc_args.dense_smooth=$dsmooth \
    proc_args.conv_smooth=$csmooth \
    proc_args.epsilon=$eps \
    "
