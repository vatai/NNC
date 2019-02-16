#!/bin/sh
#SBATCH -p p
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --time=1:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vatai.emil@mail.u-tokyo.ac.jp

eps=$1
smooth=$2
norm=$SLURM_ARRAY_TASK_ID
srun sh -c "singularity exec --nv ./singularity/tf-11-mod \
  python ./src/combined.py with \
    gen_args.batch_size=64 \
    eval_args.max_queue_size=14 \
    eval_args.workers=14 \
    eval_args.use_multiprocessing=True \
    proc_args.norm=$norm \
    proc_args.smoothing=$smooth \
    proc_args.epsilon=$eps \
    "
