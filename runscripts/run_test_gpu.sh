#!/bin/sh

#SBATCH -p p
#SBATCH -N2
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

srun python3 -c 'from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())'

srun sh -c "singularity exec --nv ./singularity/stf-oldv.sif python3 ./src/nnclib/test_gpu.py"
