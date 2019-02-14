#!/bin/sh

#SBATCH -p p
#SBATCH --gres=gpu:1

# python3 -c 'from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())'

srun sh -c "singularity exec --nv ./singularity/tf-11-mod python ./src/nnclib/test_gpu.py"
