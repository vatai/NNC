#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:2

PYTHONPATH=.. srun --gres=gpu:1 python vgg16_cifar10.py &
PYTHONPATH=.. srun --gres=gpu:1 python vgg16_cifar10.py with 'fit_args.callbacks=[]' &
wait
