#!/bin/sh
#SBATCH -p p
#SBATCH --exclusive
#SBATCH --gres=gpu:1

srun python vgg16_cifar10.py