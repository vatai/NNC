#!/bin/bash
#SBATCH -p p
#SBATCH --gres=gpu:1

PYTHONPATH=. srun python run_exp.py with 'experiment_args.dataset_name="cifar10"' 'fit_args.verbose=2'
 
