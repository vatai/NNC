#!/bin/bash
#SBATCH -p p
#SBATCH --gres=gpu:1

PYTHONPATH=. srun python run_exp.py with 'experiment_args.dataset_name="mnist"' 'fit_args.verbose=2'
 