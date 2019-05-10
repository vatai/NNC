#!/bin/bash
#SBATCH -p p
#SBATCH --gres=gpu:1

python run_exp.py