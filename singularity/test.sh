#!/bin/sh

# I used this to test the slurm scheduler, to schedule
# tasks on different nodes.

for i in $(seq 4); do
  srun -l sh -c "sleep 2 && hostname"
done
