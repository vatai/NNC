#!/bin/sh

PART=$1
shift
NUMGPU=$1
shift
CMD=$*
sed "s!num_gpus.*!num_gpus $NUMGPU!" -i $1

cat launcher.template \
  | sed s/PART/$PART/ \
  | sed s/NUMGPU/$NUMGPU/ \
  | sed s/CMD/"$CMD"/ \
  > launcher.sbatch

sbatch launcher.sbatch

