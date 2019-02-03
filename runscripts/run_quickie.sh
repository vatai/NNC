#!/bin/sh
#SBATCH -p p
#SBATCH -N2
#SBATCH --exclusive
if [ -e /usr/local/anaconda3/lib ]; then
    srun=srun
fi

for exp in $(seq 11 19); do
    for norm in True False; do
        if [ "$norm" = True ]; then
            name=norm
        else
            name=nonorm
        fi
        if [ -f /usr/bin/optirun ]; then
            optirun=optirun
        else
            optirun=""
        fi
	srun bash -c "singularity exec --nv ./singularity/stf-oldv.sif python ./src/compare_with_compression.py with proc_args.norm=False proc_args.epsilon=1e-11 'model_names=[\"resnet50\"]'"
    done
done
