#!/bin/sh
#SBATCH -p p
#SBATCH -N2
#SBATCH --exclusive
if [ -e /usr/local/anaconda3/lib ]; then
    srun=srun
fi

for exp in $(seq 19 19); do
    for norm in True; do
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
	# srun bash -c "singularity exec --nv ./singularity/stf-oldv.sif python ./src/compare_with_compression.py with proc_args.norm=False proc_args.epsilon=1e-11 'model_names=[\"resnet50\"]'"
	python3 ./src/compare_with_compression.py with proc_args.norm=$norm proc_args.epsilon=1e-$exp 'model_names=["resnet50"]' 'gen_args.fast_mode=True'
    done
done
