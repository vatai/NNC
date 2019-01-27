#!/usr/bin/bash
#SBATCH
if [ -e /usr/local/anaconda3/lib ]; then
    srun=srun
fi

for exp in $(seq 3 3); do
    for norm in True False; do
        if [[ $norm == True ]]; then
            name=norm
        else
            name=nonorm
        fi
        if [ -f /usr/bin/optirun ]; then
            optirun=optirun
        else
            optirun=""
        fi
        $srun $optirun python3 compare_with_compression.py with gen_args.fast_mode=True 'model_names=["resnet50", "vgg16"]' proc_args.norm=$norm proc_args.epsilon=1e-0$exp json_name=$name-0$exp
    done
done
