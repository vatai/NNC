#!/usr/bin/bash
#SBATCH
if [ -e /usr/local/anaconda3/lib ]; then
    export PATH=/usr/local/cuda/bin:/usr/local/anaconda3/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/anaconda3/lib:$LD_LIBRARY_PATH
    srun=srun
fi

for exp in 9 8; do
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
        $srun $optirun python3 ./src/compare_with_compression.py with "proc_args.norm=$norm" "proc_args.epsilon=0.0$exp" "json_name=$name-0.0$exp"
    done
done
