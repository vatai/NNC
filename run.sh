#!/usr/bin/bash

for exp in $(seq 3 6); do
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
        $optirun python compare_with_compression.py with proc_args.norm=$norm proc_args.epsilon=1e-0$exp json_name=$name-0$exp
    done
done
