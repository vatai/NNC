"""
This program generates shell scripts to execute multiple experiments.
"""

norms = [True, False]
exps = range(10, 12)

python_cmd = "python"
pyfile = "./src/compare_with_compression.py"

sbatch_str = """#!/bin/sh
#SBATCH -N2
#SBATCH --partition=v
srun sh -c 'singularity exec --nv ./singularity/stf-oldv.sif {} {} with {}'
"""

for norm in norms:
    for exp in exps:
        norm_str = "norm" if norm else "nonorm"
        val = 10**(-exp)
        params = "proc_args.norm={} proc_args.epsilon={}".format(norm, val)
        filename = "./runscripts/script_{}_1e-{}".format(norm_str, exp)
        with open(filename, 'w') as runfile:
            # cmd = "{} {} with {}".format(python_cmd, pyfile, params)
            cmd = sbatch_str.format(python_cmd, pyfile, params)
            runfile.write(cmd)
