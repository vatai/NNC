"""
This program generates shell scripts to execute multiple experiments.
"""

norms = [True, False]
exps = range(10, 12)

prefix = "optirun"
prefix = "srun singularity exec --nv ./singularity/stf-oldv.sif bash"
sbatch_str = """#SBATCH -N2 
#SBATCH --partition=v
"""
pyfile = "./src/compare_with_compression.py"

for norm in norms:
    for exp in exps:
        norm_str = "norm" if norm else "nonorm"
        val = 10**(-exp)
        params = "proc_args.norm={} proc_args.epsilon={}".format(norm, val)
        filename = "./runscripts/script_{}_1e-{}".format(norm_str, exp)
        with open(filename, 'w') as runfile:
            runfile.write(sbatch_str)
            cmd = "{} python {} with {}".format(prefix, pyfile, params)
            runfile.write(cmd)
