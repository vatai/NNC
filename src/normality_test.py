#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from glob import glob
from os.path import join, basename, splitext
from pickle import dump
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro


base = "../weights/"
results_file = "normality_results.pickle"
alpha = 0.05
npy_names = glob(join(base, "*"))

def proc_file(npy_name):
    results = {}
    normal, notnormal = [], []
    layer = np.load(npy_name)
    layer_name = basename(npy_name)
    layer_name = splitext(layer_name)[0]
    print(npy_name)

    stats, pvals, var_results, row_names = [], [], [], []
    for idx, row in enumerate(layer):
        row_name = "{}-{}".format(layer_name, idx)

        stat, pval = shapiro(row)
        var_result = np.var(row)

        p = qqplot(row, line="s")
        fname = "{}.pdf".format(row_name)
        plt.savefig(fname)
        plt.savefig(fname.replace('pdf', 'png'))
        plt.close(p)

        stats.append(stat)
        pvals.append(pval)
        var_results.append(var_result)
        row_names.append(row_name)

    print("DONE {}".format(layer_name))
    return stats, pvals, var_results, row_names



pool = Pool()
result = pool.map(proc_file, npy_names)
dump(result, open(results_file, 'wb'))
