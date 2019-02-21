#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from glob import glob
from os.path import join, basename, splitext, exists
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
    print(npy_name)
    pickle_name = basename(npy_name)
    pickle_name = pickle_name.replace('npy', 'pickle')
    layer_name = splitext(pickle_name)[0]
    if exists(pickle_name):
        print("SKIP {}".format(layer_name))
        return None

    layer = np.load(npy_name)
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
    result = [stats, pvals, var_results, row_names]
    dump(result, open(pickle_name, 'wb'))



pool = Pool(6)
pool.map(proc_file, npy_names)
