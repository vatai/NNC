#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from glob import glob
from os.path import join, basename, splitext, exists
from pickle import dump
from random import sample, seed
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro


seed(42)
sample_size = 10
base = "../weights/"
results_file = "normality_results.pickle"
npy_names = glob(join(base, "*"))
npy_dense = glob(join(base, "*_dense*"))


def proc_file(npy_name, do_plot=False):
    """
    Opens a numpy array from the base directory, and outputs a picle
    file with the results from the Shapiro test, variance, ?and the
    row name?.  If do_plot is true also generates a qqplot in bot pdf
    and png.
    """
    print(npy_name, do_plot)
    pickle_name = basename(npy_name)
    pickle_name = pickle_name.replace('npy', 'pickle')
    layer_name = splitext(pickle_name)[0]

    layer = np.load(npy_name)
    stats, pvals, var_results, row_names = [], [], [], []
    if do_plot:
        layer = sample(list(layer), sample_size)
    for idx, row in enumerate(layer):
        row_name = "{}-{}".format(layer_name, idx)

        stat, pval = shapiro(row)
        var_result = np.var(row)

        if do_plot:
            p = qqplot(row, line="s")
            fname = "qqplots/qq_{}.pdf".format(row_name)
            plt.savefig(fname)
            plt.savefig(fname.replace('pdf', 'png'))
            plt.close(p)
            print('FIG: {}'.format(fname))
            p = plt.hist(row)
            fname = "hists/hist_{}.pdf".format(row_name)
            plt.savefig(fname)
            plt.savefig(fname.replace('pdf', 'png'))
            plt.close()

        stats.append(stat)
        pvals.append(pval)
        var_results.append(var_result)
        row_names.append(row_name)

    print("DONE {}".format(layer_name))
    result = [stats, pvals, var_results]
    dump(result, open(pickle_name, 'wb'))



pool = Pool()
#pool.map(proc_file, npy_names)

print("----- ")
params = map(lambda t: (t, True), sample(npy_names, sample_size))
pool.starmap(proc_file, params)
params = [(t, True) for t in npy_dense]
pool.starmap(proc_file, params)
