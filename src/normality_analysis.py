"""
From the output of the normality test, generates plots and tables.
"""


from glob import glob
from random import sample, seed
from pickle import load
from pprint import pprint
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

SRC = "*.pickle"
seed(42)


def make_figs(pickle_files):
    d = {}
    for file in pickle_files:
        pvals = load(open(file, 'rb'))[1]
        model = file.split("_")[0]
        if model in d:
            d[model].append(pvals)
        else:
            d[model] = [pvals]
    for model, pvals in d.items():
        p = None
        for vals in pvals:
            if p is None:
                p = plt.plot(sorted(vals))
            else:
                p += plt.plot(sorted(vals))
        fig_name = "courtains/{}.pdf".format(model)
        plt.savefig(fig_name)
        plt.savefig(fig_name.replace("pdf", "png"))
        plt.close()


def make_tables(results):
    for model, data in sorted(results.items()):
        assert data[3] == data[4] + data[5]
        pmin, pmax, psum, plen, yes, no = data
        pavg = psum / plen
        yesperc = round(yes / plen * 1000) / 10
        line = "{:17} & {:9.5} & {:5} & {:5} & {:} \\\\"
        args = [model, pavg, yes, no, yesperc]
        line = line.format(*args)
        print(line)


def proc_file(pickle_file, alpha):
    data = load(open(pickle_file, 'rb'))
    # stats, pvals, var_results = data
    pvals = data[1]

    pmin, pmax = np.min(pvals), np.max(pvals)
    psum = np.sum(pvals)
    plen = len(pvals)
    yes, no = 0, 0
    for p in pvals:
        if p <= alpha:
            yes += 1
        else:
            no +=1
    return pmin, pmax, psum, plen, yes, no


def combine_results(pickle_files, results):
    rv = {}
    for idx, result in enumerate(results):
        pmin, pmax, psum, plen, yes, no = result
        key = pickle_files[idx].split("_")[0]
        if key in rv:
            rv[key][0] = min(rv[key][0], result[0])
            rv[key][1] = max(rv[key][1], result[1])
            rv[key][2] += result[2] # psum
            rv[key][3] += result[3] # plen
            rv[key][4] += result[4] # yes
            rv[key][5] += result[5] # no
        else:
            rv[key] = list(result)
    return rv


def proc_all(figs=False):
    pool = Pool()
    # pickle_files = sample(glob(SRC), 10)
    pickle_files = glob(SRC)
    if figs:
        make_figs(pickle_files)
    results = pool.starmap(proc_file, [(f, 0.05) for f in pickle_files])
    combined_results = combine_results(pickle_files, results)
    make_tables(combined_results)


proc_all()
