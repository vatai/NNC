"""
A program to investigate pretrained models, about the distribution of
weights.
"""

from glob import glob
from os.path import splitext, basename, join
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt


def proc_file(name, base="report/s_shapes"):
    """Process a single layer if it is Dense (or other given type)."""
    dense = np.load(name)
    sorted_dense = np.sort(dense, axis=0)
    norms_dense = np.linalg.norm(dense, axis=0)

    normalised = sorted_dense / norms_dense[np.newaxis, :]
    for norm in [0, 1]:
        title = basename(name)
        title = splitext(title)[0]
        title = "{}_nrm{}".format(title, norm)
        if norm:
            plt.plot(normalised)
        else:
            plt.plot(sorted_dense)
        # plt.title(title)
        plt.ylim(-1, 1)
        # plt.show()

        title = join(base, title)
        title += ".pdf"
        print("Saving fig: {}".format(title))
        plt.savefig(title)
        plt.savefig(title.replace('pdf', 'png'))
        plt.close()


def proc_all_files(src="report/weights/*"):
    """Process all pre-generated weight files."""
    files = glob(src)
    pool = Pool()
    pool.map(proc_file, files)

proc_all_files()
print("Done")
