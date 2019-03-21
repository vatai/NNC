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
    """Process a single layer and save the s-shape plot."""
    dense = np.load(name)
    dense = np.sort(dense, axis=0)

    for norm in [0, 1]:
        title = basename(name)
        title = splitext(title)[0]
        title = "{}_nrm{}".format(title, norm)
        if norm:
            dense /= np.linalg.norm(dense, axis=0)
        plt.plot(dense)
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


if __name__ == '__main__':
    proc_all_files()
    print("Done")
