"""
Collect the weights of Dense and Conv2D layers, and store them in json
files.
"""

from json import dump
from multiprocessing import Pool
from utils import model_dict


def proc_model(params):
    num_models = len(model_dict.items())
    idx, name = params
    model = model_dict[name][0]()
    print(">>> Started {}/{}".format(idx + 1, num_models))
    count_params = model.count_params()
    print(">>> {} has {} parameter".format(name, count_params))
    print(">>> Done {}/{}".format(idx + 1, num_models))
    return (name, count_params)


def proc_all(outfile="all_weights.json"):
    pool = Pool(8)
    results = pool.map(proc_model, enumerate(model_dict.keys()))
    print(results)
    dump(dict(results), open(outfile, 'w'))


proc_all()
print("Done.")
