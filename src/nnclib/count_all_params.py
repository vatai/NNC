"""
Collect the weights of Dense and Conv2D layers, and store them in json
files.
"""

from json import dump
from utils import model_dic


def proc_all(outfile="all_weights.json"):
    num_models = len(model_dic.items())
    results = {}
    dump(results, open(outfile, 'w'))
    for idx, (name, mcls) in enumerate(model_dic.items()):
        print(">>> Started {}/{}".format(idx + 1, num_models))
        model = mcls[0]()
        results[name] = model.count_params()
        print(">>> {} has {} parameter".format(name, results[name]))
        dump(results, open(outfile, 'w'))
        print(">>> Done {}/{}".format(idx + 1, num_models))


proc_all()
print("Done.")
