"""
Second report generation utility.  For combined.py which generates
different dictionary.
"""

from glob import glob
from json import load
from os.path import join, basename
import matplotlib.pyplot as plt
from pprint import pprint
from nnclib.utils import sum_weights


def collect_data(base='.'):
    """
    Function collect the data in the the current (or `base`)
    directory, into a big dictionary from the weights and accuracy
    json files.
    """
    files = glob(join(base, 'accuracy_*.json'))
    data = {}
    for file in files:
        fields = basename(file).split('_')
        fields = fields[1:-2]
        accuracy = load(open(file))
        weights = load(open(file.replace('accuracy', 'weights')))
        for model in accuracy.keys():
            # add both accuracy and weights
            data[tuple(fields+[model])] = accuracy[model] + weights[model]
    eps_idx = 0
    for i, v in enumerate(fields):
        if v[:3] == 'eps':
            eps_idx = i
    return data


def per_column(data, idx):
    result = {}
    for key, value in data.items():
        new_key = key[: idx] + key[idx:][1:]
        if key[idx] in result:
            result[key[idx]][new_key] = value
        else:
            result[key[idx]] = {new_key: value}
    return result


def dictmap(f, d):
    return dict(map(lambda t: (t[0], f(t[1])), d.items()))


def cmpdict(d1, d2):
    if isinstance(d1, dict) and isinstance(d2, dict):
        for key in d1.keys():
            if key not in d2:
                return False
        for key in d2.keys():
            if key not in d1:
                return False
        for key in d1.keys():
            if not cmpdict(d1[key], d2[key]):
                return False
        return True
    if not isinstance(d1, dict) and not isinstance(d2, dict):
        return d1 == d2
    return False


def get_column_idx(data, prefix):
    for key in data.keys():
        for i, value in enumerate(key):
            if value.startswith(prefix):
                return i


def proc_all(all_weights_file="../all_weights.json"):
    all_weights = load(open(all_weights_file, 'r'))
    data = collect_data()
    eps_idx = get_column_idx(data, 'eps')
    per_model = per_column(data, -1)
    per_model_eps = dictmap(lambda t: per_column(t, eps_idx), per_model)

    for model, model_data in per_model_eps.items():
        print(model)
        for ekey in sorted(model_data.keys()):
            print(model, ekey)
            if ekey == 'eps0':
                data = list(model_data[ekey].items())[0][1]
                mweights = sum_weights(data[3:])
            else:
                eps = ekey[3:]
                for params, data in model_data[ekey].items():
                    top1, top5 = data[1], data[2]
                    aweights = all_weights[model]
                    cweights = sum_weights(data[3:])
                    eff_cweights = aweights - (mweights - cweights)
                    comp_rat = eff_cweights / aweights
                    sparams = "-".join(map(lambda t: t[0] + t[-1], params))
                    line = "{:0.04} {:14.4} {:14.04} {}"
                    line = line.format(comp_rat, top1, top5, sparams) 
                    print(line)


proc_all()
