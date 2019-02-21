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


all_weights_file="../all_weights.json"
all_weights = load(open(all_weights_file, 'r'))


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


def rearange(data, *args):
    rv = {}
    for fields, results in data.items():
        d = rv
        for a in args[:-1]:
            if isinstance(a, list):
                key = tuple([fields[i] for i in a])
            else:
                key = fields[a]
            if key not in d:
                d[key] = {}
            d = d[key]
        a = args[-1]
        if isinstance(a, list):
            key = tuple([fields[i] for i in a])
        else:
            key = fields[a]
        d[key] = results
    return rv


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


def proc_cfg(cfg_info, mbest):
    cfg, cfg_dict = cfg_info
    mweights, mtop1, mtop5, aweights = mbest
    for ekey, data in sorted(cfg_dict.items()):
        eps = float(ekey[3:])
        top1, top5 = data[1:3]
        cweights = sum_weights(data[3:])
        eff_cweights = aweights - (mweights - cweights)
        comp_rat = eff_cweights / aweights
        comp_size_rat = (2 * eff_cweights) / (4 * aweights)
        line = "{:8} {:8.04} {:8.04} {:14.4} -> {:10.4} {:14.4} -> {:10.4}"
        args = [eps, comp_rat, comp_size_rat, mtop1, top1, mtop5, top5]
        line = line.format(*args)
        print(line)
    plt_data = map(lambda t: , sorted(cfg_dict.items()))
    return plt.plot(plt_data)


def proc_model(model_info):
    model_name, model_data = model_info
    print("===== {} =====".format(model_name))
    cfg0 = ('norm0', 'quant0', 'dsmooth0', 'csmooth0')
    data = model_data[cfg0]['eps0']
    mweights = sum_weights(data[3:])
    mtop1, mtop5 = data[1:3]
    aweights = all_weights[model_name]
    mbest = mweights, mtop1, mtop5, aweights
    for cfg_info in sorted(list(model_data.items())[0:1]):
        print("-- {} --".format(cfg_info[0]))
        p = proc_cfg(cfg_info, mbest)
        f = "_".join([model_name]+list(cfg_info[0]))
        print("Saving {}".format(f))
        # plt.savefig(p, f)


def proc_all():
    data = collect_data()
    fields = next(iter(data.keys()))
    eps_idx = get_column_idx(data, 'eps')
    per_model = per_column(data, -1)
    per_model_eps = dictmap(lambda t: per_column(t, eps_idx), per_model)

    result = rearange(data, 5, [0, 1, 2, 3], 4)
    for model_info in result.items():
        proc_model(model_info)

proc_all()
