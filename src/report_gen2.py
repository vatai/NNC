"""
Second report generation utility.  For combined.py which generates
different dictionary.
"""

from glob import glob
from json import load
from os.path import join, basename
from pprint import pprint


def collect_data(base='.'):
    files = glob(join(base, 'accuracy_*'))
    data = {}
    for file in files:
        fields = basename(file).split('_')
        accuracy = load(open(file))
        weights = load(open(file.replace('accuracy', 'weights')))
        for model in accuracy.keys():
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


def get_column_idx(data, prefix='eps'):
    for key in data.keys():
        for i, value in enumerate(key):
            if value.startswith(prefix):
                return i


def proc_all():
    data = collect_data()
    pm = per_column(data, -1)

proc_all():
for k, v in data.items():
    # print(k)
    # pprint(v[1:3])

    r = cmpdict(per_model, pm)
    print(r)

pme = dictmap(lambda t: per_column(t, 5), pm)
print(pme.keys())
print(sorted(pme['xception'].keys()))
print(pme['xception']['eps0.001'].keys())
