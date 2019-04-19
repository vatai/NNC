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
            key = tuple(fields + [model])
            value = accuracy[model] + weights[model]
            data[key] = value
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
    model_name, mweights, mtop1, mtop5, aweights = mbest
    comp_ratio_list, size_ratio_list = [], []
    top1_list, top5_list, eps_list = [], [], []
    for ekey, data in sorted(cfg_dict.items()):
        eps = float(ekey[3:])
        top1, top5 = data[1:3]
        cweights = sum_weights(data[3:])
        eff_cweights = aweights - (mweights - cweights)
        comp_ratio = eff_cweights / aweights
        eff_size = 4 * aweights - 4 * mweights + 2 * cweights
        size_ratio = eff_size / (4 * aweights)
        line = "{:8} & {:8.04} & {:8.04} & {:14.4} & {:10.4} & {:14.4} & {:10.4}"
        args = [eps, comp_ratio, size_ratio, mtop1, top1, mtop5, top5]
        line = line.format(*args)
        print(line)
        comp_ratio_list.append(comp_ratio)
        size_ratio_list.append(size_ratio)
        top1_list.append(top1)
        top5_list.append(top5)
        eps_list.append(eps)

    fig, ax = plt.subplots()
    top1_color = "red"
    top5_color = "blue"
    eps_list = list(sorted(map(lambda t: float(t[3:]), cfg_dict.keys())))
    ax.plot(eps_list, top1_list, color=top1_color)
    ax.plot(eps_list, top5_list, color=top5_color)
    ax.plot(eps_list, comp_ratio_list, linestyle="--")
    ax.plot(eps_list, size_ratio_list, linestyle="--")
    ax.axhline(mtop1, color=top1_color, linestyle=":")
    ax.axhline(mtop5, color=top5_color, linestyle=":")
    ax.legend(['top1', 'top5',
               'ratio (#param)', 'ratio (bytes)',
               'top1 (orig)', 'top5 (orig)'])
    # fig.show()
    # plt.show()
    tbl = list(zip(top1_list, top5_list, comp_ratio_list, size_ratio_list, eps_list))
    gtop1 = max(tbl, key=lambda t: t[0])
    gtop5 = max(tbl, key=lambda t: t[1])
    gcomp = min(tbl, key=lambda t: t[2])
    gsize = min(tbl, key=lambda t: t[3])
    args = [gtop1[0], gtop1[3], gtop1[4],
            gtop5[1], gtop5[3], gtop5[4],
            gsize[0], gsize[1], gsize[3], gsize[4]]
    args = map(lambda t: round(1000*t)/10, args)
    line = " & {:5} & {:5} & {}  & {:5} & {:5} & {}  & {:5} & {:5} & {:5} & {}".format(*args)
    #print(line)
    return fig


def proc_model(model_info):
    model_name, model_data = model_info
    #### print("===== {} =====".format(model_name))
    cfg0 = next(iter(sorted(model_data.keys())))
    #### print(cfg0)
    data = model_data[cfg0]['eps0']
    mweights = sum_weights(data[3:])
    mtop1, mtop5 = data[1:3]
    aweights = all_weights[model_name]
    mbest = model_name, mweights, mtop1, mtop5, aweights
    for cfg_info in model_data.items():
        line = "{:17} & ".format(model_name)
        line += "".join(map(lambda t: t[-1], cfg_info[0]))
        line += "0_" if len(cfg_info[0]) < 4 else "_" 
        print(line, end="")
        #print("-- {} --".format(cfg_info[0]))
        # print("-- {} --".format(cfg_info[0]))
        fig = proc_cfg(cfg_info, mbest)
        fig_name = "_".join([model_name]+list(cfg_info[0]))
        fig_name = "figs/{}.pdf".format(fig_name)
        fig.savefig(fig_name)
        fig.savefig(fig_name.replace("pdf", "png"))
        plt.close(fig)


def proc_all():
    data = collect_data()
    fields = next(iter(data.keys()))
    eps_idx = get_column_idx(data, 'eps')
    per_model = per_column(data, -1)
    per_model_eps = dictmap(lambda t: per_column(t, eps_idx), per_model)
    print('epsilon index: {}'.format(eps_idx))
    result = rearange(data, eps_idx + 1, list(range(eps_idx)), eps_idx)
    for model_info in result.items():
        proc_model(model_info)


if __name__ == '__main__':
    all_weights_file="../all_weights.json"
    all_weights = load(open(all_weights_file, 'r'))
    proc_all()
