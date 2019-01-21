"""
Generate an org file with containing with the right needed images.
"""

import os
import json
from pprint import pprint
import matplotlib.pyplot as plt


def make_gold_fn(base, *args, **kwargs):
    fname = "gold.json"
    return os.path.join(base, fname)


def get_json_gold(*args, **kwargs):
    return json.load(open(make_gold_fn(*args, **kwargs)))


def make_result_fn(base, norm, exp):
    fname = "{}-{:02}.json".format(norm, exp)
    return os.path.join(base, fname)


def get_json_result(*args, **kwargs):
    return json.load(open(make_result_fn(*args, **kwargs)))


def make_weight_fn(base, norm, exp):
    fname = "weight_{}_{}.json".format(norm, 10**-exp)
    return os.path.join(base, fname)


def get_json_weight(*args, **kwargs):
    output = json.load(open(make_weight_fn(*args, **kwargs)))
    del output['mobilenet']
    return output


def compile_results(base, exps=range(1, 6)):
    gold = get_json_gold(base)
    all_results = {}
    for model in gold.keys():
        all_results[model] = {}
        all_results[model]['gold'] = gold[model]
        for norm in ["norm", "nonorm"]:
            all_results[model]['weights'] = {norm: {}}
            all_results[model]['results'] = {norm: {}}
            for exp in exps:
                results = get_json_result(base, norm, exp)
                all_results[model]['results'][norm][exp] = results[model]
                weights = get_json_weight(base, norm, exp)
                all_results[model]['weights'][norm][exp] = weights[model]
    return all_results


def proc_model(model):
    fig, ax = plt.subplots(1,2)
    leg_txt = []
    leg_clr = []
    y_vals = []
    plots = []
    leg_txt = ["top1", "top5"]
    # 2 subfigs, top1 and top5 accuracy
    for idx in range(2):
        # Start with the gold
        ax[idx].bar(0, model['gold'][idx + 1])
        y_vals.append({})
        plots.append({})
        leg_clr = []
        # y[idx] has the norm and nonorm values
        for norm in model['results'].keys():
            x_vals = []
            y_vals[idx][norm] = []
            for exp, val in model['results'][norm].items():
                x_vals.append(exp)
                y_vals[idx][norm].append(val[idx+1])
                plots[idx][norm] = ax[idx].plot(x_vals, y_vals[idx][norm])
                leg_clr.append(plots[idx][norm])
        ax[idx].legend(leg_clr, leg_txt)
        ax[idx].set_ylim(0, 1)
    plt.show()


def proc_all_models():
    """The main function."""
    base = os.path.expanduser("~/code/NNC/report/20190121/")
    gold = get_json_gold(base)
    ar = compile_results(base)
    for model in ar.keys():
        proc_model(ar[model])
        break
    # pprint(ar)


proc_all_models()
