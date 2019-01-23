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
        all_results[model]['weights'] = {}
        all_results[model]['results'] = {}
        for norm in ["norm", "nonorm"]:
            all_results[model]['weights'][norm] = {}
            all_results[model]['results'][norm] = {}
            for exp in exps:
                results = get_json_result(base, norm, exp)
                all_results[model]['results'][norm][exp] = results[model]
                weights = get_json_weight(base, norm, exp)
                all_results[model]['weights'][norm][exp] = weights[model]
    return all_results


def fig_model(model, name):
    fig, ax = plt.subplots(1,2)
    title = ["top1", "top5"]
    # 2 subfigs, top1 and top5 accuracy
    for idx in range(2):
        ax[idx].bar(0, model['gold'][idx + 1])
        # Start with the gold
        leg_clr = []
        leg_txt = []
        # y[idx] has the norm and nonorm values
        for norm in model['results'].keys():
            x_vals = []
            y_vals = []
            for exp, val in model['results'][norm].items():
                x_vals.append(exp)
                y_vals.append(val[idx+1])
            plot = ax[idx].plot(x_vals, y_vals)
            leg_clr.append(plot[0])
            leg_txt.append(title[idx] + norm)

            x_vals = []
            y_vals = []
            for exp, val in model['weights'][norm].items():
                org, comp = val[0][0][1], val[0][1][1]
                # print(val[0], org, comp, comp / org, exp)
                y_val = comp / org
                x_vals.append(exp)
                y_vals.append(y_val)
            plot = ax[idx].plot(x_vals, y_vals, '--')
            leg_clr.append(plot[0])
            leg_txt.append("compression: " + norm)

        ax[idx].legend(leg_clr, leg_txt)
        ax[idx].set_ylim(0, 1)
        ax[idx].set_title(name + ": " + title[idx])
    return fig


def get_table(model, branch):
    results = model[branch]
    table = []
    if branch == 'results':
        line = "| {} | nomod | "
        line += " | ".join(map(str, model['gold']))
        line += " |\n"
        table.append(line)
    for norm in results.keys():
        for exp, val in results[norm].items():
            line = "| {} | $10^{{-{}}}$ | ".format(norm, exp)
            line += " | ".join(map(str, val))
            line += " |\n"
            table.append(line)
    return table


def proc_all_models():
    """The main function."""
    base = os.path.expanduser("~/code/NNC/report/20190121/")
    results = compile_results(base)
    for name, model in results.items():
        fig = fig_model(model, name)
        img_name = name + ".png"
        fig.savefig(img_name)
        plt.close(fig)


    report_name = "report.org"
    with open(report_name, 'w') as report_file:
        org_header = """#+LATEX_HEADER: \\usepackage[margin=5mm]{geometry}
#+OPTIONS: toc:nil

"""
        report_file.write(org_header)
        for name, model in results.items():
            img_name = name + ".png"
            results_table = get_table(model, 'results')
            compression_table = get_table(model, 'weights')
            report_file.write("* {}\n".format(name))
            report_file.write("file:{}\n".format(img_name))
            report_file.writelines(results_table)
            report_file.write("\n")
            report_file.writelines(compression_table)


proc_all_models()
