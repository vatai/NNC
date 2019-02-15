"""
Generate an org file with containing with the right needed images.
"""

import json
import matplotlib.pyplot as plt
import nnclib.utils

def compile_results():
    """
    This function open processes the results and compiles them into a
    single dictionary.

    Three things are read:

        - one `gold.json` file

        - the results files

        - the weights files
    """

    epsilons = nnclib.utils.get_epsilons()
    all_results = {}

    # read the gold file
    gold = json.load(open("../gold.json"))
    del gold['mobilenetv2']
    for model in gold.keys():
        all_results[model] = {}
        all_results[model]['gold'] = gold[model]
        all_results[model]['weights'] = {}
        all_results[model]['results'] = {}

        for norm in ["norm", "nonorm"]:
            all_results[model]['weights'][norm] = {}
            all_results[model]['results'][norm] = {}

            for eps in epsilons:

                # read the results file
                # path pattern from the runscript... not the best approach
                results_path = "{}-{}.json".format(norm, eps)
                results = json.load(open(results_path, 'r'))
                all_results[model]['results'][norm][eps] = results[model]

                # read the weights file
                # path pattern from get_dense_weight_size.py
                weights_path = "weight_{}_{:02}.json".format(norm, eps)
                weights = json.load(open(weights_path, 'r'))
                all_results[model]['weights'][norm][eps] = weights[model]

    return all_results


def fig_model(model, name):
    """
    Generates a figure with for the normalised and not normalised.
    """
    fig, ax = plt.subplots(1, 2)
    title = ["top1", "top5"]
    # 2 subfigs, top1 and top5 accuracy
    for idx in range(2):
        ax[idx].bar(0, model['gold'][idx + 1], 0.01)
        # Start with the gold
        leg_clr = []
        leg_txt = []
        # y[idx] has the norm and nonorm values
        for norm in model['results'].keys():
            x_vals = []
            y_vals = []
            for eps, val in model['results'][norm].items():
                x_vals.append(eps)
                y_vals.append(val[idx+1])
            plot = ax[idx].plot(x_vals, y_vals)
            leg_clr.append(plot[0])
            leg_txt.append(title[idx] + norm)

            x_vals = []
            y_vals = []
            for eps, val in model['weights'][norm].items():
                org, comp = val[0][0][1], val[0][1][1]
                # print(val[0], org, comp, comp / org, exp)
                y_val = comp / org
                x_vals.append(eps)
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
        for eps, val in results[norm].items():
            line = "| {} | ${}$ | ".format(norm, eps)
            line += " | ".join(map(str, val))
            line += " |\n"
            table.append(line)
    return table


def proc_all_models():
    """
    The main function which processes all the models.
    """

    # Step1: build `results` dictionary.
    results = compile_results()

    # Step2: generate figures.
    for name, model in results.items():
        fig = fig_model(model, name)
        img_name = name + ".png"
        fig.savefig(img_name)
        plt.close(fig)

    # Step3: generate report.
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
