"""
Generate an org file with containing with the right needed images.
"""

import json
import matplotlib.pyplot as plt
import nnclib.utils

AMMEND = json.load(open('../dense_weights.json'))


def compile_results(smooth=0):
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

        for inorm, norm in enumerate(["nonorm", "norm"]):
            all_results[model]['weights'][norm] = {}
            all_results[model]['results'][norm] = {}

            for eps in epsilons:

                # read the results file
                # path pattern from the runscript... not the best approach
                results_path = "eval_norm{}_quant0_smooth{}_eps{}.json"
                results_path = results_path.format(inorm, smooth, eps)
                results = json.load(open(results_path, 'r'))
                all_results[model]['results'][norm][eps] = results[model]

                # read the weights file
                # path pattern from get_dense_weight_size.py
                weights_path = "weight_norm{}_quant0_smooth{}_eps{}.json"
                weights_path = weights_path.format(inorm, smooth, eps)
                weights = json.load(open(weights_path, 'r'))
                weights[model] = list(zip(AMMEND[model], weights[model]))
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

        # And finally the gold as a horizontal bar
        plot = ax[idx].axhline(model['gold'][idx+1], color='magenta', linestyle=':')
        leg_clr.append(plot)
        leg_txt.append("unmodified")

        ax[idx].legend(leg_clr, leg_txt)
        ax[idx].set_ylim(0, 1)
        ax[idx].set_title(name + ": " + title[idx])
    return fig


def get_table(model, branch):
    results = model[branch]
    table = []
    if branch == 'results':
        table.append("| norm | epsilon | top1 | top5 | top1% | top5% |\n")
        val = model['gold'][1:] + [1., 1.]
        val = list(map(lambda t: "{:5.5}".format(t), val))
        line = "|  | nomod | "
        line += " | ".join(map(str, val))
        line += " |\n"
        top1, top5 = model['gold'][1:]
    else:
        line = "| norm | epsilon | rows | original | compress | compress rat |\n"
    print(line, end="")
    table.append(line)
    for norm in results.keys():
        for eps, val in results[norm].items():
            line = "| {} | ${}$ | ".format(norm, eps)
            if branch == 'results':
                ctop1, ctop5 = val[1:]
                ptop1, ptop5 = ctop1/top1, ctop5/top5
                val = [ctop1, ctop5, ptop1, ptop5]
                val = list(map(lambda t: "{:5.5}".format(t), val))
                print(val)
            else:
                rows = val[0][0][0]
                oldcols = val[0][0][1]
                newcols = val[0][1][1]
                val = [rows, oldcols, newcols, newcols/oldcols]
            line += " | ".join(map(str, val))
            line += " |\n"
            print(line, end="")
            table.append(line)
    return table


def proc_all_models(smooth=0):
    """
    The main function which processes all the models.
    """

    # Step1: build `results` dictionary.
    results = compile_results(smooth)

    # Step2: generate figures.
    for name, model in results.items():
        fig = fig_model(model, name)
        img_name = name + "{}.png".format(smooth)
        fig.savefig(img_name)
        plt.close(fig)

    # Step3: generate report.
    report_name = "report{}.org".format(smooth)
    with open(report_name, 'w') as report_file:
        org_header = """
#+LATEX_CLASS_OPTIONS: [a4paper,9pt]
#+LATEX_HEADER: \\usepackage[margin=5mm]{geometry}
#+OPTIONS: toc:nil

"""
        report_file.write(org_header)
        for name, model in results.items():
            img_name = name + "{}.png".format(smooth)
            results_table = get_table(model, 'results')
            compression_table = get_table(model, 'weights')
            report_file.write("* {}\n".format(name))
            report_file.write("file:{}\n".format(img_name))
            report_file.writelines(results_table)
            report_file.write("\n")
            report_file.writelines(compression_table)


proc_all_models(0)
proc_all_models(1)
