"""
This program plots the results obtained by
./compare_with_compression.py.
"""

import glob
import os
import json
import matplotlib.pyplot as plt


def proc_entry(name, all_results):
    """
    Save a plot for one set of data, and return the org-mode text to
    be written to a text file.
    """
    lgn1 = ([], [])
    lgn2 = ([], [])
    out = "\n* {}".format(name)
    out += "\nfile:{}.png".format(name)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i, key in enumerate(sorted(all_results[0].keys())):
        out += "\n| " + str(key) + " | "
        print(key)
        out += " {} |".format(all_results[1][key][name]) 
        out += " | ".join(map(str, all_results[0][key][name][1:])) + " |"
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        p1 = ax1.bar(i, all_results[0][key][name][1])
        p2 = ax2.bar(i, all_results[0][key][name][2])
        lgn1[0].append(p1[0])
        lgn1[1].append(key)
        lgn2[0].append(p2[0])
        lgn2[1].append(key)
    fig.legend(lgn1[0], lgn1[1], 'upper left')
    fig.legend(lgn2[0], lgn2[1], 'upper right')
    ax1.set_title(name + " top1")
    ax2.set_title(name + " top5")
    plt.savefig(name)
    print(all_results[1].keys())
    return out


def get_results(json_path):
    """
    Gather data from all json files from a directory.  The parameter
    is a pattorn selecting only json files.
    """
    json_files = glob.glob(json_path)
    acc_results = {}
    weight_result = {}
    for file in json_files:
        name = os.path.basename(file)
        name = os.path.splitext(name)[0]
        acc_results[name] = json.load(open(file, 'r'))
        # wfile = file[:18] + "/weight_" + file[18:-8] + "_{}".format(10**int(file[-8:-5])) + ".json"
        wfile = "{}weight_{}_{}.json".format(file[:18],
                                             file[18:-8],
                                             10**int(file[-8:-5]))
        print(file, wfile)
        weight_result[name] = json.load(open(wfile, 'r'))
    return (acc_results, weight_result)


json_path = "./report/20190121/n*.json"
all_results = get_results(json_path)
with open("report.org", 'w') as f:
    for name in all_results[0]['norm-02'].keys():
        print(name)
        out = proc_entry(name, all_results)
        f.write(out)
print("Done.")
