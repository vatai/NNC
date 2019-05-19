#!/usr/bin/python3

import itertools

model_names = ["resnet50", "nasnetlarge", "nasnetmobile"]

datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

layers = ["D", "C2"]

updaters = ['p', 'm', 'np', 'nm']

deltas = [0.05, 0.01, ]#0.005, 0.001]  # careful with 0!


def _make_updaters(updater):
    result = list(map(lambda d: "{}-{}".format(updater, d), deltas))
    if updater in ['m', 'nm']:
        result += [updater]
    return result


def _make_all_updater_lists():
    result = []
    for t in layers:
        for u in updaters:
            result += ["{},{}".format(t, up) for up in _make_updaters(u)]
    for u1 in updaters:
        for u2 in updaters:
            result += ["D,{};C2,{}".format(up1, up2) for up1 in
                       _make_updaters(u1) for up2 in _make_updaters(u2)]
    return result


print("START")
print(len(_make_all_updater_lists()))

cmds = []
for mn in model_names:
    for ds in datasets:
        for ul in _make_all_updater_lists():
            cmd = "./launcher.sh run_exp.py with \\\n"
            cmd += "  'experiment_args.model_name\"{}\"' \\\n".format(mn)
            cmd += "  'experiment_args.dataset_name\"{}\"' \\\n".format(ds)
            cmd += "  'experiment_args.coded_updater_list\"{}\"'\n".format(ul)
            cmds.append(cmd)

with open("big_run.sh", "w") as f:
    f.write("#!/bin/bash\n\n")
    f.writelines("\n\n".join(cmds))
