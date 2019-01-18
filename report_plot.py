"""
This program plots the results obtained by
./compare_with_compression.py.
"""

import glob
import os
import json
# from pprint import pprint
import matplotlib.pyplot as plt

json_path = "./report/bbking/*.json"

json_files = glob.glob(json_path)
all_results = {}
for file in json_files:
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]
    all_results[name] = json.load(open(file, 'r'))
# pprint(all_results)


name = "resnet50"

for key in sorted(all_results.keys()):
    out = "| " + str(key) + " | "
    out += " | ".join(map(str, all_results[key][name])) + " |"
    print(out)
    plt.show()


print("Done.")
