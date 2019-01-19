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


name = "vgg16"

legend = ([], [])
for i, key in enumerate(sorted(all_results.keys())):
    out = "| " + str(key) + " | "
    out += " | ".join(map(str, all_results[key][name])) + " |"
    print(out)
    p = plt.bar(i, all_results[key][name][2])
    legend[0].append(p[0])
    legend[1].append(key)
print(legend[1])
plt.legend(legend[0], legend[1])
plt.show()


print("Done.")
