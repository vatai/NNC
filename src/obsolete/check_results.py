"""
Check if the eval* and weight* json files have the same number of
keys.
"""

from json import load
from glob import glob
from os.path import basename

if __name__ == '__main__':
    JSONS = glob("results/combined/*/*/*.json")
    KEYS = sorted(load(open(JSONS[0], 'r')).keys())
    for j in JSONS:
        json = load(open(j, 'r'))
        k = sorted(json.keys())
        if not k == KEYS:
            print(j, k)
        else:
            print("ok: " + basename(j))
