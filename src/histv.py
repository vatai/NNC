"""History data extractor."""

import pickle
import glob
import sys

file_list = glob.glob(sys.argv[1])
for file_name in file_list:
    hist = pickle.load(open(file_name, "rb"))
    result = hist.history['sparse_categorical_accuracy']
    print(file_name)
    print(result)
