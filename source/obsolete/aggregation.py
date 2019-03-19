"""This program aggregates the results of compressing the dense layers."""

import pickle
from pprint import pprint
from sacred import Experiment
from sacred.observers import FileStorageObserver
from nnclib.utils import get_results_dir

EX = Experiment()
EX.observers.append(FileStorageObserver.create(get_results_dir(__file__)))


@EX.config
def config():
    sources = [
        "./report/weights.pickl",
        "./report/gold.pickl",
        "./report/nonorm.pickl",
        "./report/norm.pickl"
        ]


@EX.automain
def main(sources):
    weights, gold, nonorm, norm = map(lambda t: pickle.load(open(t, 'rb')),
                                      sources)
    pprint(weights)
    pprint(gold)
    pprint(nonorm)
    pprint(norm)
