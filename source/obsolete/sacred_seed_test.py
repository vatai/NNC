import numpy as np
from tensorflow import set_random_seed
import keras
from sacred import Experiment

EX = Experiment()


@EX.config
def config():
    seed = 42


@EX.automain
def main(_seed):
    print(_seed)
    set_random_seed(_seed)
    result = np.random.randn(2, 3)
    print(result)

    result = keras.backend.random_normal([2, 3])
    print(keras.backend.eval(result))
