"""
Experiment description:

- Data: cifar10

- Model: modified VGG16
"""

from functools import partial
from os.path import exists
import sys

from keras.layers import Dense
from sacred import Experiment
from sacred.observers import FileStorageObserver, TelegramObserver
import tensorflow as tf

from nnclib.compression import evaluator, trainer, \
    reshape_norm_meld, WeightsUpdater
from nnclib.experiments import run_experiment, model_factory, data_factory

ex = Experiment()
ex.observers.append(FileStorageObserver.create(sys.argv[0]+'resluts'))
if exists('telegram.json'):
    ex.observers.append(TelegramObserver.from_config('telegram.json'))


@ex.config
def config():
    """Experiment parameters."""
    # pylint: disable=unused-variable
    # flake8: noqa: F841
    seed = 42 # random seed
    data_getter = data_factory.cifar10_float32
    model_maker = model_factory.vgg16_mod
    num_gpus = 1
    compile_args = {}
    fit_args = dict(epochs=300,
                    validation_split=0.2,
                    verbose=2,
                    batch_size=128,
                    callbacks=[
                        WeightsUpdater(
                            updater_list=[(Dense, reshape_norm_meld)],
                            on_nth_epoch=10)
                    ])
    evaluation = evaluator
    modifier = None


@ex.automain
def main(_seed, data_getter, model_maker, num_gpus, compile_args,
         fit_args, evaluation, modifier):
    """Experiment automain function."""
    tf.set_random_seed(_seed)
    return run_experiment(data_getter,
                          partial(model_maker,
                                  num_gpus=num_gpus,
                                  compile_args=compile_args),
                          partial(trainer, **fit_args),
                          evaluator=evaluation,
                          modifier=modifier)

