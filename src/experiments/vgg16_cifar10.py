"""
Experiment description:

- Data: cifar10

- Model: modified VGG16
"""

from functools import partial
from os.path import exists
import sys

from keras.callbacks import TensorBoard
from keras.layers import Dense
from sacred import Experiment
from sacred.observers import FileStorageObserver, TelegramObserver

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
    data_getter = data_factory.cifar10_float32
    model_maker = model_factory.vgg16_mod
    compile_args = {}
    fit_args = dict(epochs=300,
                    validation_split=0.2,
                    verbose=2,
                    batch_size=128,
                    callbacks=[
                        TensorBoard('./log', 1),
                        WeightsUpdater(
                            updater_list=[(Dense, reshape_norm_meld)],
                            on_nth_epoch=10)
                    ])
    evaluation = evaluator
    modifier = None


@ex.automain
def main(data_getter, model_maker, compile_args, fit_args, evaluation, modifier):
    """Experiment automain function."""
    return run_experiment(data_getter,
                          partial(model_maker, compile_args=compile_args),
                          partial(trainer, **fit_args),
                          evaluator=evaluation,
                          modifier=modifier)

