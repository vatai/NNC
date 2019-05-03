"""Data factories: functions which return a datasets to be used by
`run_experiment`.

"""

from keras.datasets import cifar10


def cifar10_float32():
    """Data factory for cifar10 dataset, flattened and converted to
    float32.

    """
    (x_train, t_train), (x_test, t_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    t_train = t_train.flatten()
    t_test = t_test.flatten()
    return (x_train, t_train), (x_test, t_test)
