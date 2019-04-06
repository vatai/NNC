"""Simple script used to test the correctness of the custom op
``unsort``.

The expected behaviour is that the last two matrices printed
(``exact_output`` and ``output``) to be similar (or the same).

"""


def decompose(weights):
    """This function converts the weights (the ``kernel`` matrix) to the
    approximating representation of ``indices`` and ``mean``.  See
    details in :py:class:`custom_layer_test.CompressedPrototype`.

    """
    indices = np.argsort(weights, axis=0)
    sorted_weights = np.take_along_axis(weights, indices, axis=0)
    mean = np.mean(sorted_weights, axis=1)
    return np.argsort(indices, axis=0), mean


if __name__ == '__main__':
    import os

    import numpy as np
    import tensorflow
    import keras.backend as K

    # Suppress warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    # Load module from ``.so``.
    unsort_module = tensorflow.load_op_library('./src/unsortOp/unsort_ops.so')

    # Basically a random input batch.  Here 
    inputs = np.array([[ 0.71238331,  1.38283577,  0.37231196],
                       [ 0.08651744, -1.05230163,  1.00697538],
                       [-1.56814483, -0.30826115, -0.13245332],
                       [-0.55250484, -1.53117801,  1.65068967]])
    inputs_tensor = K.variable(inputs)

    weights = np.array([[-0.5, 0.5],
                        [0.5, -0.5],
                        [0.0, 0]])
    indices, mean = decompose(weights)
    indices_tensor = K.variable(indices, dtype='int32')
    mean_tensor = K.variable(mean)

    output = unsort_module.unsort(inputs_tensor,
                                  indices_tensor,
                                  mean_tensor)

    print("inputs")
    print(inputs)

    print("weights")
    print(weights)

    print("indices")
    print(indices)

    print("mean")
    print(mean)

    exact_output = np.dot(inputs, weights)
    print("exact_output")
    print(exact_output)

    print('output')
    print(K.eval(output))
