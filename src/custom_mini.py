import numpy as np
import tensorflow
import keras.backend as K

unsort_module = tensorflow.load_op_library('./src/unsortOp/unsort_ops.so')


def eval_print(x):
    print(K.eval(x))


def shape_print(x):
    print(K.int_shape(x))


def decompose(weights):
    indices = np.argsort(weights, axis=0)
    sorted_weights = np.take_along_axis(weights, indices, axis=0)
    mean = np.mean(sorted_weights, axis=1)
    return indices, mean

weights = np.array([[10, 19],
                    [21, 10]])
indices, mean = decompose(weights)
indices_tensor = K.variable(indices, dtype='int32')
mean_tensor = K.variable(mean)

inputs = np.array([[1.2, 3.4],
                   [2.3, 4.5],
                   [9.9, 7.7]])
inputs_tensor = K.variable(inputs)

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
eval_print(output)

