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
    return np.argsort(indices, axis=0), mean


# inputs = np.random.randn(4, 3)
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
eval_print(output)

