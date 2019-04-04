import numpy as np
import tensorflow
import keras.backend as K

unsort_module = tensorflow.load_op_library('./src/unsortOp/unsort_ops.so')


def eval_print(x):
    print(K.eval(x))


def shape_print(x):
    print(K.int_shape(x))


a = K.variable(np.array([[1.2, 3.4],
                         [2.3, 4.5],
                         [9.9, 7.7]]))
b = K.variable(np.array([[0, 1], [1, 0]], dtype=np.int16), dtype='int32')
c = K.variable(np.array([10, 20]))
output = unsort_module.unsort(a, b, c)

eval_print(a)
eval_print(b)
eval_print(output)
shape_print(output)
