from pprint import pprint
import os
import numpy as np
import sacred
import keras
import keras.initializers
import keras.backend as K
from nnclib.generators import CropGenerator

EX = sacred.Experiment()


@EX.config
def config():
    compile_args = {'optimizer': 'RMSprop',
                    'loss': 'categorical_crossentropy',
                    'metrics': ['categorical_accuracy',
                                'top_k_categorical_accuracy']}
    gen_args = {'img_dir': os.path.expanduser("~/tmp/ilsvrc/db"),
                'val_file': os.path.expanduser("~/tmp/ilsvrc/caffe_ilsvrc12/val.txt"),
                'batch_size': 32,
                'fast_mode': 1}
    eval_args = {'max_queue_size': 10,
                 'workers': 1,
                 'use_multiprocessing': True,
                 'verbose': True}
    # proc_args = {'norm': 0, 'epsilon': 0, 'dense_smooth': 0, 'conv_smooth': 0, 'quantization': 0}
    seed = 42


class MyLayer(keras.layers.Layer):
    """Implementation of a custom, compressed dense layer.

    The layer implements a simple xA + b linear transformation.  The
    bias ``b`` is implemented as usual.

    In this implementation, the ``A`` matrix is stored as a matrix of
    ``indices`` and a vector which is per column ``mean`` of the rows
    of ``A``.  If ``A`` has shape ``(in_dim, out_dim)``, then
    ``indices`` have the same shape, and ``mean`` has ``(in_dim,)``
    shape.

    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        def my_init(shape, dtype=None):
            return K.variable(np.zeros(shape=shape), dtype='int32')

        # indices
        initializer = my_init
        self.indices = self.add_weight(name='indices',
                                       dtype='int32',
                                       shape=(input_shape[1], self.output_dim),
                                       initializer=initializer,
                                       trainable=False)
        # mean
        self.mean = self.add_weight(name='mean',
                                    shape=(input_shape[1], ),
                                    initializer='uniform',
                                    trainable=False)
        # bias
        initializer = keras.initializers.get('uniform')
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim, ),
                                    initializer=initializer,
                                    trainable=True)

        super(MyLayer, self).build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        import tensorflow
        so_name = './src/unsortOp/unsort_ops.so'
        unsort_module = tensorflow.load_op_library(so_name)
        output = unsort_module.unsort(inputs, self.indices, self.mean)
        return K.bias_add(output, self.bias, data_format='channels_last')

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.output_dim)


def get_new_weights(weights):
    """Args:

        weights (tuple): The weights of a dense layer, containing:
        ``kernel`` with shape ``(in_dim, out_dim)``; ``bias`` with shape
        ``(out_dim, )``
    
    Returns: list: ``[indices, mean, bias]`` where ``bias`` is the
       original bias, and the ``kernel`` can be approximated using
       ``indices`` and ``mean``.

    """
    kernel, bias = weights
    out_dim, in_dim = kernel.shape
    indices = np.argsort(kernel, axis=0)

    sorted_kernel = np.take_along_axis(kernel, indices, 0)
    mean = np.mean(sorted_kernel, axis=1)
    return [bias, indices, mean]


@EX.automain
def main(compile_args, gen_args, eval_args):
    # model = keras.applications.vgg16.VGG16()
    model = keras.applications.resnet50.ResNet50()
    last = model.layers[-1]
    weights = last.get_weights()

    fc = model.layers[-2]   # the last layer we keep
    output_dim = K.int_shape(last.output)[1]
    my_layer = MyLayer(output_dim)
    new_outputs = my_layer(fc.output)

    new_model = keras.Model(inputs=model.input,
                            outputs=new_outputs)
    new_last = new_model.layers[-1]
    new_weights = new_last.get_weights()
    pprint(list(map(np.shape, new_weights)))

    new_weights = get_new_weights(weights)
    pprint(list(map(np.shape, new_weights)))
    new_last.set_weights(new_weights)

    model.compile(**compile_args)
    new_model.compile(**compile_args)
    generator = CropGenerator(**gen_args)

    result = model.evaluate_generator(generator, **eval_args)
    print("Evaluation results:")
    print(result)

    new_result = new_model.evaluate_generator(generator, **eval_args)
    print("NEW Evaluation results:")
    print(new_result)

    print('Done.')
