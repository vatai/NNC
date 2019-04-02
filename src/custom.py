from nnclib.generators import CropGenerator
from pprint import pprint
import os
import numpy as np
import sacred
import keras
import keras.initializers
import keras.backend as K

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
                 'workers': 4,
                 'use_multiprocessing': True,
                 'verbose': True}
    # proc_args = {'norm': 0, 'epsilon': 0, 'dense_smooth': 0, 'conv_smooth': 0, 'quantization': 0}
    seed = 42


def my_init(shape, dtype=None):
    # print('>>> ', shape)
    # print('>>> ', np.zeros(shape=shape))
    return K.variable(np.zeros(shape=shape), dtype='int32')


class MyLayer(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        # initializer = keras.initializers.Zeros(dtype='int32')
        initializer = my_init
        self.indices = self.add_weight(name='indices',
                                       dtype='int32',
                                       shape=(input_shape[1], self.output_dim),
                                       initializer=initializer,
                                       trainable=False)
        initializer = keras.initializers.get('uniform')
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim, ),
                                    initializer=initializer,
                                    trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        return K.bias_add(output, self.bias, data_format='channels_last')

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.output_dim)


@EX.automain
def main(compile_args, gen_args, eval_args):
    # model = keras.applications.mobilenet.MobileNet()
    model = keras.applications.vgg16.VGG16()

    pprint(model.layers[-4:])

    last = model.layers[-1]
    fc = model.layers[-2]   # the last layer we keep
    output_dim = K.int_shape(model.layers[-1].output)[1]
    my_layer = MyLayer(output_dim)
    x = my_layer(fc.output)

    new_model = keras.Model(inputs=model.input, outputs=x)

    model.compile(**compile_args)
    new_model.compile(**compile_args)
    generator = CropGenerator(**gen_args)
    result = new_model.evaluate_generator(generator, **eval_args)

    print("Evaluation results:")
    print(result)

    print('Done.')
