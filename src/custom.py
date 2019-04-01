from nnclib.generators import CropGenerator
from pprint import pprint
import os
import sacred
import keras

EX = sacred.Experiment()


class MyLayer(keras.layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim),
                                    initializer='uniform',
                                    trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        return K.add(K.dot(x, self.kernel), self.bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


@EX.config
def config():
    compile_args = {'optimizer': 'RMSprop',
                    'loss': 'categorical_crossentropy',
                    'metrics': ['categorical_accuracy',
                                'top_k_categorical_accuracy']}
    gen_args = {'img_dir': os.path.expanduser("~/tmp/ilsvrc/db"),
                'val_file': os.path.expanduser("~/tmp/ilsvrc/caffe_ilsvrc12/val.txt"),
                'batch_size': 32,
                'fast_mode': 8}
    eval_args = {'max_queue_size': 10,
                 'workers': 4,
                 'use_multiprocessing': True,
                 'verbose': True}
    # proc_args = {'norm': 0, 'epsilon': 0, 'dense_smooth': 0, 'conv_smooth': 0, 'quantization': 0}
    seed = 42


@EX.automain
def main(compile_args, gen_args, eval_args):
    # model = keras.applications.mobilenet.MobileNet()
    model = keras.applications.vgg16.VGG16()

    pprint(model.layers[-4:])
    firsthalf = keras.Model(inputs=model.input, outputs=model.layers[-4].output)
    print(type(model.input))
    pprint(firsthalf.layers[-1:])

    

    model.compile(**compile_args)
    generator = CropGenerator(**gen_args)
    result = model.evaluate_generator(generator, **eval_args)

    print("Evaluation results:")
    print(result)

    print('Done.')
