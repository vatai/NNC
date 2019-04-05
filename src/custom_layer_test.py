# flake8: noqa: F841
"""This is `Sacred <https://github.com/IDSIA/sacred>`_ experiment to
test the custom Tensorflow operator ``unsort``, by applying it as part
of a custom layer.

A pretrained Keras model is loaded (VGG16 by default, but check the
first lines of the source of :py:func:`main`).  A new model is
created, with the last layer (which should be a dense layer) swapped
out with the custom layer which is being tested in this experiment.
Finally both the original and the modified models are evaluated, and
the two results are displayed for comparison.

About implementation details see the :py:func:`main` function
documentation and source.

Args:

    various: This experiment is similar to :py:mod:`combined`.  See
        the experiment parameters there.

"""

import os
from pprint import pprint
import numpy as np
import sacred
import keras
import keras.initializers
import keras.backend as K
from nnclib.generators import CropGenerator

EX = sacred.Experiment()

@EX.config
def config():
    #pylint: disable=unused-variable
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
    seed = 42


class CompressedPrototype(keras.layers.Layer):
    """A (prototype) implementation of a custom, compressed dense layer.

    The layer implements a simple :math:`x \\mapsto xA + b` linear
    transformation.  The ``kernel`` matrix :math:`A` is compressed
    while the bias :math:`b` is implemented as usual.

    In this implementation, the ``kernel`` matrix is stored as a
    matrix of ``indices`` and a vector which is the per column
    ``mean`` of the rows of (the transpose of) the ``kernel`` matrix
    with the columns sorted.  If ``kernel`` has shape ``(in_dim,
    out_dim)``, then ``indices`` have the same shape, and ``mean`` has
    ``(in_dim,)`` shape.

    Attributes:

        indices (tensor with dtype='int32'): A tensor of indices to
            "unsort" the :py:attr:`mean` into an approximation of the
            original ``kernel``.  These indices can be obtained by
            doing an ``argsort`` on the ``kernel`` matrix on
            ``axis=0`` (sorting each column), and then applying the
            same kind of ``argsort`` on the result (the second
            ``argsort`` gives the unsorting indices).

            .. note:: The ``dtype`` will probably change in the future
                to ``int16``.

        mean (tensor): Mean of the rows of the sorted ``kernel``
            matrix.  The assumption of the whole project is, that each
            column has weights from the same distribution.

        bias (tensor): The bias is no different than the usual dense
            layer bias.

    Args:

        output_dim (int): The size of the output vectors.

    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CompressedPrototype, self).__init__(**kwargs)

    def build(self, input_shape):
        """Creates the layer weights: ``indices``, ``mean`` and ``bias``.

        .. seealso:: Obtained using :py:func:`get_new_weights`.

        Args:

            input_shape (shape): the shape of the input.  For example:
                ``(batch_size, input_dim)``.
        """
        # indices
        initializer = lambda t: K.variable(np.zeros(shape=t),
                                           dtype='int32')
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
                                    trainable=False)

        super(CompressedPrototype, self).build(input_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """Load and apply the custom ``unsort`` op. First, the custom op is
        applied to the ``inputs``, and the ``indices`` and ``mean``
        weights of the layer.  Then the ``bias`` is added as usual.

        Args:

            inputs (tensor): a tensor containing the inputs as row
                vectors with shape ``(batch_size, input_dim)``.

        Returns:

            tensor: Tensor with the applied ops with shape
            ``(batch_size, output_dim)``.

        """
        # Load and apply the custom Op unsort.
        import tensorflow
        so_name = './src/unsortOp/unsort_ops.so'
        unsort_module = tensorflow.load_op_library(so_name)
        output = unsort_module.unsort(inputs, self.indices, self.mean)
        return K.bias_add(output, self.bias, data_format='channels_last')

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer, which is ``(batch_size,
        output_dim)``.
        
        Args:

            input_shape (tuple): shape of input tensor ``(batch_size,
                input_dim)``.

        Returns:

            tuple: A tuple with the shape of the output tensor
            ``(batch_size, output_dim)``.

        """
        batch_size = input_shape[0]
        return (batch_size, self.output_dim)


def get_new_weights(weights):
    """Used to obtain the weights for the *compressed* representation of
    :py:class:`CompressedPrototype`.

    Args:

        weights (tuple): The weights of a dense layer, containing:
            ``kernel`` with shape ``(in_dim, out_dim)``; ``bias`` with
            shape ``(out_dim, )``.

    Returns: 

        list: ``[unsort_indices, mean, bias]`` where ``bias`` is the
        original bias, and the ``kernel`` can be approximated using
        ``unsort_indices`` and ``mean``.

    """
    kernel, bias = weights
    indices = np.argsort(kernel, axis=0)

    sorted_kernel = np.take_along_axis(kernel, indices, 0)
    mean = np.mean(sorted_kernel, axis=1)

    # We need to return the "unsorting" indices.
    return [np.argsort(indices, axis=0), mean, bias]


@EX.automain
def main(compile_args, gen_args, eval_args):
    """This is the the entry point (i.e. ``automain``) of the experiment
    function of the experiment.  See :py:mod:`custom_layer_test` for
    details.

    Swapping out (or transplanting) a layer of a pretrained model is
    discussed `here
    <https://github.com/keras-team/keras/issues/3465>`_: the
    `functional API <https://keras.io/models/model/>`_ is used.  The
    :py:class:`CompressedPrototype` is applied to the output of the
    layer before the last (i.e. the last layers input is redirected to
    the new layer).  Output of this layer is used as the output of a
    ``new_model``. After output/input redirection, the model has to be
    created and compiled too.
    
    Evaluation is done using the common ``evaluate_generator``.

    """
    model = keras.applications.vgg16.VGG16()
    # model = keras.applications.resnet50.ResNet50()
    model.compile(**compile_args)

    last = model.layers[-1]
    output_dim = K.int_shape(last.output)[1]

    my_layer = CompressedPrototype(output_dim)
    new_model = keras.Model(inputs=model.input,
                            outputs=my_layer(model.layers[-2].output))
    new_weights = get_new_weights(last.get_weights())
    new_model.layers[-1].set_weights(new_weights)
    new_model.compile(**compile_args)

    generator = CropGenerator(**gen_args)

    result = model.evaluate_generator(generator, **eval_args)
    print("Evaluation results:")
    print(result)

    new_result = new_model.evaluate_generator(generator, **eval_args)
    print("NEW Evaluation results:")
    print(new_result)

    print('Done.')
