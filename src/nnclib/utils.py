"""Utility functions for various purposes."""


import keras.applications as Kapp


def get_results_dir(file, base="."):
    """
    Generate a string for the file observer.
    """
    import os
    import datetime
    full_name = os.path.basename(file)
    name = os.path.splitext(full_name)[0]
    now = datetime.datetime.now()
    results_dir = 'results/' + name + '/' + now.strftime('%Y%m%d/%H%M%S')
    results_dir += '-' + str(os.getpid()) + '_' + os.uname()[1]
    return os.path.join(base, results_dir)


def get_epsilons(base=".", sort=True):
    """
    Returns an iterable, based on the files in the base directory
    containing the exponents.
    """
    from glob import glob

    files = glob('eval_*.json')
    if not files:
        print("No eval_*.json files.")
        raise UserWarning

    # This should be basically returned
    epsilons = map(lambda t: float(t[t.find('eps') + 3: -5]), files)
    if sort:
        epsilons = sorted(epsilons)
    return epsilons


def reshape_weights(weights):
    """
    Takes a :math:`d_1 \\times d_2 \\times \\ldots \\times d_{n-1}
    \\times d_n` dimensional tensor, and reshapes it to a :math:`d_1
    \\cdots d_{n-2} \\cdot d_n \\times d_{n-1}` dimensional matrix.
    """ 
    import numpy as np
    shape = np.shape(weights)  # old shape
    # calculate new shape and reshape weights
    height = shape[-2]
    width = shape[-1]
    for dim in shape[:-2]:
        width *= dim
    new_shape = (height, width)
    weights = np.reshape(weights, new_shape)
    return weights


def sum_weights(pairs):
    """
    Calculate the sum of weights from a list of pairs.  This function
    is used to summarise the compression of weights.  It process the
    output of TODO.
    """
    total = 0
    for rows, cols in pairs:
        total += rows * cols
    return total


model_dict = {"xception":
              (Kapp.xception.Xception,
               {'preproc': Kapp.xception.preprocess_input,
                'target_size': 299}),
              "vgg16":
              (Kapp.vgg16.VGG16,
               {'preproc': Kapp.vgg16.preprocess_input,
                'target_size': 224}),
              "vgg19":
              (Kapp.vgg19.VGG19,
               {'preproc': Kapp.vgg19.preprocess_input,
                'target_size': 224}),
              "resnet50":
              (Kapp.resnet50.ResNet50,
               {'preproc': Kapp.resnet50.preprocess_input,
                'target_size': 224}),
              "inceptionv3":
              (Kapp.inception_v3.InceptionV3,
               {'preproc': Kapp.inception_v3.preprocess_input,
                'target_size': 299}),
              "inceptionresnetv2":
              (Kapp.inception_resnet_v2.InceptionResNetV2,
               {'preproc': Kapp.inception_resnet_v2.preprocess_input,
                'target_size': 299}),
              "mobilenet":
              (Kapp.mobilenet.MobileNet,
               {'preproc': Kapp.mobilenet.preprocess_input,
                'target_size': 224}),
              # "mobilenetv2":
              # (Kapp.mobilenet_v2.MobileNetV2,
              #  {'preproc': Kapp.mobilenet_v2.preprocess_input,
              #   'target_size': 224}),
              "densenet121":
              (Kapp.densenet.DenseNet121,
               {'preproc': Kapp.densenet.preprocess_input,
                'target_size': 224}),
              "densenet169":
              (Kapp.densenet.DenseNet169,
               {'preproc': Kapp.densenet.preprocess_input,
                'target_size': 224}),
              "densenet201":
              (Kapp.densenet.DenseNet201,
               {'preproc': Kapp.densenet.preprocess_input,
                'target_size': 224}),
              "nasnetmobile":
              (Kapp.nasnet.NASNetMobile,
               {'preproc': Kapp.nasnet.preprocess_input,
                'target_size': 224}),
              "nasnetlarge":
              (Kapp.nasnet.NASNetLarge,
               {'preproc': Kapp.nasnet.preprocess_input,
                'target_size': 331})}
"""Model dictionary.  The keys of ``model_dict`` are strings representing the names of pretrained models found in Keras.  The corresponding value of the dictionary, is a tuple, containing the class of the model, and a dictionary containing the information needed for preprocessing, namely the ``preprocess_input`` function, and the target size for resizing the input images."""
