"""
A program to investigate existing DNNs (e.g. VGG16), about the
distribution of weights."""

import numpy as np
import matplotlib.pyplot as plt
import keras.applications
from keras.layers.core import Dense


def get_model(name: str):
    """The main procedure to be called."""
    class_dic = {"xception": keras.applications.xception.Xception,
                 "vgg16": keras.applications.vgg16.VGG16,
                 "vgg19": keras.applications.vgg19.VGG19,
                 "resnet50": keras.applications.resnet50.ResNet50,
                 "inceptionv3": keras.applications.inception_v3.InceptionV3,
                 "inceptionresnetv2":
                 keras.applications.inception_resnet_v2.InceptionResNetV2,
                 "mobilenet": keras.applications.mobilenet.MobileNet,
                 "mobilenetv2": keras.applications.mobilenet_v2.MobileNetV2,
                 "densenet121": keras.applications.densenet.DenseNet121,
                 "densenet169": keras.applications.densenet.DenseNet169,
                 "densenet201": keras.applications.densenet.DenseNet201,
                 "nasnetmobile": keras.applications.nasnet.NASNetMobile,
                 "nasnetlarge": keras.applications.nasnet.NASNetLarge}
    return class_dic[name]()


def get_same_type_layers(layers, ltype=Dense):
    """Return only Dense layers (or any other type)."""
    return list(filter(lambda x: type(x) == ltype, layers))


def proc_dense_layer(name, layer, idx, normalised=True):
    """Process a single layer if it is Dense (or other given type)."""
    assert type(layer) == Dense
    weights = layer.get_weights()
    dense = weights[0]
    sorted_dense = np.sort(dense, axis=1)
    norms_dense = np.linalg.norm(dense, axis=1)
    if normalised:
        out = sorted_dense / norms_dense[:, np.newaxis]
    else:
        out = sorted_dense
    plt.plot(out.T)
    plt.title("{}-{} - Normalised: {}".format(name, idx, normalised))
    #plt.show()
    plt.savefig("imgs/{}-{}{}".format(name, idx, "-norm" if normalised else ""))
    plt.close()


def proc_model(name="vgg16"):
    """Process one model based on the model name."""
    model = get_model(name)
    layers = get_same_type_layers(model.layers)
    for idx, layer in enumerate(layers):
        proc_dense_layer(name, layer, idx, True)
        proc_dense_layer(name, layer, idx, False)


def proc_all_models():
    """Process all models."""
    model_names = ["xception", "vgg16", "vgg19", "resnet50", "inceptionv3",
                   "inceptionresnetv2", "mobilenet", "mobilenetv2",
                   "densenet121", "densenet169", "densenet201", "nasnetmobile",
                   "nasnetlarge"]
    for name in model_names:
        print("{} - {}/{}".format(name, model_names.index(name), len(model_names)))
        proc_model(name)


def main():
    proc_all_models()


if __name__ == "__main__":
    print("CLI run")
    main()
else:
    main()
