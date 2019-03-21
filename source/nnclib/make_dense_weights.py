"""
Collect the weights of Dense and Conv2D layers, and store them in json
files.
"""

from json import load, dump
import keras.applications
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D

JSON = "dense_weights.json"


def proc(model, name):
    layers = filter(lambda x: isinstance(x, (Dense, Conv2D)), model.layers)
    if layers:
        weights = load(open(JSON, 'r'))
        result = []
        for layer in layers:
            dense, bias = layer.get_weights()
            result.append(dense.shape)
        weights[name] = result
        dump(weights, open(JSON, 'w'))


if __name__ == '__main__':
    dump({}, open(JSON, 'w'))

    print("Started: 1/13")
    model = keras.applications.xception.Xception()
    proc(model, "xception")
    print("Started: 2/13")
    model = keras.applications.vgg16.VGG16()
    proc(model, "vgg16")
    print("Started: 3/13")
    model = keras.applications.vgg19.VGG19()
    proc(model, "vgg19")
    print("Started: 4/13")
    model = keras.applications.resnet50.ResNet50()
    proc(model, "resnet50")
    print("Started: 5/13")
    model = keras.applications.inception_v3.InceptionV3()
    proc(model, "inceptionv3")
    print("Started: 6/13")
    model = keras.applications.inception_resnet_v2.InceptionResNetV2()
    proc(model, "inceptionresnetv2")
    print("Started: 7/13")
    model = keras.applications.mobilenet.MobileNet()
    proc(model, "mobilenet")
    print("Started: 8/13")
    model = keras.applications.mobilenet_v2.MobileNetV2()
    proc(model, "mobilenetv2")
    print("Started: 9/13")
    model = keras.applications.densenet.DenseNet121()
    proc(model, "densenet121")
    print("Started: 10/13")
    model = keras.applications.densenet.DenseNet169()
    proc(model, "densenet169")
    print("Started: 11/13")
    model = keras.applications.densenet.DenseNet201()
    proc(model, "densenet201")
    print("Started: 12/13")
    model = keras.applications.nasnet.NASNetMobile()
    proc(model, "nasnetmobile")
    print("Started: 13/13")
    model = keras.applications.nasnet.NASNetLarge()
    proc(model, "nasnetlarge")
    print("Done.")
