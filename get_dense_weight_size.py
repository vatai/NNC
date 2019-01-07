"""This program outputs the size of the dense layers in each keras
pretrained model."""
import pickle
import pprint
import keras.applications as Kapp
from keras.layers.core import Dense


def get_same_type_layers(layers, ltype=Dense):
    """Return only Dense layers (or any other type)."""
    return list(filter(lambda x: isinstance(x, ltype), layers))


def proc_model(name):
    """Process one model based on the model name."""
    # because of sacred:
    # pylint: disable=no-value-for-parameter
    model_dic = {"xception":
                 Kapp.xception.Xception,
                 "vgg16":
                 Kapp.vgg16.VGG16,
                 "vgg19":
                 Kapp.vgg19.VGG19,
                 "resnet50":
                 Kapp.resnet50.ResNet50,
                 "inceptionv3":
                 Kapp.inception_v3.InceptionV3,
                 "inceptionresnetv2":
                 Kapp.inception_resnet_v2.InceptionResNetV2,
                 "mobilenet":
                 Kapp.mobilenet.MobileNet,
                 "mobilenetv2":
                 Kapp.mobilenet_v2.MobileNetV2,
                 "densenet121":
                 Kapp.densenet.DenseNet121,
                 "densenet169":
                 Kapp.densenet.DenseNet169,
                 "densenet201":
                 Kapp.densenet.DenseNet201,
                 "nasnetmobile":
                 Kapp.nasnet.NASNetMobile,
                 "nasnetlarge":
                 Kapp.nasnet.NASNetLarge}
    model_cls = model_dic[name]
    model = model_cls()
    layers = get_same_type_layers(model.layers)
    result = []
    for layer in layers:
        weights = layer.get_weights()
        shape = weights[0].shape
        result.append(shape)
    return result


def proc_all_models():
    """Process all models."""
    model_names = ["xception", "vgg16", "vgg19", "resnet50",
                   "inceptionv3", "inceptionresnetv2", "mobilenet",
                   "mobilenetv2", "densenet121", "densenet169",
                   "densenet201", "nasnetmobile", "nasnetlarge"]
    result = {}
    for index, name in enumerate(model_names):
        print(">>>>>> {} - {}/{}".format(name, index + 1, len(model_names)))
        result[name] = proc_model(name)
    return result


result = proc_all_models()
pickle.dump(result, open("weights.pickl", 'bw'))
pprint.pprint(result)
