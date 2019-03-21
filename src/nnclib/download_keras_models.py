"""Download each model by instantiating it."""

import keras.applications


if __name__ == '__main__':
    print("Started: 1/13")
    keras.applications.xception.Xception()
    print("Started: 2/13")
    keras.applications.vgg16.VGG16()
    print("Started: 3/13")
    keras.applications.vgg19.VGG19()
    print("Started: 4/13")
    keras.applications.resnet50.ResNet50()
    print("Started: 5/13")
    keras.applications.inception_v3.InceptionV3()
    print("Started: 6/13")
    keras.applications.inception_resnet_v2.InceptionResNetV2()
    print("Started: 7/13")
    keras.applications.mobilenet.MobileNet()
    print("Started: 8/13")
    keras.applications.mobilenet_v2.MobileNetV2()
    print("Started: 9/13")
    keras.applications.densenet.DenseNet121()
    print("Started: 10/13")
    keras.applications.densenet.DenseNet169()
    print("Started: 11/13")
    keras.applications.densenet.DenseNet201()
    print("Started: 12/13")
    keras.applications.nasnet.NASNetMobile()
    print("Started: 13/13")
    keras.applications.nasnet.NASNetLarge()
    print("Done.")
