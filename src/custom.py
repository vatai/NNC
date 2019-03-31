from nnclib.generators import CropGenerator
from pprint import pprint
import sacred
import keras

EX = sacred.Experiment()


@EX.config
def config():
    batch_size = 2


@EX.automain
def main(batch_size):
    # model = keras.applications.mobilenet.MobileNet()
    model = keras.applications.vgg16.VGG16()
    pprint(model.layers[-4:])
    firsthalf = keras.Model(inputs=model.input, outputs=model.layers[-4].output)
    
    pprint(firsthalf.layers[-1:])
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    print('Done.')
