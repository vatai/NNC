"""See the experiments documentation for more details."""

from keras.applications import InceptionV3
from keras.datasets import cifar10
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical


def main(batch_size=32, epochs=1):
    """Simple evaluation of any dataset."""

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_categories = 10

    print(x_train[0].shape)
    input_tensor = Input(shape=x_train[0].shape)
    base_model = InceptionV3(input_tensor=input_tensor, include_top=False)
    # base_model = InceptionResNetV2(include_top=False)
    x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_categories, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile('sgd', 'categorical_crossentropy')

    y_train = to_categorical(y_train, num_categories)
    y_test = to_categorical(y_test, num_categories)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

    return base_model


main()
