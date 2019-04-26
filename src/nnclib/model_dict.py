import keras.applications as Kapp

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
