"""
This program was used to verify that the gpu is accessible.
"""

if __name__ == '__main__':
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    from keras import backend
    print(backend.tensorflow_backend._get_available_gpus())

