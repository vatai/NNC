"""Temporary script which will do basic GPU operator testing by only
loading the operator.  This is important for seeing if the custom ops
can be used at all.  This should be possible from the singularity
container, and it will probably not be possible from the host.

"""


if __name__ == '__main__':
    import numpy as np
    import tensorflow


    # Load module from ``.so``.
    gpu_module = tensorflow.load_op_library('../../custom-op-dir/_time_two_ops.so')

    # Basically a random input batch.
    inputs = np.array([[ 0.71238331,  1.38283577,  0.37231196],
                       [ 0.08651744, -1.05230163,  1.00697538],
                       [-1.56814483, -0.30826115, -0.13245332],
                       [-0.55250484, -1.53117801,  1.65068967]],
                      dtype=np.float32)
    # Tensor version of the variable.
    inputs_tensor = tensorflow.convert_to_tensor(inputs)


    output = gpu_module.time_two(inputs_tensor)

    print("inputs")
    print(inputs)

    print('output')
    with tensorflow.Session():
        out = tensorflow.Tensor.eval(output)
        print(out)
