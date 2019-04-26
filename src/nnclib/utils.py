"""Utility functions for various purposes."""



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

