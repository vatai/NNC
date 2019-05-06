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
