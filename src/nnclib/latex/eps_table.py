"""Return a latex table with the epsilon values for the models.

Beautiful latex tables:

https://inf.ethz.ch/personal/markusp/teaching/guides/guide-tables.pdf

Notes:

- only conv2d and dense layers

-
"""

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os

import numpy as np


def _get_eps_kernel(file_name):
    weights = np.load(file_name)
    norms = np.linalg.norm(weights, axis=0)
    sorted_weights = np.sort(weights, axis=0) / norms

    mean = np.mean(sorted_weights,
                   axis=1)[:, np.newaxis]
    layer_epsilons = np.abs(sorted_weights - mean)
    key = os.path.basename(file_name)
    return (key, layer_epsilons)


def get_epsilons_dict(model_name,
                      base=os.path.expanduser("~/tmp/nnc_weights")):
    """Collect the needed data from the filesystem in a dictionary.

    Args:

        model_name (str): Search for files starting with `model_name`
            in `base`.

        base (str): A path to the directory to search for numpy arrays
            which are the weights.

    Returns:

        result (dict): A dictionary with a key value pair for each
            file which starting with `model_name` (followed by an
            underscore) and the matrix of "epsilons" for the key's
            value.

    """
    from glob import glob
    from os.path import join

    pattern = join(base, model_name) + "_*"
    file_list = glob(pattern)

    data = {}
    with ProcessPoolExecutor() as executor:
        data.update(executor.map(_get_eps_kernel, file_list))
    return data


def _apply_fn_list(weights, fn_list):
    """Apply `fn_list` (function list) to `weights`.
    Note: the functions are applied from last to first.

    Args:

        weights (list): List of numpy matrices.

        fn_list (list): list of (numpy) functions with `len(fn_list)`
            equal 2 or 3.  As a conveninience if the elements are
            strings, are converted to numpy functions as well (see
            example below).

    Returns:

        float: The result of unctions applied to the weights in revers
            order.

    If `fn_list = [np.min, 'average', "max"], then the return value
    is the minimum of the average of the maximum of weights of columns
    of the weights.

    """

    num_fns = len(fn_list)

    if num_fns == 3:
        weights = list(map(fn_list[2], weights, repeat(0)))
    if num_fns >= 2:
        weights = list(map(fn_list[1], weights))
    weights = list(map(np.ravel, weights))
    weights = np.concatenate(weights)
    return fn_list[0](weights)


def _measure_kernel(model_name, fn_lists):
    data = get_epsilons_dict(model_name)
    data = data.values()
    result = list(map(_apply_fn_list, repeat(data), fn_lists))
    return (model_name, result)


def _measure(model_names, fn_lists):
    """Applies _apply_fn_list() to multiple models."""
    output = {}
    with ProcessPoolExecutor() as executor:
        result = executor.map(_measure_kernel, model_names, repeat(fn_lists))
        output.update(result)
    return output


def _latexify(results, header_list):
    first_line = '\\renewcommand{\\arraystretch}{1.3}\n'
    first_line = ""
    first_line += '\\begin{tabular}'
    pos = 'r{}'.format('c'*len(header_list))
    first_line += '{{@{{}}{}@{{}}}}'.format(pos)
    first_line += '\\toprule'
    first_line += '\n'

    header_list = ['Model'] + list(header_list)
    header = " & ".join(header_list) + '\\\\\\midrule\n'

    body = []
    for key, val in results.items():
        fmt = list(map(" {:.4f} ".format, val))
        line = " & ".join([key] + fmt)
        body.append(line)
    body = "\\\\\n".join(body) + "\\\\"

    last_line = '\n'
    last_line += '\\bottomrule'
    last_line += '\\end{tabular}'

    return first_line + header + body + last_line


def eps_table(model_names, measure, force=False):
    """The main entry point of the module.

    Args:

        model_names (list): A list of model names which will be used
            as patterns for reading.
    """
    file_name = 'eps_table.tex'  # TODO(vatai) add to config
    if force or not os.path.exists(file_name):
        fn_lists, header_list = zip(*measure)
        result = _measure(model_names, fn_lists)
        output = _latexify(result, header_list)
        with open(file_name, 'w') as file:
            file.write(output)
        return output
    return open(file_name, 'r').read()


def test_eps_table():
    """Simple test for eps_table()."""
    models = ['xception', 'vgg16', 'vgg19']
    fns = [
        ([np.min], '$\\min$'),
        ([np.max], '$\\max$'),
        ([np.average], '$m_1$'),
        ([np.average, np.min], '$m_2$'),
        ([np.average, np.median, np.median], '$m_3$')
    ]
    epsilon_table = eps_table(models, fns, True)
    print(epsilon_table)


if __name__ == "__main__":
    test_eps_table()
