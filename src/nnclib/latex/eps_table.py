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
    """TODO(vatai) proper documentation - just notes now.

    Args:
        model_name (str): Search for files starting with
            `model_name` in `base`.

        base (str): A path to the directory to search for
            numpy arrays which are the weights.

    Returns:
        result (dict): A dictionary with a key value pair
            for each file which starting with `model_name`
            and the matrix of "epsilons" for the key's
            value.
    """
    from glob import glob
    from os.path import join

    pattern = join(base, model_name) + "*"
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
        
        fn_list (list): list of (numpy) functions with 
            `len(fn_list)` equal 2 or 3.  As a 
            conveninience if the elements are strings,
    Returns:
        float: The result of unctions applied to the 
            weights in revers order.
    
    If `fn_list = [np.min, np.average, np.max], then 
    the return vaule is the minimum of the average of 
    the maximum of weights of columns of the weights.
    """

    n = len(fn_list)
    proc_if_str = lambda s: eval('np.' + s) if isinstance(s, str) else s
    fn_list = list(map(proc_if_str, fn_list))
    if n == 3:
        f = lambda w: fn_list[2](w, axis=0)
        weights = list(map(f, weights))
    if n >= 2:
        weights = list(map(fn_list[1], weights))
    weights = list(map(np.ravel, weights))
    weights = np.concatenate(weights)
    return fn_list[0](weights)


def _measure_kernel(model_name, fn_lists):
        data = get_epsilons_dict(model_name)
        data = data.values()
        partial = lambda t: _apply_fn_list(data, t)
        result = list(map(partial, fn_lists))
        return (model_name, result)


def _measure(model_names, fn_lists):
    output = {}
    args = ((n, fn_lists) for n in model_names)
    with ProcessPoolExecutor() as executor:
        output.update(executor.map(_measure_kernel, model_names, repeat(fn_lists)))
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

    body=[]
    for key, val in results.items():
        fmt = list(map(lambda x: " {:.4f} ".format(x), val))
        line = " & ".join([key] + fmt)
        body.append(line)
    body = "\\\\\n".join(body) + "\\\\"

    last_line = '\n'
    last_line += '\\bottomrule'
    last_line += '\\end{tabular}'

    return first_line + header + body + last_line


def eps_table(model_names, measure, force=False):
    file_name = 'eps_table.tex'  # TODO(vatai) add to config
    if force or not os.path.exists(file_name):
        fn_lists, header_list = zip(*measure)
        result = _measure(model_names, fn_lists)
        output = _latexify(result, header_list)
        with open(file_name, 'w') as file:
            file.write(output)
        return output
    else:
        return open(file_name, 'r').read()


if __name__ == "__main__":
    MODELS = ['xception', 'vgg16', 'vgg19']
    FNS = [
        (['min'],     '$\\min$'),
        (['max'],     '$\\max$'),
        (['average'], '$m_1$'),
        (['average', 'min'],              '$m_2$'),
        (['average', 'median', 'median'], '$m_3$')
    ]
    EPSILON_TABLE = eps_table(MODELS, FNS, True)
    print(EPSILON_TABLE)
