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


def get_exps(base=".", sort=True):
    """
    Returns an iterable, based on the files in the base directory
    containing the exponents.
    """
    from glob import glob

    files = glob('norm-*.json')
    if not files:
        print("No norm-*.json files.")
        raise UserWarning

    # This should be basically returned
    exps = map(lambda t: int(t[5:-5]), files)
    if sort: exps = sorted(exps)
    return exps
