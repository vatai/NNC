"""Utility functions for various purposes."""

import os
import datetime


def get_results_dir():
    """Generate a string for the file observer."""
    ex_name = os.path.basename(__file__).split('.')[0]
    now = datetime.datetime.now()
    results_dir = 'results/' + ex_name + '/' + now.strftime('%Y%m%d/%H%M%S')
    results_dir += '-' + str(os.getpid()) + '_' + os.uname()[1]
    return results_dir
