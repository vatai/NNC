#!/bin/sh

# Try to compare different versions of a singularity image.

pip --version
python -c 'import sacred;
import telegram;
print("sacred: {}, telegram: {}".format(
    sacred.__version__,
    telegram.__version__))'

