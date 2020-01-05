#!/bin/bash
# This script is useful for running Python in an activated Conda enviroment via the
# Docker command line.
# That in turn, is useful for IDEs like PyCharm - use this script as the Python interpreter
# /usr/local/bin/run-conda-python.sh

export BOOST_ROOT=$CONDA_PREFIX TANGO_ROOT=$CONDA_PREFIX ZMQ_ROOT=$CONDA_PREFIX OMNI_ROOT=$CONDA_PREFIX
source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)
python "$@"

