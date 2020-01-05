# Partly based on https://medium.com/@chadlagore/conda-environments-with-docker-82cdc9d25754
# Source for base image:  https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile
FROM continuumio/miniconda3

ARG PYTHON_VERSION=3.7
ARG TANGO_VERSION=9.3.2alpha.1
ADD environment-py${PYTHON_VERSION}-tango${TANGO_VERSION}.yml /tmp/environment.yml

RUN conda update -n base -c defaults conda
RUN conda env create -f /tmp/environment.yml
# Pull the environment name out of the environment.yml, and set environment vars for PyTango build
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc \
    && echo "export BOOST_ROOT=\$CONDA_PREFIX TANGO_ROOT=\$CONDA_PREFIX ZMQ_ROOT=\$CONDA_PREFIX OMNI_ROOT=\$CONDA_PREFIX" >> ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

# additional requirements for building PyTango
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get -y install --no-install-recommends pkg-config \
    # and clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# copy the Python runner utility script
ADD run-conda-python.sh /usr/local/bin/run-conda-python.sh
