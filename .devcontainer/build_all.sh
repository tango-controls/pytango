#!/bin/bash
# Simple script for building Docker images with various versions of Tango and Python
# TODO: use a Makefile and CI instead...

PYTHON_VERSION=2.7
TANGO_VERSION=9.2.5
echo py${PYTHON_VERSION}-tango${TANGO_VERSION}
docker build . -t pytango-dev:py${PYTHON_VERSION}-tango${TANGO_VERSION} --build-arg PYTHON_VERSION --build-arg TANGO_VERSION --no-cache

PYTHON_VERSION=2.7
TANGO_VERSION=9.3.2
echo py${PYTHON_VERSION}-tango${TANGO_VERSION}
docker build . -t pytango-dev:py${PYTHON_VERSION}-tango${TANGO_VERSION} --build-arg PYTHON_VERSION --build-arg TANGO_VERSION --no-cache

PYTHON_VERSION=3.7
TANGO_VERSION=9.2.5
echo py${PYTHON_VERSION}-tango${TANGO_VERSION}
docker build . -t pytango-dev:py${PYTHON_VERSION}-tango${TANGO_VERSION} --build-arg PYTHON_VERSION --build-arg TANGO_VERSION --no-cache

PYTHON_VERSION=3.7
TANGO_VERSION=9.3.2
echo py${PYTHON_VERSION}-tango${TANGO_VERSION}
docker build . -t pytango-dev:py${PYTHON_VERSION}-tango${TANGO_VERSION} --build-arg PYTHON_VERSION --build-arg TANGO_VERSION --no-cache
