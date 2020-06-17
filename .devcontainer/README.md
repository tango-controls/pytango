# Docker images for development

## Introduction

Docker containers are useful for developing PyTango locally.  This folder is for that purpose, and
the Docker images provide a similar environment to that used by Travis for the Continuous Integration
tests.  The name of the folder is chosen to match Visual Studio Code's naming convention.
Using some command line overrides, images for various versions of Python and Tango can be built.

## Building the Docker image

Run commands like the following:

```shell script
export PYTHON_VERSION=3.7
export TANGO_VERSION=9.3.2
docker build . -t pytango-dev:py${PYTHON_VERSION}-tango${TANGO_VERSION} --build-arg PYTHON_VERSION --build-arg TANGO_VERSION
```

Note: the Tango version must exist in the channel used, the default is here:
https://anaconda.org/tango-controls/tango/files

## Build, install and test PyTango in a container

Run an instance of the container, volume mounting an external PyTango repo into the container.  For example:
```shell script
docker run -it --rm -v ~/tango-src/pytango:/opt/pytango pytango-dev:py3.7-tango9.3.2 /bin/bash
```

Inside the container:
```shell script
cd /opt/pytango
python setup.py build
python setup.py test
```

## Creating/updating Conda environment files

The `environment-*.yml` files are created and can be updated using the commands below.
It is convenient to do this in a container:
 ```shell script
$ docker run -it --rm -v $PWD:/opt/current continuumio/miniconda3 /bin/bash
```

Then run the commands inside that container - here's an example for a specific version of Python and Tango.
Due to the volume mount above, the last line will output the environment file to your host's current folder.
```shell script
export PYTHON_VERSION=3.7
export TANGO_VERSION=9.3.2
conda create --yes --name env-py${PYTHON_VERSION}-tango${TANGO_VERSION} python=${PYTHON_VERSION}
conda activate env-py${PYTHON_VERSION}-tango${TANGO_VERSION}
conda install --yes boost gxx_linux-64
conda install --yes -c tango-controls tango=${TANGO_VERSION}
conda install --yes pytest pytest-xdist 'gevent != 1.5a1' psutil
conda env export > /opt/current/environment-py${PYTHON_VERSION}-tango${TANGO_VERSION}.yml
```

For Python 2.7, the requirements are slightly different, so replace the line with `pytest` with:
```shell script
conda install --yes trollius futures 'pyparsing < 3' 'pytest < 5' pytest-xdist 'gevent != 1.5a1' psutil enum34
```

Note:  the packages lists were taken from `.travis.yml` and `setup.py` - these will need
to be kept in sync manually.

## Using a container with an IDE

Once the image has been built, it can be used with IDEs like
[PyCharm](https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html#config-docker)
(Professional version only), and
[Visual Studio Code](https://code.visualstudio.com/docs/remote/containers)

### PyCharm:
Add a new interpreter:
- Open the _Add Interpreter..._ dialog
- Select _Docker_
- Pick the image to use, e.g., `pytango-dev:py3.7-tango9.3.2`
- Change the Python interpreter path to `/usr/local/bin/run-conda-python.sh`

Running tests:
- If you want to run all the tests, it will work out the box.
- If you only want to run a subset, the `setup.cfg` file needs to be change temporarily:
  - In the `[tool:pytest]` section, remove the `tests` path from the additional options, to give:
     `addopts = -v --boxed`
  - If the change isn't made you may get errors like:
    ```
    collecting ... collected 0 items
    ERROR: file not found: tests
    ```

### Visual Studio Code
TODO - probably need to add a `devcontainer.json` file...
