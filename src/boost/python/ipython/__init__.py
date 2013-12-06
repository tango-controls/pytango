# -----------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

__all__ = ["init_ipython", "install", "load_ipython_extension",
           "unload_ipython_extension", "load_config", "run", "run_qt"]

from .common import get_python_version
from .common import get_ipython_version
from .common import get_pytango_version


def default_init_ipython(*args, **kwargs):
    print("Unsupported IPython version (%s) for tango profile" \
        % get_ipython_version())
    print("Supported IPython versions are: >= 0.10")
    print("Starting normal IPython console...")

def default_install(*args, **kwargs):
    print("Unsupported IPython version (%s) for tango profile" \
        % get_ipython_version())
    print("Supported IPython versions are: >= 0.10")
    print("Tango extension to IPython will NOT be installed.")

init_ipython = default_init_ipython
install = default_install
is_installed = lambda : False

__run = None
__run_qt = None

ipv = get_ipython_version()

if ipv >= "0.10" and ipv < "0.11":
    from . import ipython_00_10
    init_ipython = ipython_00_10.init_ipython
    install = ipython_00_10.install
    is_installed = ipython_00_10.is_installed
    __run = ipython_00_10.run
    load_config = None
    load_ipython_extension = None
    unload_ipython_extension = None
elif ipv >= "0.11" and ipv < "1.0":
    from . import ipython_00_11
    init_ipython = None
    install = ipython_00_11.install
    is_installed = ipython_00_11.is_installed
    __run = ipython_00_11.run
    load_config = ipython_00_11.load_config
    load_ipython_extension = ipython_00_11.load_ipython_extension
    unload_ipython_extension = ipython_00_11.unload_ipython_extension
elif ipv >= "1.00":
    from . import ipython_10_00
    init_ipython = None
    install = ipython_10_00.install
    is_installed = ipython_10_00.is_installed
    __run = ipython_10_00.run
    load_config = ipython_10_00.load_config
    load_ipython_extension = ipython_10_00.load_ipython_extension
    unload_ipython_extension = ipython_10_00.unload_ipython_extension
    
def run():
    if not is_installed():
        install(verbose=False)
    __run()

def run_qt():
    if not is_installed():
        install(verbose=False)
    __run(qt=True)