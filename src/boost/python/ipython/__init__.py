################################################################################
##
## This file is part of PyTango, a python binding for Tango
## 
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
## 
## PyTango is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## PyTango is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################

__all__ = ["init_ipython", "install", "load_ipython_extension",
           "unload_ipython_extension", "load_config"]

from .common import get_python_version, get_python_version_number, \
    get_ipython_version, get_ipython_version_list, \
    get_ipython_version_number, get_pytango_version, get_pytango_version_number

def default_init_ipython(ip, store=True, pytango=True, colors=True,
                         console=True, magic=True):
    print("Unsupported IPython version (%s) for tango profile" \
        % get_ipython_version())
    print("Supported IPython versions are: >= 0.10")
    print("Starting normal IPython console...")

def default_install(ipydir=None, verbose=True):
    print("Unsupported IPython version (%s) for tango profile" \
        % get_ipython_version())
    print("Supported IPython versions are: >= 0.10")
    print("Tango extension to IPython will NOT be installed.")

init_ipython = default_init_ipython
install = default_install

ipv = get_ipython_version_list()
if ipv >= [0, 10] and ipv < [0, 11]:
    from . import ipython_00_10
    init_ipython = ipython_00_10.init_ipython
    install = ipython_00_10.install
    is_installed = ipython_00_10.is_installed
    __run = ipython_00_10.run
    load_config = None
    load_ipython_extension = None
    unload_ipython_extension = None
elif ipv >= [0, 11] and ipv < [1, 0]:
    from . import ipython_00_11
    init_ipython = None
    install = ipython_00_11.install
    is_installed = ipython_00_11.is_installed
    __run = ipython_00_11.run
    load_config = ipython_00_11.load_config
    load_ipython_extension = ipython_00_11.load_ipython_extension
    unload_ipython_extension = ipython_00_11.unload_ipython_extension

def run():
    if not is_installed():
        install(verbose=False)
    __run()
