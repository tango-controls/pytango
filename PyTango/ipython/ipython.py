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

__all__ = ["init_ipython", "install"]

import os

try:
    import IPython
    ipython = IPython
except:
    ipython = None

def get_ipython_version():
    """Returns the current IPython version"""
    if ipython is None:return None
    v = None
    try:
        try:
            v = ipython.Release.version
        except Exception:
            try:
                v = ipython.release.version
            except Exception:
                pass
    except Exception:
        pass
    return v

def get_ipython_version_list():
    ipv_str = get_ipython_version()

    if ipv_str is None:
        ipv = [0, 0]
    else:
        ipv = []
        for i in ipv_str.split(".")[:2]:
            try:
                i = int(i)
            except:
                i = 0
            ipv.append(i)
    return ipv

def default_init_ipython(ip, store=True, pytango=True, colors=True,
                         console=True, magic=True):
    print "Unsupported IPython version (%s) for spock profile" \
        % get_ipython_version()
    print "Supported IPython versions are: 0.10"
    print "Starting normal IPython console..."

def default_install(ipydir=None, verbose=True):
    print "Unsupported IPython version (%s) for spock profile" \
        % get_ipython_version()
    print "Supported IPython versions are: 0.10"
    print "Tango extension to IPyhon will NOT be installed."
    
def __define():
    ipv = get_ipython_version_list()
    ret = default_init_ipython, default_install
    if ipv >= [0, 10] and ipv < [0, 11]:
        import ipython_00_10
        ret = ipython_00_10.init_ipython, ipython_00_10.install
    elif ipv >= [0, 11] and ipv <= [0, 12]:
        import ipython_00_11
        ret = ipython_00_11.init_ipython, ipython_00_11.install        
    return ret
    
init_ipython, install = __define()

