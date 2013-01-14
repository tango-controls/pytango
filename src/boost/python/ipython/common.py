#!/usr/bin/env python

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

"""functions common (hopefully) to all ipython versions"""

__all__ = ["translate_version_str2int", "translate_version_str2list",
           "get_python_version", "get_python_version_number",
           "get_ipython_version", "get_ipython_version_list",
           "get_ipython_version_number",
           "get_pytango_version", "get_pytango_version_number"]

import sys
import math

def translate_version_str2int(version_str):
    """Translates a version string in format 'x[.y[.z[...]]]' into a 000000 number"""
    
    parts = version_str.split('.')
    i, v, l = 0, 0, len(parts)
    if not l:
        return v
    while i<3:
        try:
            v += int(parts[i])*int(math.pow(10,(2-i)*2))
            l -= 1
            i += 1
        except ValueError:
            return v
        if not l: return v
    return v
    
    try:
        v += 10000*int(parts[0])
        l -= 1
    except ValueError:
        return v
    if not l: return v
    
    try:
        v += 100*int(parts[1])
        l -= 1
    except ValueError:
        return v
    if not l: return v

    try:
        v += int(parts[0])
        l -= 1
    except ValueError:
        return v
    if not l: return v

def translate_version_str2list(version_str):
    """Translates a version string in format 'x[.y[.z[...]]]' into a list of
    numbers"""
    if version_str is None:
        ver = [0, 0]
    else:
        ver = []
        for i in version_str.split(".")[:2]:
            try:
                i = int(i)
            except:
                i = 0
            ver.append(i)
    return ver

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
# Python utilities
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

def get_python_version():
    return '.'.join(map(str, sys.version_info[:3]))

def get_python_version_number():
    pyver_str = get_python_version()
    return translate_version_str2int(pyver_str)

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
# IPython utilities
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

def get_ipython_version():
    """Returns the current IPython version"""
    import IPython
    v = None
    try:
        try:
            v = IPython.Release.version
        except:
            try:
                v = IPython.release.version
            except:
                pass
    except:
        pass
    return v

def get_ipython_version_list():
    ipv_str = get_ipython_version()
    return translate_version_str2list(ipv_str)

def get_ipython_version_number():
    """Returns the current IPython version number"""
    ipyver_str = get_ipython_version()
    if ipyver_str is None: return None
    return translate_version_str2int(ipyver_str)

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
# PyTango utilities
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

def get_pytango_version():
    try:
        import PyTango
    except:
        return
    try:
        return PyTango.Release.version
    except:
        return '0.0.0'

def get_pytango_version_number():
    tgver_str = get_pytango_version()
    if tgver_str is None: return None
    return translate_version_str2int(tgver_str)