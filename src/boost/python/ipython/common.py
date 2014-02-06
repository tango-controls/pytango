#!/usr/bin/env python

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

"""functions common (hopefully) to all ipython versions"""

__all__ = ["get_python_version",
           "get_ipython_version",
           "get_pytango_version"]

import sys
from distutils.version import StrictVersion

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
# Python utilities
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

def get_python_version():
    return StrictVersion('.'.join(map(str, sys.version_info[:3])))

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
# IPython utilities
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

def get_ipython_version():
    """Returns the current IPython version"""
    import IPython
    v = None
    if hasattr(IPython, "version_info"):
        v = '.'.join(map(str, IPython.version_info[:3]))
    else:
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
    return StrictVersion(v)

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
# PyTango utilities
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

def get_pytango_version():
    try:
        import PyTango
    except:
        return
    try:
        v = PyTango.Release.version
    except:
        v = '0.0.0'
    return StrictVersion(v)
