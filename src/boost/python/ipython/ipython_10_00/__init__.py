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

__all__ = ['load_ipython_extension', 'unload_ipython_extension', 'load_config',
           'run', 'install', 'is_installed']

from .ipython_10_00 import load_ipython_extension, unload_ipython_extension, \
    load_config, run
from .ipy_install import install, is_installed

