# ------------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

from . import release 
__author__ = release.Release.author_lines
__version_info__ = release.Release.version_info
__version__ = release.Release.version
__version_long__ = release.Release.version_long
__version_number__ = release.Release.version_number
__version_description__ = release.Release.version_description
__doc__ = release.Release.long_description
del release

from . import __init_tango
from . import Tango
__init_tango.init(Tango, None)
del __init_tango
del Tango

from .Tango import *

