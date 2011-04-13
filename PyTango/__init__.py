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

"""
This is the main PyTango package file.
Documentation for this package can be found online:

http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
"""

__docformat__ = "restructuredtext"

try:
    import time
    from _PyTango import *
except ImportError, ie:
    if not ie.args[0].count("_PyTango"):
        raise ie
    print 80*"-"
    print ie
    print 80*"-"
    print "Probably your current directory is the PyTango's source installation directory."
    print "You must leave this directory first before using PyTango, otherwise the"
    print "source distribution will conflict with the installed PyTango"
    print 80*"-"
    import sys
    sys.exit(1)

ArgType = _PyTango.CmdArgType

from release import *

__author__ = "\n".join([ "%s <%s>" % x for x in Release.authors.values()])
__version_info__ = Release.version_info
__version__ = Release.version
__version_long__ = Release.version_long
__version_number__ = Release.version_number
__version_description__  = Release.version_description
__doc__ = Release.long_description

import pytango_init
from log4tango import *
from device_server import *
from attribute_proxy import *
from group import *
from pyutil import *
from device_class import *
from globals import *
from utils import *
from tango_numpy import *
from exception import *
from encoded_attribute import *