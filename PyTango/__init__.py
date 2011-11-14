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

from release import Release

__author__ = "\n".join([ "%s <%s>" % x for x in Release.authors.values()])
__version_info__ = Release.version_info
__version__ = Release.version
__version_long__ = Release.version_long
__version_number__ = Release.version_number
__version_description__ = Release.version_description
__doc__ = Release.long_description

import pytango_init
from attr_data import *
from log4tango import *
from device_server import *
from attribute_proxy import *
from group import *
from pyutil import *
from device_class import *
from globals import *
from utils import *
from tango_numpy import *

# The following lines will replace the '*' imports above in the future.
#from attr_data import AttrData
#from log4tango import TangoStream, LogIt, DebugIt, InfoIt, WarnIt, \
#    ErrorIt, FatalIt
#from device_server import ChangeEventProp, PeriodicEventProp, \
#    ArchiveEventProp, AttributeAlarm, EventProperties, AttributeConfig, \
#    AttributeConfig_2, AttributeConfig_3
#from attribute_proxy import AttributeProxy
#from group import Group
#from pyutil import Util
#from device_class import DeviceClass
#from globals import get_class, get_classes, get_cpp_class, get_cpp_classes, \
#    get_constructed_class, get_constructed_classes, class_factory, \
#    delete_class_list, class_list, cpp_class_list, constructed_class
#from utils import is_scalar_type, is_array_type, is_numerical_type, \
#    is_int_type, is_float_type, obj_2_str, seqStr_2_obj, document_method, \
#    document_static_method, document_enum, CaselessList, CaselessDict, \
#    EventCallBack, get_home, from_version_str_to_hex_str, \
#    from_version_str_to_int
#from tango_numpy import NumpyType, numpy_type, numpy_spectrum, numpy_image