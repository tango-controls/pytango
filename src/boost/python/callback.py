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
This is an internal PyTango module.
"""

__all__ = ["callback_init"]

__docformat__ = "restructuredtext"

from ._PyTango import CmdDoneEvent, AttrReadEvent, AttrWrittenEvent

def __init_Callback():
    pass

def __doc_Callback():
    CmdDoneEvent.__doc__ = """
        This class is used to pass data to the callback method in
        asynchronous callback model for command execution.

        It has the following members:
            - device     : (DeviceProxy) The DeviceProxy object on which the call was executed.
            - cmd_name   : (str) The command name
            - argout_raw : (DeviceData) The command argout
            - argout     : The command argout
            - err        : (bool) A boolean flag set to true if the command failed. False otherwise
            - errors     : (sequence<DevError>) The error stack
            - ext        :
    """

    AttrReadEvent.__doc__ = """
        This class is used to pass data to the callback method in
        asynchronous callback model for read_attribute(s) execution.

        It has the following members:
            - device     : (DeviceProxy) The DeviceProxy object on which the call was executed
            - attr_names : (sequence<str>) The attribute name list
            - argout     : (DeviceAttribute) The attribute value
            - err        : (bool) A boolean flag set to true if the command failed. False otherwise
            - errors     : (sequence<DevError>) The error stack
            - ext        :
    """

    AttrWrittenEvent.__doc__ = """
        This class is used to pass data to the callback method in
        asynchronous callback model for write_attribute(s) execution

        It has the following members:
            - device     : (DeviceProxy) The DeviceProxy object on which the call was executed
            - attr_names : (sequence<str>) The attribute name list
            - err        : (bool) A boolean flag set to true if the command failed. False otherwise
            - errors     : (NamedDevFailedList) The error stack
            - ext        :
    """

def callback_init(doc=True):
    __init_Callback()
    if doc:
        __doc_Callback()
