#############################################################################
##
## This file is part of PyTango, a python binding for Tango
##
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## (copyleft) CELLS / ALBA Synchrotron, Bellaterra, Spain
##
## This is free software; you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This software is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with this program; if not, see <http://www.gnu.org/licenses/>.
###########################################################################

from PyTango import *

import PyTango
PyTango.EventType.CHANGE = PyTango.EventType.CHANGE_EVENT
PyTango.EventType.QUALITY = PyTango.EventType.QUALITY_EVENT
PyTango.EventType.PERIODIC = PyTango.EventType.PERIODIC_EVENT
PyTango.EventType.ARCHIVE = PyTango.EventType.ARCHIVE_EVENT
PyTango.EventType.USER = PyTango.EventType.USER_EVENT
PyTango.EventType.DATA_READY = PyTango.EventType.DATA_READY_EVENT
PyTango.EventType.ATTR_CONF = PyTango.EventType.ATTR_CONF_EVENT

PyUtil = PyTango.Util
PyDeviceClass = PyTango.DeviceClass

set_attribute_value = PyTango.Attribute.set_value
set_attribute_value_date_quality = PyTango.Attribute.set_value_date_quality

class AttributeValue(PyTango.DeviceAttribute):
    pass

__original_DeviceProxy = PyTango.DeviceProxy

class DeviceProxy3(__original_DeviceProxy):
    defaultCommandExtractAs = ExtractAs.PyTango3

    def __init__(self, *args, **kwds):
        self.__init_state_status()
        super(DeviceProxy3, self).__init__(*args, **kwds)

    def __init_state_status(self):
        if hasattr(self, "State"):
            if callable(getattr(self, "State")):
                self.dev_state = self.State
        if hasattr(self, "Status"):
            if callable(getattr(self, "Status")):
                self.dev_status = self.Status

    def write_attribute(self, attr_name, value=None):
        if isinstance(attr_name, PyTango.DeviceAttribute):
            if value is not None:
                raise AttributeError('Using DeviceAttribute as attribute, only one parameter is expected.')
            da = attr_name
            attr_name = da.name
            value = da.value
        return super(DeviceProxy3, self).write_attribute(attr_name, value)

    def read_attribute(self, attr_name, extract_as=DeviceAttribute.ExtractAs.PyTango3):
        return super(DeviceProxy3, self).read_attribute(attr_name, extract_as)

    def read_attribute_as_str(self, attr_name):
        """
            read_attribute_as_str( (DeviceProxy)self, (str)attr_name ) -> DeviceAttribute :
                Read a single attribute.
            Parameters :
                    - attr_name  : A string, the name of the attribute to read.
            Return     : a PyTango.DeviceAttribute. It's "value" field will contain
                        a string representing the raw data sent by Tango.

            Throws     : ConnectionFailed, CommunicationFailed, DevFailed from Device
        """
        return super(DeviceProxy3, self).read_attribute(attr_name, DeviceAttribute.ExtractAs.String)


    def read_attributes(self, attr_names, extract_as=DeviceAttribute.ExtractAs.PyTango3):
        return super(DeviceProxy3, self).read_attributes(attr_names, extract_as)

    def read_attributes_as_str(self, attr_names):
        """
            read_attributes( (DeviceProxy)self, (object)attr_names, (ExtractAs)extract_as) -> object :
                Read the list of specified attributes.
            Parameters :
                    - attr_names : A list of attributes to read. It should
                                be a StdStringVector or a sequence of str.
            Return     : a list of PyTango.DeviceAttribute. The "value" field
                        is just a string representing the raw data.

            Throws     : ConnectionFailed, CommunicationFailed, DevFailed from device
        """
        return super(DeviceProxy3, self).read_attribute(attr_names, DeviceAttribute.ExtractAs.String)



    def write_read_attribute(self, attr_name, value, extract_as=DeviceAttribute.ExtractAs.PyTango3):
        return super(DeviceProxy3, self).write_read_attribute(attr_name, extract_as)

    def write_read_attribute_as_str(self, args):
        """
            write_read_attribute( (DeviceProxy)self, (str)attr_name, (object)values, (ExtractAs)xs) -> object :
                Write then read a single attribute in a single network call. By
                default (serialisation by device), the execution of this call in
                the server can't be interrupted by other clients.
            Parameters : see write_attribute(attr_name, value)
            Return     : A PyTango.DeviceAttribute object. See read_attribute_as_str()

            Throws     : ConnectionFailed, CommunicationFailed, DeviceUnlocked, DevFailed from device, WrongData
            New in PyTango 7.0.0
        """
        return super(DeviceProxy3, self).write_read_attribute(attr_name, DeviceAttribute.ExtractAs.String)


    def read_attributes_reply(self, idx, timeout=None, extract_as=DeviceAttribute.ExtractAs.PyTango3):
        if timeout is None:
            return super(DeviceProxy3, self).read_attributes_reply(idx, extract_as=extract_as)
        else:
            return super(DeviceProxy3, self).read_attributes_reply(idx, timeout, extract_as)

    def read_attribute_reply(self, idx, timeout=None, extract_as=DeviceAttribute.ExtractAs.PyTango3):
        return self.read_attributes_reply(idx, timeout, extract_as)[0]

    def read_attributes_reply_as_str(self, idx, timeout=None):
        """
            See read_attributes_reply().
            The result is given as in read_attributes_as_str().
            New in PyTango 7.0.0
        """
        return self.read_attributes_reply(idx, timeout, extract_as=DeviceAttribute.ExtractAs.String)

    def read_attribute_reply_as_str(self, idx, timeout=None):
        """
            New in PyTango 7.0.0
        """
        return self.read_attributes_reply_as_str(idx, timeout)[0]


def __copy_doc(fnname):
    getattr(DeviceProxy3, fnname).im_func.__doc__ = getattr(DeviceProxy3.__base__, fnname).im_func.__doc__

__copy_doc('read_attribute')
__copy_doc('read_attributes')
__copy_doc('read_attribute_reply')
__copy_doc('read_attributes_reply')

DeviceProxy = DeviceProxy3
