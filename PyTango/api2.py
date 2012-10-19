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

"""This is a PyTango module which provides a high level device server API."""

from __future__ import with_statement
from __future__ import print_function

__all__ = ["DeviceMeta", "Device", "attribute", "command",
           "device_property", "class_property"]

import inspect
import functools

from ._PyTango import DeviceImpl, Attribute, WAttribute, CmdArgType, \
    AttrDataFormat, AttrWriteType, DispLevel
from .attr_data import AttrData
from .device_class import DeviceClass
from .utils import get_tango_device_classes
from .log4tango import DebugIt

API_VERSION = 2

LatestDeviceImpl = get_tango_device_classes()[-1]

def check_tango_device_klass_attribute_methods(tango_device_klass, attr_data):
    """Checks if the read and write methods have the correct signature. If a 
    read/write method doesn't have a parameter (the traditional Attribute),
    then the method is wrapped into another method to make this work
    
    :param tango_device_klass: a DeviceImpl class
    :type tango_device_klass: class
    :param attr_data: the attribute data information
    :type attr_data: AttrData"""
    
    read_name = attr_data.read_method_name
    read_obj = real_read_obj = getattr(tango_device_klass, read_name)

    # discover the real method because it may be hidden by a tango decorator
    # like PyTango.DebugIt. Unfortunately we cannot detect other decorators yet
    while hasattr(real_read_obj, "_wrapped"):
        real_read_obj = real_read_obj._wrapped
    argspec = inspect.getargspec(real_read_obj)
    if argspec.varargs and len(argspec.varargs):
        return
    nb = len(argspec.args) - 1
    if argspec.defaults:
        nb -= len(argspec.defaults)
    if nb > 0:
        return
    @functools.wraps(read_obj)
    def read_attr(self, attr):
        return read_obj(self)
    setattr(tango_device_klass, read_name, read_attr)
    
def create_tango_deviceclass_klass(tango_device_klass, attrs=None):
    klass_name = tango_device_klass.__name__
    if not issubclass(tango_device_klass, (Device)):
        raise Exception("{0} device must inherit from PyTango.api2.Device".format(klass_name))
    
    if attrs is None:
        attrs = tango_device_klass.__dict__
        
    attr_list = {}
    for attr_name, attr_obj in attrs.items():
        if isinstance(attr_obj, AttrData2):
            attr_obj._set_name(attr_name)
            attr_list[attr_name] = attr_obj
            check_tango_device_klass_attribute_methods(tango_device_klass, attr_obj)
            
    class_property_list = {}
    device_property_list = {}
    cmd_list = {}
    devclass_name = klass_name + "Class"
    devclass_attrs = dict(class_property_list=class_property_list,
                          device_property_list=device_property_list,
                          cmd_list=cmd_list, attr_list=attr_list)
    return type(devclass_name, (DeviceClass,), devclass_attrs)

def init_tango_device_klass(tango_device_klass, attrs=None, tango_class_name=None):
    klass_name = tango_device_klass.__name__
    tango_deviceclass_klass = create_tango_deviceclass_klass(tango_device_klass,
                                                             attrs=attrs)
    if tango_class_name is None:
        tango_klass_name = klass_name
    tango_device_klass._DeviceClass = tango_deviceclass_klass
    tango_device_klass._DeviceClassName = tango_klass_name
    tango_device_klass._api = API_VERSION
    return tango_device_klass

def create_tango_device_klass(name, bases, attrs):
    klass_name = name

    LatestDeviceImplMeta = type(LatestDeviceImpl)
    klass = LatestDeviceImplMeta(klass_name, bases, attrs)
    init_tango_device_klass(klass, attrs)
    return klass
    
def DeviceMeta(name, bases, attrs):
    return create_tango_device_klass(name, bases, attrs)


class Device(LatestDeviceImpl):
    """High level DeviceImpl API"""
    
    def __init__(self, cl, name):
        super(Device, self).__init__(cl, name)
        self.debug_stream("-> __init__()")
        with_exception = True
        try:
            self.init_device()
            with_exception = False
        finally:
            if with_exception:
                debug_msg = "<- __init__() raised exception!"
            else:
                debug_msg = "<- __init__()"
            self.debug_stream(debug_msg)
            
    @DebugIt()
    def init_device(self):
        """Tango init_device method. Default implementation calls
        get_device_properties()"""
        self.get_device_properties()
    
    @DebugIt()
    def always_executed_hook(self):
        """Tango always_executed_hook. Default implementation does nothing"""
        pass


class AttrData2(AttrData):
    """High level AttrData. To be used """
    
    def get_attribute(self, obj):
        return obj.get_device_attr().get_attr_by_name(self.attr_name)
        
    def __get__(self, obj, objtype):
        return self.get_attribute(obj)

    def __set__(self, obj, value):
        is_tuple = isinstance(value, tuple)
        attr = self.get_attribute(obj)
        dtype, fmt = attr.get_data_type(), attr.get_data_format()
        if dtype == CmdArgType.DevEncoded:
            if is_tuple and len(value) == 4:
                attr.set_value_date_quality(*value)
            else:
                attr.set_value(value)
        else:
            if is_tuple:
                if len(value) == 3:
                    if fmt == AttrDataFormat.SCALAR:
                        attr.set_value_date_quality(*value)
                    elif fmt == AttrDataFormat.SPECTRUM:
                        if is_seq(value[0]):
                            attr.set_value_date_quality(*value)
                        else:
                            attr.set_value(value)
                    else:
                        if is_seq(value[0]) and is_seq(value[0][0]):
                            attr.set_value_date_quality(*value)
                        else:
                            attr.set_value(value)
                else:
                    attr.set_value(value)
            else:
                attr.set_value(value)
    
    def __delete__(self, obj):
        obj.remove_attribute(self.attr_name)
    
def attribute(**kwargs):
    return AttrData2.from_dict(kwargs)
    

def cmd():
    pass
    
    
def device_property():
    pass

def class_property():
    pass

