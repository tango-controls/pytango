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

"""High Level API for writing Tango device servers.

.. _pytango-hlapi-datatypes:

.. rubric:: Data types

When declaring attributes, properties or commands, one of the most important
information is the data type. It is given by the keyword argument *dtype*.
In order to provide a more *pythonic* interface, this argument is not restricted
to the :obj:`~PyTango.CmdArgType` options.

For example, to define a *SCALAR* :obj:`~PyTango.CmdArgType.DevLong`
attribute you have several possibilities:

#. :obj:`int`
#. 'int'
#. 'int32'
#. 'integer' 
#. :obj:`PyTango.CmdArgType.DevLong`
#. 'DevLong' 
#. :obj:`numpy.int32`

To define a *SPECTRUM* attribute simply wrap the scalar data type in any
python sequence:

* using a *tuple*: ``(:obj:`int`,)`` or
* using a *list*: ``[:obj:`int`]`` or
* any other sequence type

To define an *IMAGE* attribute simply wrap the scalar data type in any
python sequence of sequences:

* using a *tuple*: ``((:obj:`int`,),)`` or
* using a *list*: ``[[:obj:`int`]]`` or
* any other sequence type

Below is the complete table of equivalences.

========================================  ========================================
 type                                      tango type                             
========================================  ========================================
 ``None``                                  ``DevVoid``
 ``'None'``                                ``DevVoid``
 ``DevVoid``                               ``DevVoid``
 ``'DevVoid'``                             ``DevVoid``

 ``DevState``                              ``DevState``                           
 ``'DevState'``                            ``DevState``                           

 :py:obj:`bool`                            ``DevBoolean``
 ``'bool'``                                ``DevBoolean``
 ``'boolean'``                             ``DevBoolean``
 ``DevBoolean``                            ``DevBoolean``
 ``'DevBoolean'``                          ``DevBoolean``
 :py:obj:`numpy.bool_`                     ``DevBoolean``

 ``'char'``                                ``DevUChar``
 ``'chr'``                                 ``DevUChar``
 ``'byte'``                                ``DevUChar``
 ``chr``                                   ``DevUChar``
 ``DevUChar``                              ``DevUChar``
 ``'DevUChar'``                            ``DevUChar``
 :py:obj:`numpy.uint8`                     ``DevUChar``

 ``'int16'``                               ``DevShort``
 ``DevShort``                              ``DevShort``
 ``'DevShort'``                            ``DevShort``
 :py:obj:`numpy.int16`                     ``DevShort``

 ``'uint16'``                              ``DevUShort``
 ``DevUShort``                             ``DevUShort``
 ``'DevUShort'``                           ``DevUShort``
 :py:obj:`numpy.uint16`                    ``DevUShort``

 :py:obj:`int`                             ``DevLong``
 ``'int'``                                 ``DevLong``
 ``'int32'``                               ``DevLong``
 ``DevLong``                               ``DevLong``
 ``'DevLong'``                             ``DevLong``
 :py:obj:`numpy.int32`                     ``DevLong``

 ``'uint'``                                ``DevULong``
 ``'uint32'``                              ``DevULong``
 ``DevULong``                              ``DevULong``
 ``'DevULong'``                            ``DevULong``
 :py:obj:`numpy.uint32`                    ``DevULong``

 ``'int64'``                               ``DevLong64``
 ``DevLong64``                             ``DevLong64``
 ``'DevLong64'``                           ``DevLong64``
 :py:obj:`numpy.int64`                     ``DevLong64``
 
 ``'uint64'``                              ``DevULong64``
 ``DevULong64``                            ``DevULong64``
 ``'DevULong64'``                          ``DevULong64``
 :py:obj:`numpy.uint64`                    ``DevULong64``

 ``DevInt``                                ``DevInt``                             
 ``'DevInt'``                              ``DevInt``                             
 
 ``'float32'``                             ``DevFloat``
 ``DevFloat``                              ``DevFloat``
 ``'DevFloat'``                            ``DevFloat``
 :py:obj:`numpy.float32`                   ``DevFloat``
 
 :py:obj:`float`                           ``DevDouble``
 ``'double'``                              ``DevDouble``
 ``'float'``                               ``DevDouble``
 ``'float64'``                             ``DevDouble``
 ``DevDouble``                             ``DevDouble``
 ``'DevDouble'``                           ``DevDouble``
 :py:obj:`numpy.float64`                   ``DevDouble``
 
 :py:obj:`str`                             ``DevString``
 ``'str'``                                 ``DevString``
 ``'string'``                              ``DevString``
 ``'text'``                                ``DevString``
 ``DevString``                             ``DevString``
 ``'DevString'``                           ``DevString``
 
 :py:obj:`bytearray`                       ``DevEncoded``
 ``'bytearray'``                           ``DevEncoded``
 ``'bytes'``                               ``DevEncoded``
 ``DevEncoded``                            ``DevEncoded``
 ``'DevEncoded'``                          ``DevEncoded``

 ``DevVarBooleanArray``                    ``DevVarBooleanArray``
 ``'DevVarBooleanArray'``                  ``DevVarBooleanArray``
 
 ``DevVarCharArray``                       ``DevVarCharArray``
 ``'DevVarCharArray'``                     ``DevVarCharArray``
 
 ``DevVarShortArray``                      ``DevVarShortArray``
 ``'DevVarShortArray'``                    ``DevVarShortArray``
 
 ``DevVarLongArray``                       ``DevVarLongArray``
 ``'DevVarLongArray'``                     ``DevVarLongArray``
 
 ``DevVarLong64Array``                     ``DevVarLong64Array``
 ``'DevVarLong64Array'``                   ``DevVarLong64Array``
 
 ``DevVarULong64Array``                    ``DevVarULong64Array``
 ``'DevVarULong64Array'``                  ``DevVarULong64Array``
 
 ``DevVarFloatArray``                      ``DevVarFloatArray``
 ``'DevVarFloatArray'``                    ``DevVarFloatArray``
 
 ``DevVarDoubleArray``                     ``DevVarDoubleArray``
 ``'DevVarDoubleArray'``                   ``DevVarDoubleArray``
 
 ``DevVarUShortArray``                     ``DevVarUShortArray``
 ``'DevVarUShortArray'``                   ``DevVarUShortArray``
 
 ``DevVarULongArray``                      ``DevVarULongArray``
 ``'DevVarULongArray'``                    ``DevVarULongArray``
 
 ``DevVarStringArray``                     ``DevVarStringArray``
 ``'DevVarStringArray'``                   ``DevVarStringArray``
 
 ``DevVarLongStringArray``                 ``DevVarLongStringArray``
 ``'DevVarLongStringArray'``               ``DevVarLongStringArray``
 
 ``DevVarDoubleStringArray``               ``DevVarDoubleStringArray``
 ``'DevVarDoubleStringArray'``             ``DevVarDoubleStringArray``
========================================  ========================================
"""

from __future__ import with_statement
from __future__ import print_function

__all__ = ["DeviceMeta", "Device", "LatestDeviceImpl", "attribute", "command",
           "device_property", "class_property", "server_run"]

import __builtin__
import sys
import inspect
import functools
import traceback

from ._PyTango import DeviceImpl, Attribute, WAttribute, CmdArgType
from ._PyTango import AttrDataFormat, AttrWriteType, DispLevel, constants
from ._PyTango import DevFailed
from .attr_data import AttrData
from .device_class import DeviceClass
from .utils import get_tango_device_classes, is_seq, is_non_str_seq
from .utils import scalar_to_array_type

API_VERSION = 2

LatestDeviceImpl = get_tango_device_classes()[-1]

def __build_to_tango_type():
    ret = \
    {
        int         : CmdArgType.DevLong,
        str         : CmdArgType.DevString,
        bool        : CmdArgType.DevBoolean,
        bytearray   : CmdArgType.DevEncoded,
        float       : CmdArgType.DevDouble,
        chr         : CmdArgType.DevUChar,
        None        : CmdArgType.DevVoid,

        'int'       : CmdArgType.DevLong,
        'int16'     : CmdArgType.DevShort,
        'int32'     : CmdArgType.DevLong,
        'int64'     : CmdArgType.DevLong64,
        'uint'      : CmdArgType.DevULong,
        'uint16'    : CmdArgType.DevUShort,
        'uint32'    : CmdArgType.DevULong,
        'uint64'    : CmdArgType.DevULong64,
        'str'       : CmdArgType.DevString,
        'string'    : CmdArgType.DevString,
        'text'      : CmdArgType.DevString,
        'bool'      : CmdArgType.DevBoolean,
        'boolean'   : CmdArgType.DevBoolean,
        'bytes'     : CmdArgType.DevEncoded,
        'bytearray' : CmdArgType.DevEncoded,
        'float'     : CmdArgType.DevDouble,
        'float32'   : CmdArgType.DevFloat,
        'float64'   : CmdArgType.DevDouble,
        'double'    : CmdArgType.DevDouble,
        'byte'      : CmdArgType.DevUChar,
        'chr'       : CmdArgType.DevUChar,
        'char'      : CmdArgType.DevUChar,
        'None'      : CmdArgType.DevVoid,
    }

    for key in dir(CmdArgType):
        if key.startswith("Dev"):
            value = getattr(CmdArgType, key)
            ret[key] = ret[value] = value

    if constants.NUMPY_SUPPORT:
        import numpy
        FROM_TANGO_TO_NUMPY_TYPE = { \
                   CmdArgType.DevBoolean : numpy.bool8,
                     CmdArgType.DevUChar : numpy.ubyte,
                     CmdArgType.DevShort : numpy.short,
                    CmdArgType.DevUShort : numpy.ushort,
                      CmdArgType.DevLong : numpy.int32,
                     CmdArgType.DevULong : numpy.uint32,
                    CmdArgType.DevLong64 : numpy.int64,
                   CmdArgType.DevULong64 : numpy.uint64,
                    CmdArgType.DevString : numpy.str,
                    CmdArgType.DevDouble : numpy.float64,
                     CmdArgType.DevFloat : numpy.float32,
        }

        for key, value in FROM_TANGO_TO_NUMPY_TYPE.items():
            ret[value] = key
    return ret

TO_TANGO_TYPE = __build_to_tango_type()


def get_tango_type_format(dtype=None, dformat=None):
    if dformat is None:
        dformat = AttrDataFormat.SCALAR
        if is_non_str_seq(dtype):
            dtype = dtype[0]
            dformat = AttrDataFormat.SPECTRUM
            if is_non_str_seq(dtype):
                dtype = dtype[0]
                dformat = AttrDataFormat.IMAGE
    return TO_TANGO_TYPE[dtype], dformat


def from_typeformat_to_type(dtype, dformat):
    if dformat == AttrDataFormat.SCALAR:
        return dtype
    elif dformat == AttrDataFormat.IMAGE:
        raise TypeError("Cannot translate IMAGE to tango type")
    return scalar_to_array_type(dtype)


def set_complex_value(attr, value):
    is_tuple = isinstance(value, tuple)
    dtype, fmt = attr.get_data_type(), attr.get_data_format()
    if dtype == CmdArgType.DevEncoded:
        if is_tuple and len(value) == 4:
            attr.set_value_date_quality(*value)
        elif is_tuple and len(value) == 3 and is_non_str_seq(value[0]):
            attr.set_value_date_quality(value[0][0], value[0][1], *value[1:])
        else:
            attr.set_value(*value)
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


def check_tango_device_klass_attribute_read_method(tango_device_klass, method_name):
    """Checks if method given by it's name for the given DeviceImpl class has
    the correct signature. If a read/write method doesn't have a parameter
    (the traditional Attribute), then the method is wrapped into another method
    which has correct parameter definition to make it work.
    
    :param tango_device_klass: a DeviceImpl class
    :type tango_device_klass: class
    :param method_name: method to be cheched
    :type attr_data: str"""
    read_method = getattr(tango_device_klass, method_name)

    @functools.wraps(read_method)
    def read_attr(self, attr):
        ret = read_method(self)
        if not attr.get_value_flag() and ret is not None:
            set_complex_value(attr, ret)
        return ret
    setattr(tango_device_klass, method_name, read_attr)


def check_tango_device_klass_attribute_write_method(tango_device_klass, method_name):
    """Checks if method given by it's name for the given DeviceImpl class has
    the correct signature. If a read/write method doesn't have a parameter
    (the traditional Attribute), then the method is wrapped into another method
    which has correct parameter definition to make it work.
    
    :param tango_device_klass: a DeviceImpl class
    :type tango_device_klass: class
    :param method_name: method to be cheched
    :type attr_data: str"""
    write_method = getattr(tango_device_klass, method_name)

    @functools.wraps(write_method)
    def write_attr(self, attr):
        value = attr.get_write_value()
        return write_method(self, value)
    setattr(tango_device_klass, method_name, write_attr)


def check_tango_device_klass_attribute_methods(tango_device_klass, attr_data):
    """Checks if the read and write methods have the correct signature. If a 
    read/write method doesn't have a parameter (the traditional Attribute),
    then the method is wrapped into another method to make this work
    
    :param tango_device_klass: a DeviceImpl class
    :type tango_device_klass: class
    :param attr_data: the attribute data information
    :type attr_data: AttrData"""
    if attr_data.attr_write in (AttrWriteType.READ, AttrWriteType.READ_WRITE):
        check_tango_device_klass_attribute_read_method(tango_device_klass, attr_data.read_method_name)
    if attr_data.attr_write in (AttrWriteType.WRITE, AttrWriteType.READ_WRITE):
        check_tango_device_klass_attribute_write_method(tango_device_klass, attr_data.write_method_name)


class _DeviceClass(DeviceClass):

    def __init__(self, name):
        DeviceClass.__init__(self, name)
        self.set_type(name)

    def dyn_attr(self, dev_list):
        """Invoked to create dynamic attributes for the given devices.
        Default implementation calls
        :meth:`TT.initialize_dynamic_attributes` for each device
    
        :param dev_list: list of devices
        :type dev_list: :class:`PyTango.DeviceImpl`"""

        for dev in dev_list:
            init_dyn_attrs = getattr(dev, "initialize_dynamic_attributes", None)
            if init_dyn_attrs and callable(init_dyn_attrs):
                try:
                    init_dyn_attrs()
                except Exception:
                    import traceback
                    dev.warn_stream("Failed to initialize dynamic attributes")
                    dev.debug_stream("Details: " + traceback.format_exc())


def create_tango_deviceclass_klass(tango_device_klass, attrs=None):
    klass_name = tango_device_klass.__name__
    if not issubclass(tango_device_klass, (Device)):
        msg = "{0} device must inherit from PyTango.server.Device".format(klass_name)
        raise Exception(msg)

    if attrs is None:
        attrs = tango_device_klass.__dict__

    attr_list = {}
    class_property_list = {}
    device_property_list = {}
    cmd_list = {}

    for attr_name, attr_obj in attrs.items():
        if isinstance(attr_obj, attribute):
            attr_obj._set_name(attr_name)
            attr_list[attr_name] = attr_obj
            check_tango_device_klass_attribute_methods(tango_device_klass, attr_obj)
        elif isinstance(attr_obj, device_property):
            device_property_list[attr_name] = [attr_obj.dtype, attr_obj.doc, attr_obj.default_value]
        elif isinstance(attr_obj, class_property):
            class_property_list[attr_name] = [attr_obj.dtype, attr_obj.doc, attr_obj.default_value]
        elif inspect.isroutine(attr_obj):
            if hasattr(attr_obj, "__tango_command__"):
                cmd_name, cmd_info = attr_obj.__tango_command__
                cmd_list[cmd_name] = cmd_info

    devclass_name = klass_name + "Class"

    devclass_attrs = dict(class_property_list=class_property_list,
                          device_property_list=device_property_list,
                          cmd_list=cmd_list, attr_list=attr_list)
    return type(devclass_name, (_DeviceClass,), devclass_attrs)


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
    """The :py:data:`metaclass` callable for :class:`Device`. Every subclass of
    :class:`Device` must have associated this metaclass to itself in order to
    work properly (boost-python internal limitation).
    
    Example (python 2.x)::
    
        from PyTango.server import Device, DeviceMeta

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

    Example (python 3.x)::
    
        from PyTango.server import Device, DeviceMeta

        class PowerSupply(Device, metaclass=DeviceMeta):
            pass
    """
    return create_tango_device_klass(name, bases, attrs)


class Device(LatestDeviceImpl):
    """High level DeviceImpl API. All Device specific classes should inherit
    from this class."""

    def __init__(self, cl, name):
        LatestDeviceImpl.__init__(self, cl, name)
        self.init_device()

    def init_device(self):
        """Tango init_device method. Default implementation calls
        :meth:`get_device_properties`"""
        self.get_device_properties()

    def always_executed_hook(self):
        """Tango always_executed_hook. Default implementation does nothing"""
        pass

    def initialize_dynamic_attributes(self):
        """Method executed at initializion phase to create dynamic attributes.
        Default implementation does nothing. Overwrite when necessary."""
        pass


class attribute(AttrData):
    """declares a new tango attribute in a :class:`Device`. To be used like
the python native :obj:`property` function. For example, to declare a
scalar, `PyTango.DevDouble`, read-only attribute called *voltage* in a
*PowerSupply* :class:`Device` do::

    class PowerSupply(Device):
        __metaclass__ = DeviceMeta
        
        voltage = attribute()
        
        def read_voltage(self):
            self.voltage = 1.0

It receives multiple keyword arguments.

===================== ================================ ======================================= =======================================================================================
parameter              type                                       default value                                 description
===================== ================================ ======================================= =======================================================================================
name                   :obj:`str`                       class member name                       alternative attribute name
dtype                  :obj:`object`                    :obj:`~PyTango.CmdArgType.DevDouble`    data type (see :ref:`Data type equivalence <pytango-hlapi-datatypes>`)
dformat                :obj:`~PyTango.AttrDataFormat`   :obj:`~PyTango.AttrDataFormat.SCALAR`   data format
max_dim_x              :obj:`int`                       1                                       maximum size for x dimension (ignored for SCALAR format) 
max_dim_y              :obj:`int`                       0                                       maximum size for y dimension (ignored for SCALAR and SPECTRUM formats) 
display_level          :obj:`~PyTango.DispLevel`        :obj:`~PyTango.DisLevel.OPERATOR`       display level
polling_period         :obj:`int`                       -1                                      polling period
memorized              :obj:`bool`                      False                                   attribute should or not be memorized
hw_memorized           :obj:`bool`                      False                                   write method should be called at startup when restoring memorize value (dangerous!)
access                 :obj:`~PyTango.AttrWriteType`    :obj:`~PyTango.AttrWriteType.READ`      read only/ read write / write only access
fget (or fread)        :obj:`str` or :obj:`callable`    'read_<attr_name>'                      read method name or method object
fset (or fwrite)       :obj:`str` or :obj:`callable`    'write_<attr_name>'                     write method name or method object
is_allowed             :obj:`str` or :obj:`callable`    'is_<attr_name>_allowed'                is allowed method name or method object
label                  :obj:`str`                       '<attr_name>'                           attribute label
doc (or description)   :obj:`str`                       ''                                      attribute description
unit                   :obj:`str`                       ''                                      physical units the attribute value is in
standard_unit          :obj:`str`                       ''                                      physical standard unit
display_unit           :obj:`str`                       ''                                      physical display unit (hint for clients)
format                 :obj:`str`                       '6.2f'                                  attribute representation format
min_value              :obj:`str`                       None                                    minimum allowed value
max_value              :obj:`str`                       None                                    maximum allowed value
min_alarm              :obj:`str`                       None                                    minimum value to trigger attribute alarm
max_alarm              :obj:`str`                       None                                    maximum value to trigger attribute alarm
min_warning            :obj:`str`                       None                                    minimum value to trigger attribute warning
max_warning            :obj:`str`                       None                                    maximum value to trigger attribute warning
delta_val              :obj:`str`                       None
delta_t                :obj:`str`                       None
abs_change             :obj:`str`                       None                                    minimum value change between events that causes event filter to send the event
rel_change             :obj:`str`                       None                                    minimum relative change between events that causes event filter to send the event (%)
period                 :obj:`str`                       None
archive_abs_change     :obj:`str`                       None
archive_rel_change     :obj:`str`                       None
archive_period         :obj:`str`                       None
===================== ================================ ======================================= =======================================================================================

.. note::
    avoid using *dformat* parameter. If you need a SPECTRUM attribute of say,
    boolean type, use instead ``dtype=(bool,)``.

"""

    def __init__(self, **kwargs):
        name = kwargs.pop("name", None)
        class_name = kwargs.pop("class_name", None)
        super(attribute, self).__init__(name, class_name)
        if 'dtype' in kwargs:
            kwargs['dtype'], kwargs['dformat'] = \
                get_tango_type_format(kwargs['dtype'], kwargs.get('dformat'))
        self.build_from_dict(kwargs)

    def get_attribute(self, obj):
        return obj.get_device_attr().get_attr_by_name(self.attr_name)

    # --------------------
    # descriptor interface
    # --------------------

    def __get__(self, obj, objtype):
        return self.get_attribute(obj)

    def __set__(self, obj, value):
        attr = self.get_attribute(obj)
        set_complex_value(attr, value)

    def __delete__(self, obj):
        obj.remove_attribute(self.attr_name)


def _attribute(**kwargs):
    if 'dtype' in kwargs:
        kwargs['dtype'], kwargs['dformat'] = \
          get_tango_type_format(kwargs['dtype'], kwargs.get('dformat'))
    return attribute.from_dict(kwargs)


def command(f=None, dtype_in=None, dformat_in=None, doc_in="",
            dtype_out=None, dformat_out=None, doc_out="",):
    """declares a new tango command in a :class:`Device`.
    To be used like a decorator in the methods you want to declare as tango
    commands. For example, to declare a *ramp* command that receives a
    `PyTango.DevDouble` parameter called *current*, do::

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta
            
            @command(dtype_in=float)
            def ramp(self, current):
                self.info_stream("Ramping on %f..." % current)

            # Another more elaborate command
            
            @command(dtype_in=float, doc_in="the pressure to be set",
                     dtype_out=bool, doc_out="True if it worked, False otherwise")
            def pressurize(self, pressure):
                self.info_stream("Pressurizing to %f..." % pressure)

    .. note::
        avoid using *dformat* parameter. If you need a SPECTRUM attribute of
        say, boolean type, use instead ``dtype=(bool,)``.
                    
    :param dtype_in: a :ref:`data type <pytango-hlapi-datatypes>`
                     describing the type of parameter. Default is None meaning
                     no parameter.
    :param dformat_in: parameter data format. Default is None.
    :type dformat_in: AttrDataFormat
    :param doc_in: parameter documentation
    :type doc_in: str

    :param dtype_out: a :ref:`data type <pytango-hlapi-datatypes>`
                      describing the type of return value. Default is None
                      meaning no return value.
    :param dformat_out: return value data format. Default is None.
    :type dformat_out: AttrDataFormat
    :param doc_out: return value documentation
    :type doc_out: str

    """
    if f is None:
        return functools.partial(command,
            dtype_in=dtype_in, dformat_in=dformat_in, doc_in=doc_in,
            dtype_out=dtype_out, dformat_out=dformat_out, doc_out=doc_out)
    name = f.__name__

    dtype_in, dformat_in = get_tango_type_format(dtype_in, dformat_in)
    dtype_out, dformat_out = get_tango_type_format(dtype_out, dformat_out)

    din = [from_typeformat_to_type(dtype_in, dformat_in), doc_in]
    dout = [from_typeformat_to_type(dtype_out, dformat_out), doc_out]
    f.__tango_command__ = name, [din, dout]
    return f


class _property(object):

    def __init__(self, dtype, doc='', default_value=None):
        self.__value = None
        dtype = from_typeformat_to_type(*get_tango_type_format(dtype))
        self.dtype = dtype
        self.doc = doc
        self.default_value = default_value

    def __get__(self, obj, objtype):
        return self.__value

    def __set__(self, obj, value):
        self.__value = value

    def __delete__(self, obj):
        del self.__value


class device_property(_property):
    pass


class class_property(_property):
    pass


def __server_run(classes, args=None, msg_stream=sys.stdout, util=None,
                 event_loop=None):
    import PyTango
    if msg_stream is None:
        import io
        msg_stream = io.BytesIO()

    if args is None:
        args = sys.argv

    if util is None:
        util = PyTango.Util(args)

    if is_seq(classes):
        for klass_info in classes:
            if not hasattr(klass_info, '_api') or klass_info._api < 2:
                raise Exception("When giving a single class, it must implement HLAPI (see PyTango.server)")
            klass_klass = klass_info._DeviceClass
            klass_name = klass_info._DeviceClassName
            klass = klass_info
            util.add_class(klass_klass, klass, klass_name)
    else:
        for klass_name, klass_info in classes.items():
            if is_seq(klass_info):
                klass_klass, klass = klass_info
            else:
                if not hasattr(klass_info, '_api') or klass_info._api < 2:
                    raise Exception("When giving a single class, it must implement HLAPI (see PyTango.server)")
                klass_klass = klass_info._DeviceClass
                klass_name = klass_info._DeviceClassName
                klass = klass_info
            util.add_class(klass_klass, klass, klass_name)
    u_instance = PyTango.Util.instance()
    if event_loop is not None:
        u_instance.server_set_event_loop(event_loop)
    u_instance.server_init()
    msg_stream.write("Ready to accept request\n")
    u_instance.server_run()
    return util


def server_run(classes, args=None, msg_stream=sys.stdout,
               verbose=False, util=None, event_loop=None):
    """Provides a simple way to run a tango server. It handles exceptions
       by writting a message to the msg_stream.

       The `classes` parameter can be either a sequence of :class:`~PyTango.server.Device`
       classes or a dictionary where:
       
       * key is the tango class name
       * value is either:
           #. a :class:`~PyTango.server.Device` class or
           #. a a sequence of two elements :class:`~PyTango.DeviceClass`, :class:`~PyTango.DeviceImpl`
           
       Example 1: registering and running a PowerSupply inheriting from :class:`~PyTango.server.Device`::
       
           from PyTango import server_run
           from PyTango.server import Device, DeviceMeta
       
           class PowerSupply(Device):
               __metaclass__ = DeviceMeta
               
           server_run((PowerSupply,))
           
       Example 2: registering and running a MyServer defined by tango classes 
       `MyServerClass` and `MyServer`::
       
           import PyTango

           class MyServer(PyTango.Device_4Impl):
               pass
               
           class MyServerClass(PyTango.DeviceClass):
               pass
       
           PyTango.server_run({"MyServer": (MyServerClass, MyServer)})
       
       :param classes:
           a sequence of :class:`~PyTango.server.Device` classes or
           a dictionary where keyword is the tango class name and value is a 
           sequence of Tango Device Class python class, and Tango Device python class
       :type classes: sequence or dict
       
       :param args:
           list of command line arguments [default: None, meaning use sys.argv]
       :type args: list
       
       :param msg_stream:
           stream where to put messages [default: sys.stdout]
       
       :param util:
           PyTango Util object [default: None meaning create a Util instance]
       :type util: :class:`~PyTango.Util`

       :param event_loop: event_loop callable
       :type event_loop: callable
       
       :return: The Util singleton object
       :rtype: :class:`~PyTango.Util`
       
       .. versionadded:: 8.0.0
       
       .. versionchanged:: 8.0.3
           Added `util` keyword parameter.
           Returns util object

       .. versionchanged:: 8.1.1
           Changed default msg_stream from *stderr* to *stdout*
           Added `event_loop` keyword parameter.
           Returns util object"""

    if msg_stream is None:
        import io
        msg_stream = io.BytesIO()
    write = msg_stream.write
    try:
        return __server_run(classes, args=args, util=util, event_loop=event_loop)
    except KeyboardInterrupt:
        write("Exiting: Keyboard interrupt\n")
    except DevFailed as df:
        write("Exiting: Server exited with PyTango.DevFailed:\n" + str(df) + "\n")
        if verbose:
            write(traceback.format_exc())
    except Exception as e:
        write("Exiting: Server exited with unforseen exception:\n" + str(e) + "\n")
        if verbose:
            write(traceback.format_exc())
    write("\nExited\n")
