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

"""Server helper classes for writing Tango device servers."""

from __future__ import with_statement
from __future__ import print_function
from __future__ import absolute_import

__all__ = ["DeviceMeta", "Device", "LatestDeviceImpl", "attribute",
           "command", "device_property", "class_property",
           "run", "server_run", "Server"]

import os
import sys
import types
import inspect
import logging
import weakref
import operator
import functools
import traceback

from ._PyTango import (CmdArgType, AttrDataFormat, AttrWriteType,
                       DevFailed, Except, GreenMode, constants,
                       Database, DbDevInfo, DevState, CmdArgType,
                       Attr)
from .attr_data import AttrData
from .device_class import DeviceClass
from .utils import (get_tango_device_classes, is_seq, is_non_str_seq,
                    scalar_to_array_type)
from .codec import loads, dumps

API_VERSION = 2

LatestDeviceImpl = get_tango_device_classes()[-1]

def __build_to_tango_type():
    ret = \
    {
        int         : CmdArgType.DevLong64,
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
        'state'     : CmdArgType.DevState,
    }

    try:
        ret[long] = ret[int]
    except NameError:
        pass


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


def _get_tango_type_format(dtype=None, dformat=None):
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
            attr.set_value_date_quality(value[0][0],
                                        value[0][1],
                                        *value[1:])
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


def check_dev_klass_attr_read_method(tango_device_klass, attribute):
    """
    Checks if method given by it's name for the given DeviceImpl
    class has the correct signature. If a read/write method doesn't
    have a parameter (the traditional Attribute), then the method is
    wrapped into another method which has correct parameter definition
    to make it work.

    :param tango_device_klass: a DeviceImpl class
    :type tango_device_klass: class
    :param attribute: the attribute data information
    :type attribute: AttrData
    """
    read_method = getattr(attribute, "fget", None)
    if read_method:
        method_name = "__read_{0}__".format(attribute.attr_name)
        attribute.read_method_name = method_name
    else:
        method_name = attribute.read_method_name
        read_method = getattr(tango_device_klass, method_name)

    read_args = inspect.getargspec(read_method)

    if len(read_args.args) < 2:
        @functools.wraps(read_method)
        def read_attr(self, attr):
            runner = _get_runner()
            if runner:
                ret = runner.execute(read_method, self)
            else:
                ret = read_method(self)
            if not attr.get_value_flag() and ret is not None:
                set_complex_value(attr, ret)
            return ret
    else:
        @functools.wraps(read_method)
        def read_attr(self, attr):
            runner = _get_runner()
            if runner:
                ret = runner.execute(read_method, self, attr)
            else:
                ret = read_method(self, attr)
            return ret

    method_name = "__read_{0}_wrapper__".format(attribute.attr_name)
    attribute.read_method_name = method_name

    setattr(tango_device_klass, method_name, read_attr)


def check_dev_klass_attr_write_method(tango_device_klass, attribute):
    """
    Checks if method given by it's name for the given DeviceImpl
    class has the correct signature. If a read/write method doesn't
    have a parameter (the traditional Attribute), then the method is
    wrapped into another method which has correct parameter definition
    to make it work.

    :param tango_device_klass: a DeviceImpl class
    :type tango_device_klass: class
    :param attribute: the attribute data information
    :type attribute: AttrData
    """
    write_method = getattr(attribute, "fset", None)
    if write_method:
        method_name = "__write_{0}__".format(attribute.attr_name)
        attribute.write_method_name = method_name
    else:
        method_name = attribute.write_method_name
        write_method = getattr(tango_device_klass, method_name)

    @functools.wraps(write_method)
    def write_attr(self, attr):
        value = attr.get_write_value()
        runner = _get_runner()
        if runner:
            ret = runner.execute(write_method, self, value)
        else:
            ret = write_method(self, value)
        return ret
    setattr(tango_device_klass, method_name, write_attr)


def check_dev_klass_attr_methods(tango_device_klass, attribute):
    """
    Checks if the read and write methods have the correct signature.
    If a read/write method doesn't have a parameter (the traditional
    Attribute), then the method is wrapped into another method to make
    this work.

    :param tango_device_klass: a DeviceImpl class
    :type tango_device_klass: class
    :param attribute: the attribute data information
    :type attribute: AttrData
    """
    if attribute.attr_write in (AttrWriteType.READ,
                                AttrWriteType.READ_WRITE):
        check_dev_klass_attr_read_method(tango_device_klass,
                                         attribute)
    if attribute.attr_write in (AttrWriteType.WRITE,
                                AttrWriteType.READ_WRITE):
        check_dev_klass_attr_write_method(tango_device_klass,
                                          attribute)


class _DeviceClass(DeviceClass):

    def __init__(self, name):
        DeviceClass.__init__(self, name)
        self.set_type(name)

    def _new_device(self, klass, dev_class, dev_name):
        runner = _get_runner()
        if runner:
            return runner.execute(DeviceClass._new_device, self,
                                  klass, dev_class, dev_name)
        else:
            return DeviceClass._new_device(self, klass, dev_class,
                                           dev_name)

    def dyn_attr(self, dev_list):
        """Invoked to create dynamic attributes for the given devices.
        Default implementation calls
        :meth:`TT.initialize_dynamic_attributes` for each device

        :param dev_list: list of devices
        :type dev_list: :class:`PyTango.DeviceImpl`"""

        for dev in dev_list:
            init_dyn_attrs = getattr(dev,
                                     "initialize_dynamic_attributes",
                                     None)
            if init_dyn_attrs and callable(init_dyn_attrs):
                try:
                    init_dyn_attrs()
                except Exception:
                    dev.warn_stream("Failed to initialize dynamic " \
                                    "attributes")
                    dev.debug_stream("Details: " + \
                                     traceback.format_exc())


def create_tango_deviceclass_klass(tango_device_klass, attrs=None):
    klass_name = tango_device_klass.__name__
    if not issubclass(tango_device_klass, (Device)):
        msg = "{0} device must inherit from " \
              "PyTango.server.Device".format(klass_name)
        raise Exception(msg)

    if attrs is None:
        attrs = tango_device_klass.__dict__

    attr_list = {}
    class_property_list = {}
    device_property_list = {}
    cmd_list = {}

    for attr_name, attr_obj in attrs.items():
        if isinstance(attr_obj, attribute):
            if attr_obj.attr_name is None:
                attr_obj._set_name(attr_name)
            else:
                attr_name = attr_obj.attr_name
            attr_list[attr_name] = attr_obj
            check_dev_klass_attr_methods(tango_device_klass, attr_obj)
        elif isinstance(attr_obj, device_property):
            attr_obj.name = attr_name
            device_property_list[attr_name] = [attr_obj.dtype,
                                               attr_obj.doc,
                                               attr_obj.default_value]
        elif isinstance(attr_obj, class_property):
            attr_obj.name = attr_name
            class_property_list[attr_name] = [attr_obj.dtype,
                                              attr_obj.doc,
                                              attr_obj.default_value]
        elif inspect.isroutine(attr_obj):
            if hasattr(attr_obj, "__tango_command__"):
                cmd_name, cmd_info = attr_obj.__tango_command__
                cmd_list[cmd_name] = cmd_info

    devclass_name = klass_name + "Class"

    devclass_attrs = dict(class_property_list=class_property_list,
                          device_property_list=device_property_list,
                          cmd_list=cmd_list, attr_list=attr_list)
    return type(devclass_name, (_DeviceClass,), devclass_attrs)


def init_tango_device_klass(tango_device_klass, attrs=None,
                            tango_class_name=None):
    klass_name = tango_device_klass.__name__
    tango_deviceclass_klass = create_tango_deviceclass_klass(
        tango_device_klass, attrs=attrs)
    if tango_class_name is None:
        if hasattr(tango_device_klass, "TangoClassName"):
            tango_class_name = tango_device_klass.TangoClassName
        else:
            tango_class_name = klass_name
    tango_device_klass.TangoClassClass = tango_deviceclass_klass
    tango_device_klass.TangoClassName = tango_class_name
    tango_device_klass._api = API_VERSION
    return tango_device_klass


def create_tango_device_klass(name, bases, attrs):
    klass_name = name

    LatestDeviceImplMeta = type(LatestDeviceImpl)
    klass = LatestDeviceImplMeta(klass_name, bases, attrs)
    init_tango_device_klass(klass, attrs)
    return klass


def DeviceMeta(name, bases, attrs):
    """
    The :py:data:`metaclass` callable for :class:`Device`.Every
    sub-class of :class:`Device` must have associated this metaclass
    to itself in order to work properly (boost-python internal
    limitation).

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
    """
    High level DeviceImpl API. All Device specific classes should
    inherit from this class."""

    def __init__(self, cl, name):
        self._tango_properties = {}
        LatestDeviceImpl.__init__(self, cl, name)
        self.init_device()

    def init_device(self):
        """
        Tango init_device method. Default implementation calls
        :meth:`get_device_properties`"""
        self.get_device_properties()

    def always_executed_hook(self):
        """
        Tango always_executed_hook. Default implementation does
        nothing
        """
        pass

    def initialize_dynamic_attributes(self):
        """
        Method executed at initializion phase to create dynamic
        attributes. Default implementation does nothing. Overwrite
        when necessary.
        """
        pass


class attribute(AttrData):
    '''
    Declares a new tango attribute in a :class:`Device`. To be used
    like the python native :obj:`property` function. For example, to
    declare a scalar, `PyTango.DevDouble`, read-only attribute called
    *voltage* in a *PowerSupply* :class:`Device` do::

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

            voltage = attribute()

            def read_voltage(self):
                return 999.999

    The same can be achieved with::

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

            @attribute
            def voltage(self):
                return 999.999


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
        avoid using *dformat* parameter. If you need a SPECTRUM
        attribute of say, boolean type, use instead ``dtype=(bool,)``.

    Example of a integer writable attribute with a customized label,
    unit and description::

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

            current = attribute(label="Current", unit="mA", dtype=int,
                                access=AttrWriteType.READ_WRITE,
                                doc="the power supply current")

            def init_device(self):
                Device.init_device(self)
                self._current = -1

            def read_current(self):
                return self._current

            def write_current(self, current):
                self._current = current

    The same, but using attribute as a decorator::

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

            def init_device(self):
                Device.init_device(self)
                self._current = -1

            @attribute(label="Current", unit="mA", dtype=int)
            def current(self):
                """the power supply current"""
                return 999.999

            @current.write
            def current(self, current):
                self._current = current

    In this second format, defining the `write` implies setting the
    attribute access to READ_WRITE.
    '''

    def __init__(self, fget=None, **kwargs):
        self._kwargs = dict(kwargs)
        name = kwargs.pop("name", None)
        class_name = kwargs.pop("class_name", None)

        if fget:
            if inspect.isroutine(fget):
                self.fget = fget
                if 'doc' not in kwargs and 'description' not in kwargs:
                    kwargs['doc'] = fget.__doc__
            else:
                kwargs['fget'] = fget

        super(attribute, self).__init__(name, class_name)
        if 'dtype' in kwargs:
            kwargs['dtype'], kwargs['dformat'] = \
                _get_tango_type_format(kwargs['dtype'],
                                      kwargs.get('dformat'))
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

    def setter(self, fset):
        """
        To be used as a decorator. Will define the decorated method
        as a write attribute method to be called when client writes
        the attribute
        """
        self.fset = fset
        if self.attr_write == AttrWriteType.READ:
            if getattr(self, 'fget', None):
                self.attr_write = AttrWriteType.READ_WRITE
            else:
                self.attr_write = AttrWriteType.WRITE
        return self

    def write(self, fset):
        """
        To be used as a decorator. Will define the decorated method
        as a write attribute method to be called when client writes
        the attribute
        """
        return self.setter(fset)

    def __call__(self, fget):
        return type(self)(fget=fget, **self._kwargs)


def command(f=None, dtype_in=None, dformat_in=None, doc_in="",
            dtype_out=None, dformat_out=None, doc_out="",):
    """
    Declares a new tango command in a :class:`Device`.
    To be used like a decorator in the methods you want to declare as
    tango commands. The following example declares commands:

        * `void TurnOn(void)`
        * `void Ramp(DevDouble current)`
        * `DevBool Pressurize(DevDouble pressure)`

    ::

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

            @command
            def TurnOn(self):
                self.info_stream('Turning on the power supply')

            @command(dtype_in=float)
            def Ramp(self, current):
                self.info_stream('Ramping on %f...' % current)

            @command(dtype_in=float, doc_in='the pressure to be set',
                     dtype_out=bool, doc_out='True if it worked, False otherwise')
            def Pressurize(self, pressure):
                self.info_stream('Pressurizing to %f...' % pressure)

    .. note::
        avoid using *dformat* parameter. If you need a SPECTRUM
        attribute of say, boolean type, use instead ``dtype=(bool,)``.

    :param dtype_in:
        a :ref:`data type <pytango-hlapi-datatypes>` describing the
        type of parameter. Default is None meaning no parameter.
    :param dformat_in: parameter data format. Default is None.
    :type dformat_in: AttrDataFormat
    :param doc_in: parameter documentation
    :type doc_in: str

    :param dtype_out:
        a :ref:`data type <pytango-hlapi-datatypes>` describing the
        type of return value. Default is None meaning no return value.
    :param dformat_out: return value data format. Default is None.
    :type dformat_out: AttrDataFormat
    :param doc_out: return value documentation
    :type doc_out: str
    """
    if f is None:
        return functools.partial(command,
            dtype_in=dtype_in, dformat_in=dformat_in, doc_in=doc_in,
            dtype_out=dtype_out, dformat_out=dformat_out,
            doc_out=doc_out)
    name = f.__name__

    dtype_in, dformat_in = _get_tango_type_format(dtype_in, dformat_in)
    dtype_out, dformat_out = _get_tango_type_format(dtype_out,
                                                    dformat_out)

    din = [from_typeformat_to_type(dtype_in, dformat_in), doc_in]
    dout = [from_typeformat_to_type(dtype_out, dformat_out), doc_out]

    @functools.wraps(f)
    def cmd(self, *args, **kwargs):
        runner = _get_runner()
        if runner:
            ret = runner.execute(f, self, *args, **kwargs)
        else:
            ret = f(self, *args, **kwargs)
        return ret
    cmd.__tango_command__ = name, [din, dout]
    return cmd


class _property(object):

    def __init__(self, dtype, doc='', default_value=None):
        self.name = None
        self.__value = None
        dtype = from_typeformat_to_type(*_get_tango_type_format(dtype))
        self.dtype = dtype
        self.doc = doc
        self.default_value = default_value

    def __get__(self, obj, objtype):
        return obj._tango_properties.get(self.name)

    def __set__(self, obj, value):
        obj._tango_properties[self.name] = value

    def __delete__(self, obj):
        del obj._tango_properties[self.name]


class device_property(_property):
    """
    Declares a new tango device property in a :class:`Device`. To be
    used like the python native :obj:`property` function. For example,
    to declare a scalar, `PyTango.DevString`, device property called
    *host* in a *PowerSupply* :class:`Device` do::

        from PyTango.server import Device, DeviceMeta
        from PyTango.server import device_property

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

            host = device_property(dtype=str)

    :param dtype: Data type (see :ref:`pytango-data-types`)
    :param doc: property documentation (optional)
    :param default_value: default value for the property (optional)
    """
    pass


class class_property(_property):
    """
    Declares a new tango class property in a :class:`Device`. To be
    used like the python native :obj:`property` function. For example,
    to declare a scalar, `PyTango.DevString`, class property called
    *port* in a *PowerSupply* :class:`Device` do::

        from PyTango.server import Device, DeviceMeta
        from PyTango.server import class_property

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

            port = class_property(dtype=int, default_value=9788)

    :param dtype: Data type (see :ref:`pytango-data-types`)
    :param doc: property documentation (optional)
    :param default_value: default value for the property (optional)
    """
    pass


def __to_cb(post_init_callback):
    if post_init_callback is None:
        return lambda : None

    err_msg = "post_init_callback must be a callable or " \
              "sequence <callable [, args, [, kwargs]]>"
    if operator.isCallable(post_init_callback):
        f = post_init_callback
    elif is_non_str_seq(post_init_callback):
        length = len(post_init_callback)
        if length < 1 or length > 3:
            raise TypeError(err_msg)
        cb = post_init_callback[0]
        if not operator.isCallable(cb):
            raise TypeError(err_msg)
        args, kwargs = [], {}
        if length > 1:
            args = post_init_callback[1]
        if length > 2:
            kwargs = post_init_callback[2]
        f = functools.partial(cb, *args, **kwargs)
    else:
        raise TypeError(err_msg)

    return f


def _to_classes(classes):
    uclasses = []
    if is_seq(classes):
        for klass_info in classes:
            if is_seq(klass_info):
                if len(klass_info) == 2:
                    klass_klass, klass = klass_info
                    klass_name = klass.__name__
                else:
                    klass_klass, klass, klass_name = klass_info
            else:
                if not hasattr(klass_info, '_api') or klass_info._api < 2:
                    raise Exception(
                        "When giving a single class, it must " \
                        "implement HLAPI (see PyTango.server)")
                klass_klass = klass_info.TangoClassClass
                klass_name = klass_info.TangoClassName
                klass = klass_info
            uclasses.append((klass_klass, klass, klass_name))
    else:
        for klass_name, klass_info in classes.items():
            if is_seq(klass_info):
                if len(klass_info) == 2:
                    klass_klass, klass = klass_info
                else:
                    klass_klass, klass, klass_name = klass_info
            else:
                if not hasattr(klass_info, '_api') or klass_info._api < 2:
                    raise Exception(
                        "When giving a single class, it must " \
                        "implement HLAPI (see PyTango.server)")
                klass_klass = klass_info.TangoClassClass
                klass_name = klass_info.TangoClassName
                klass = klass_info
            uclasses.append((klass_klass, klass, klass_name))
    return uclasses


def _add_classes(util, classes):
    for class_info in _to_classes(classes):
        util.add_class(*class_info)


def __server_run(classes, args=None, msg_stream=sys.stdout, util=None,
                 event_loop=None, post_init_callback=None,
                 green_mode=None):
    if green_mode is None:
        from PyTango import get_green_mode
        green_mode = get_green_mode()
    gevent_mode = green_mode == GreenMode.Gevent

    import PyTango
    if msg_stream is None:
        write = lambda msg: None
    else:
        write = msg_stream.write

    if args is None:
        args = sys.argv

    post_init_callback = __to_cb(post_init_callback)

    if util is None:
        util = PyTango.Util(args)
    u_instance = PyTango.Util.instance()

    if gevent_mode:
        runner = _create_runner()
        if event_loop:
            event_loop = functools.partial(runner.execute, event_loop)

    if event_loop is not None:
        u_instance.server_set_event_loop(event_loop)

    log = logging.getLogger("PyTango")

    def tango_loop(runner=None):
        _add_classes(util, classes)
        u_instance.server_init()
        if runner:
            runner.execute(post_init_callback)
        else:
            post_init_callback()
        write("Ready to accept request\n")
        u_instance.server_run()
        if runner:
            runner.stop()
        log.debug("Tango loop exit")

    if gevent_mode:
        runner = _create_runner()
        start_new_thread = runner._threading.start_new_thread
        tango_thread_id = start_new_thread(tango_loop, (runner,))
        runner.run()
        log.debug("Runner finished")
    else:
        tango_loop()

    return util

def run(classes, args=None, msg_stream=sys.stdout,
        verbose=False, util=None, event_loop=None,
        post_init_callback=None, green_mode=None):
    """
    Provides a simple way to run a tango server. It handles exceptions
    by writting a message to the msg_stream.

    The `classes` parameter can be either a sequence of:

    * :class:`~PyTango.server.Device` or
    * a sequence of two elements
      :class:`~PyTango.DeviceClass`, :class:`~PyTango.DeviceImpl` or
    * a sequence of three elements
      :class:`~PyTango.DeviceClass`, :class:`~PyTango.DeviceImpl`,
      tango class name (str)

    or a dictionary where:

    * key is the tango class name
    * value is either:
        * a :class:`~PyTango.server.Device` class or
        * a sequence of two elements
          :class:`~PyTango.DeviceClass`, :class:`~PyTango.DeviceImpl`
          or
        * a sequence of three elements
          :class:`~PyTango.DeviceClass`, :class:`~PyTango.DeviceImpl`,
          tango class name (str)

    The optional `post_init_callback` can be a callable (without
    arguments) or a tuple where the first element is the callable,
    the second is a list of arguments (optional) and the third is a
    dictionary of keyword arguments (also optional).

    .. note::
       the order of registration of tango classes defines the order
       tango uses to initialize the corresponding devices.
       if using a dictionary as argument for classes be aware that the
       order of registration becomes arbitrary. If you need a
       predefined order use a sequence or an OrderedDict.

    Example 1: registering and running a PowerSupply inheriting from
    :class:`~PyTango.server.Device`::

        from PyTango.server import Device, DeviceMeta, run

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

        run((PowerSupply,))

    Example 2: registering and running a MyServer defined by tango
    classes `MyServerClass` and `MyServer`::

        from PyTango import Device_4Impl, DeviceClass
        from PyTango.server import run

        class MyServer(Device_4Impl):
            pass

        class MyServerClass(DeviceClass):
            pass

        run({'MyServer': (MyServerClass, MyServer)})

    Example 3: registering and running a MyServer defined by tango
    classes `MyServerClass` and `MyServer`::

        from PyTango import Device_4Impl, DeviceClass
        from PyTango.server import Device, DeviceMeta, run

        class PowerSupply(Device):
            __metaclass__ = DeviceMeta

        class MyServer(Device_4Impl):
            pass

        class MyServerClass(DeviceClass):
            pass

        run([PowerSupply, [MyServerClass, MyServer]])
        # or: run({'MyServer': (MyServerClass, MyServer)})

    :param classes:
        a sequence of :class:`~PyTango.server.Device` classes or
        a dictionary where keyword is the tango class name and value
        is a sequence of Tango Device Class python class, and Tango
        Device python class
    :type classes: sequence or dict

    :param args:
        list of command line arguments [default: None, meaning use
        sys.argv]
    :type args: list

    :param msg_stream:
        stream where to put messages [default: sys.stdout]

    :param util:
        PyTango Util object [default: None meaning create a Util
        instance]
    :type util: :class:`~PyTango.Util`

    :param event_loop: event_loop callable
    :type event_loop: callable

    :param post_init_callback:
        an optional callback that is executed between the calls
        Util.server_init and Util.server_run
    :type post_init_callback:
        callable or tuple (see description above)

    :return: The Util singleton object
    :rtype: :class:`~PyTango.Util`

    .. versionadded:: 8.1.2

    .. versionchanged:: 8.1.4
        when classes argument is a sequence, the items can also be
        a sequence <TangoClass, TangoClassClass>[, tango class name]
    """
    if msg_stream is None:
        write = lambda msg : None
    else:
        write = msg_stream.write
    try:
        return __server_run(classes, args=args, msg_stream=msg_stream,
                            util=util, event_loop=event_loop,
                            post_init_callback=post_init_callback,
                            green_mode=green_mode)
    except KeyboardInterrupt:
        write("Exiting: Keyboard interrupt\n")
    except DevFailed as df:
        write("Exiting: Server exited with PyTango.DevFailed:\n" + \
              str(df) + "\n")
        if verbose:
            write(traceback.format_exc())
    except Exception as e:
        write("Exiting: Server exited with unforseen exception:\n" + \
              str(e) + "\n")
        if verbose:
            write(traceback.format_exc())
    write("\nExited\n")

def server_run(classes, args=None, msg_stream=sys.stdout,
        verbose=False, util=None, event_loop=None,
        post_init_callback=None, green_mode=None):
    """
    Since PyTango 8.1.2 it is just an alias to
    :func:`~PyTango.server.run`. Use :func:`~PyTango.server.run`
    instead.

    .. versionadded:: 8.0.0

    .. versionchanged:: 8.0.3
        Added `util` keyword parameter.
        Returns util object

    .. versionchanged:: 8.1.1
        Changed default msg_stream from *stderr* to *stdout*
        Added `event_loop` keyword parameter.
        Returns util object

    .. versionchanged:: 8.1.2
        Added `post_init_callback` keyword parameter

    .. deprecated:: 8.1.2
        Use :func:`~PyTango.server.run` instead.

    """
    return run(classes, args=args, msg_stream=msg_stream,
               verbose=verbose, util=util, event_loop=event_loop,
               post_init_callback=post_init_callback,
               green_mode=green_mode)


__RUNNER = None

def _get_runner():
    return __RUNNER

def _create_runner():
    global __RUNNER
    if __RUNNER:
        return __RUNNER

    try:
        from queue import Queue
    except:
        from Queue import Queue

    import gevent
    import gevent.event

    class Runner:

        from gevent import _threading

        class Task:

            def __init__(self, event, func, *args, **kwargs):
                self.__event = event
                self.__func = func
                self.__args = args
                self.__kwargs = kwargs
                self.value = None
                self.exception = None

            def __call__(self):
                func = self.__func
                if func:
                    try:
                        self.value = func(*self.__args, **self.__kwargs)
                    except:
                        self.exception = sys.exc_info()
                self.__event.set()

            def run(self):
                return gevent.spawn(self)

        def __init__(self, max_queue_size=0):
            self.__tasks = Queue(max_queue_size)
            self.__stop_event = gevent.event.Event()
            self.__watcher = gevent.get_hub().loop.async()
            self.__watcher.start(self.__step)

        def __step(self):
            task = self.__tasks.get()
            return task.run()

        def run(self, timeout=None):
            return gevent.wait(objects=(self.__stop_event,),
                               timeout=timeout)

        def execute(self, func, *args, **kwargs):
            event = self._threading.Event()
            task = self.Task(event, func, *args, **kwargs)
            self.__tasks.put(task)
            self.__watcher.send()
            event.wait()
            if task.exception:
                Except.throw_python_exception(*task.exception)
            return task.value

        def stop(self):
            task = self.Task(self.__stop_event, None)
            self.__tasks.put(task)
            self.__watcher.send()

    __RUNNER = Runner()
    return __RUNNER


_CLEAN_UP_TEMPLATE = """
import sys
from PyTango import Database

db = Database()
server_instance = '{server_instance}'
try:
    devices = db.get_device_class_list(server_instance)[::2]
    for device in devices:
        db.delete_device(device)
        try:
            db.delete_device_alias(db.get_alias(device))
        except:
            pass
except:
    print ('Failed to cleanup!')
"""

import numpy

def __to_tango_type_fmt(value):
    dfmt = AttrDataFormat.SCALAR
    value_t = type(value)
    dtype = TO_TANGO_TYPE.get(value_t)
    max_dim_x, max_dim_y = 1, 0
    if dtype is None:
        if isinstance(value, numpy.ndarray):
            dtype = TO_TANGO_TYPE.get(value.dtype.name)
            shape_l = len(value.shape)
            if shape_l == 1:
                dfmt = AttrDataFormat.SPECTRUM
                max_dim_x = max(2**16, value.shape[0])
            elif shape_l == 2:
                dfmt = AttrDataFormat.IMAGE
                max_dim_x = max(2**16, value.shape[0])
                max_dim_y = max(2**16, value.shape[1])
        else:
            dtype = CmdArgType.DevEncoded
    return dtype, dfmt, max_dim_x, max_dim_y


def create_tango_class(server, obj, tango_class_name=None, member_filter=None):
    slog = server.server_instance.replace("/", ".")
    log = logging.getLogger("PyTango.Server." + slog)

    obj_klass = obj.__class__
    obj_klass_name = obj_klass.__name__

    if tango_class_name is None:
        tango_class_name = obj_klass_name

    class DeviceDispatcher(Device):
        __metaclass__ = DeviceMeta

        TangoClassName = tango_class_name

        def __init__(self, tango_class_obj, name):
            tango_object = server.get_tango_object(name)
            self.__tango_object = weakref.ref(tango_object)
            Device.__init__(self, tango_class_obj, name)

        def init_device(self):
            Device.init_device(self)
            self.set_state(DevState.ON)

        @property
        def _tango_object(self):
            return self.__tango_object()

        @property
        def _object(self):
            return self._tango_object._object

    DeviceDispatcher.__name__ = tango_class_name
    DeviceDispatcherClass = DeviceDispatcher.TangoClassClass

    for name in dir(obj):
        if name.startswith("_"):
            continue
        log.debug("inspecting %s.%s", obj_klass_name, name)
        try:
            member = getattr(obj, name)
        except:
            log.info("failed to inspect member '%s.%s'",
                        obj_klass_name, name)
            log.debug("Details:", exc_info=1)
        if inspect.isclass(member) or inspect.ismodule(member):
            continue
        if member_filter and not member_filter(obj, tango_class_name,
                                               name, member):
            log.debug("filtered out %s from %s", name, tango_class_name)
            continue
        if inspect.isroutine(member):
            # try to find out if there are any parameters
            in_type = CmdArgType.DevEncoded
            out_type = CmdArgType.DevEncoded
            try:
                arg_spec = inspect.getargspec(member)
                if not arg_spec.args:
                    in_type = CmdArgType.DevVoid
            except TypeError:
                pass

            if in_type == CmdArgType.DevVoid:
                def _command(dev, func_name=None):
                    obj = dev._object
                    f = getattr(obj, func_name)
                    if server.runner:
                        result = server.runner.execute(f)
                    else:
                        result = f()
                    return server.dumps(result)
            else:
                def _command(dev, param, func_name=None):
                    obj = dev._object
                    args, kwargs = loads(*param)
                    f = getattr(obj, func_name)
                    if server.runner:
                        result = server.runner.execute(f, *args, **kwargs)
                    else:
                        result = f(*args, **kwargs)
                    return server.dumps(result)
            cmd = functools.partial(_command, func_name=name)
            cmd.__name__ = name
            doc = member.__doc__
            if doc is None:
                doc = ""
            cmd.__doc__ = doc
            cmd = types.MethodType(cmd, None, DeviceDispatcher)
            setattr(DeviceDispatcher, name, cmd)
            DeviceDispatcherClass.cmd_list[name] = \
                [[in_type, doc], [out_type, ""]]
        else:
            read_only = False
            if hasattr(obj_klass, name):
                kmember = getattr(obj_klass, name)
                if inspect.isdatadescriptor(kmember):
                    if kmember.fset is None:
                        read_only = True
                else:
                    continue
            value = member
            dtype, fmt, x, y = __to_tango_type_fmt(value)
            if dtype is None or dtype == CmdArgType.DevEncoded:
                dtype = CmdArgType.DevEncoded
                fmt = AttrDataFormat.SCALAR
                def read(dev, attr):
                    name = attr.get_name()
                    if server.runner:
                        value = server.runner.execute(getattr, dev._object, name)
                    else:
                        value = getattr(dev._object, name)
                    attr.set_value(*server.dumps(value))
                def write(dev, attr):
                    name = attr.get_name()
                    value = attr.get_write_value()
                    value = loads(*value)
                    if server.runner:
                        server.runner.execute(setattr, dev._object, name, value)
                    else:
                        setattr(dev._object, name, value)
            else:
                def read(dev, attr):
                    name = attr.get_name()
                    if server.runner:
                        value = server.runner.execute(getattr, dev._object, name)
                    else:
                        value = getattr(dev._object, name)
                    attr.set_value(value)
                def write(dev, attr):
                    name = attr.get_name()
                    value = attr.get_write_value()
                    if server.runner:
                        server.runner.execute(setattr, dev._object, name, value)
                    else:
                        setattr(dev._object, name, value)
            read.__name__ = "_read_" + name
            setattr(DeviceDispatcher, read.__name__, read)

            pars = dict(name=name, dtype=dtype, dformat=fmt,
                        max_dim_x=x, max_dim_y=y, fget=read)
            if read_only:
                write = None
            else:
                write.__name__ = "_write_" + name
                pars['fset'] = write
                setattr(DeviceDispatcher, write.__name__, write)
            attr_data = AttrData.from_dict(pars)
            DeviceDispatcherClass.attr_list[name] = attr_data
    return DeviceDispatcher


class Server:
    """
    Server helper
    """

    Phase0, Phase1, Phase2 = range(3)
    PreInitPhase = Phase1
    PostInitPhase = Phase2

    class TangoObjectAdapter:

        def __init__(self, server, obj, full_name, alias=None,
                     tango_class_name=None):
            self.__server = weakref.ref(server)
            self.full_name = full_name
            self.alias = alias
            self.class_name = obj.__class__.__name__
            if tango_class_name is None:
                tango_class_name = self.class_name
            self.tango_class_name = tango_class_name
            self.__object = weakref.ref(obj, self.__onObjectDeleted)

        def __onObjectDeleted(self, object_weak):
            self.__object = None
            server = self._server
            server.log.info("object deleted %s(%s)", self.class_name,
                            self.full_name)
            server.unregister_object(self.full_name)

        @property
        def _server(self):
            return self.__server()

        @property
        def _object(self):
            obj = self.__object
            if obj is None:
                return None
            return obj()

    def __init__(self, server_name, server_type=None, port=None,
                 event_loop_callback=None, init_callbacks=None,
                 auto_clean=False, green_mode=None, tango_classes=None,
                 protocol="pickle"):
        if server_name is None:
            raise ValueError("Must give a valid server name")
        self.__server_name = server_name
        self.__server_type = server_type
        self.__port = port
        self.__event_loop_callback = event_loop_callback
        if init_callbacks is None:
            init_callbacks = {}
        self.__init_callbacks = init_callbacks
        self.__util = None
        self.__objects = {}
        self.__running = False
        self.__auto_clean = auto_clean
        self.__green_mode = green_mode
        self.__protocol = protocol
        self.__tango_classes = _to_classes(tango_classes or [])
        self.__tango_devices = []
        if self.gevent_mode:
            self.__runner = _create_runner()
        else:
            self.__runner = None
        self.log = logging.getLogger("PyTango.Server")
        self.__phase = Server.Phase0

    def __build_args(self):
        args = [self.server_type, self.__server_name]
        if self.__port is not None:
            args.extend(["-ORBendPoint",
                         "giop:tcp::{0}".format(self.__port)])
        return args

    def __exec_cb(self, cb):
        if not cb:
            return
        if self.gevent_mode:
            self.__runner.execute(cb)
        else:
            cb()

    def __find_tango_class(self, key):
        pass

    def __prepare(self):
        """Update database with existing devices"""
        self.log.debug("prepare")

        if self.__phase > 0:
            raise RuntimeError("Internal error: Can only prepare in phase 0")

        server_instance = self.server_instance
        db = Database()

        # get list of server devices if server was already registered
        server_registered = server_instance in db.get_server_list()

        if server_registered:
            dserver_name = "dserver/{0}".format(server_instance)
            if db.import_device(dserver_name).exported:
                import PyTango
                dserver = PyTango.DeviceProxy(dserver_name)
                try:
                    dserver.ping()
                    raise Exception("Server already running")
                except:
                    self.log.info("Last time server was not properly "
                                  "shutdown!")
            db_class_map, db_device_map = self.get_devices()
        else:
            db_class_map, db_device_map = {}, {}

        db_devices_add = {}

        # all devices that are registered in database that are not registered
        # as tango objects or for which the tango class changed will be removed
        db_devices_remove = set(db_device_map) - set(self.__objects)

        for local_name, local_object in self.__objects.items():
            local_class_name = local_object.tango_class_name
            db_class_name = db_device_map.get(local_name)
            if db_class_name:
                if local_class_name != db_class_name:
                    db_devices_remove.add(local_name)
                    db_devices_add[local_name] = local_object
            else:
                db_devices_add[local_name] = local_object

        for device in db_devices_remove:
            db.delete_device(device)
            try:
                db.delete_device_alias(db.get_alias(device))
            except:
                pass

        # register devices in database

        # add DServer
        db_dev_info = DbDevInfo()
        db_dev_info.server = server_instance
        db_dev_info._class = "DServer"
        db_dev_info.name = "dserver/" + server_instance

        db_dev_infos = [db_dev_info]
        aliases = []
        for obj_name, obj in db_devices_add.items():
            db_dev_info = DbDevInfo()
            db_dev_info.server = server_instance
            db_dev_info._class = obj.tango_class_name
            db_dev_info.name = obj.full_name
            db_dev_infos.append(db_dev_info)
            if obj.alias:
                aliases.append((obj.full_name, obj.alias))

        db.add_server(server_instance, db_dev_infos)

        # add aliases
        for alias_info in aliases:
            db.put_device_alias(*alias_info)

    def __clean_up_process(self):
        if not self.__auto_clean:
            return
        clean_up = _CLEAN_UP_TEMPLATE.format(server_instance=self.server_instance)
        import subprocess
        res = subprocess.call([sys.executable, "-c", clean_up])
        if res:
            self.log.error("Failed to cleanup")

    def __initialize(self):
        self.log.debug("initialize")
        gevent_mode = self.gevent_mode
        event_loop = self.__event_loop_callback

        util = self.tango_util
        u_instance = util.instance()

        if gevent_mode:
            if event_loop:
                event_loop = functools.partial(self.__runner.execute,
                                               event_loop)
        if event_loop:
            u_instance.server_set_event_loop(event_loop)

        _add_classes(util, self.__tango_classes)

        if gevent_mode:
            start_new_thread = self.__runner._threading.start_new_thread
            tango_thread_id = start_new_thread(self.__tango_loop, ())

    def __run(self, timeout=None):
        if self.gevent_mode:
            return self.__runner.run(timeout=timeout)
        else:
            self.__tango_loop()

    def __tango_loop(self):
        self.log.debug("tango_loop")
        self.__running = True
        u_instance = self.tango_util.instance()
        u_instance.server_init()
        self._phase = Server.Phase2
        self.log.info("Ready to accept request")
        u_instance.server_run()
        if self.gevent_mode:
            self.__runner.stop()
        if self.__auto_clean:
            self.__clean_up_process()
        self.log.debug("Tango loop exit")

    @property
    def _phase(self):
        return self.__phase

    @_phase.setter
    def _phase(self, phase):
        self.__phase = phase
        cb = self.__init_callbacks.get(phase)
        self.__exec_cb(cb)

    @property
    def server_type(self):
        server_type = self.__server_type
        if server_type is None:
            server_file = os.path.basename(sys.argv[0])
            server_type = os.path.splitext(server_file)[0]
        return server_type

    @property
    def server_instance(self):
        return "{0}/{1}".format(self.server_type, self.__server_name)

    @property
    def tango_util(self):
        if self.__util is None:
            import PyTango
            self.__util = PyTango.Util(self.__build_args())
            self._phase = Server.Phase1
        return self.__util

    @property
    def green_mode(self):
        gm = self.__green_mode
        if gm is None:
            from PyTango import get_green_mode
            gm = get_green_mode()
        return gm

    @green_mode.setter
    def green_mode(self, gm):
        if gm == self.__green_mode:
            return
        if self.__running:
            raise RuntimeError("Cannot change green mode while "
                               "server is running")
        self.__green_mode = gm

    @property
    def gevent_mode(self):
        return self.green_mode == GreenMode.Gevent

    @property
    def runner(self):
        return self.__runner

    def dumps(self, obj):
        return dumps(self.__protocol, obj)

    def get_devices(self):
        """
        Helper that retuns a dict of devices for this server.

        :return:
            Returns a tuple of two elements:
              - dict<tango class name : list of device names>
              - dict<device names : tango class name>
        :rtype: tuple<dict, dict>
        """
        import PyTango
        db = PyTango.Database()
        server = self.server_instance
        dev_list = db.get_device_class_list(server)
        class_map, dev_map  = {}, {}
        for class_name, dev_name in zip(dev_list[1::2], dev_list[::2]):
            dev_names = class_map.get(class_name)
            if dev_names is None:
                class_map[class_name] = dev_names = []
            dev_name = dev_name.lower()
            dev_names.append(dev_name)
            dev_map[dev_name] = class_name
        return class_map, dev_map

    def get_tango_object(self, name):
        return self.__objects.get(name.lower())

    def get_tango_class(self, tango_class_name):
        for klass in self.__tango_classes:
            if klass.TangoClassName == tango_class_name:
                return klass

    def register_tango_device(self, klass, name):
        if inspect.isclass(klass):
            if isinstance(klass, Device):
                kk, k, kname = Device.TangoClassClass, Device, Device.TangoClassName
            else:
                raise ValueError
        else:
            raise NotImplementedError

    def register_tango_class(self, klass):
        if self._phase > Server.Phase1:
            raise RuntimeError("Cannot add new class after phase 1 "
                               "(i.e. after server_init)")
        self.__tango_classes.append(klass)

    def unregister_object(self, name):
        tango_object = self.__objects.pop(name.lower())
        if self._phase > Server.Phase1:
            import PyTango
            util = PyTango.Util.instance()
            if not util.is_svr_shutting_down():
                util.delete_device(tango_object.tango_class_name, name)

    def register_object(self, obj, name, tango_class_name=None,
                        member_filter=None):
        """
        :param member_filter:
            callable(obj, tango_class_name, member_name, member) -> bool
        """
        slash_count = name.count("/")
        if slash_count == 0:
            alias = name
            full_name = "{0}/{1}".format(self.server_instance, name)
        elif slash_count == 2:
            alias = None
            full_name = name
        else:
            raise ValueError("Invalid name")

        class_name = tango_class_name or obj.__class__.__name__
        tango_class = self.get_tango_class(class_name)

        if tango_class is None:
            tango_class = create_tango_class(self, obj, class_name,
                                             member_filter=member_filter)
            self.register_tango_class(tango_class)

        tango_object = self.TangoObjectAdapter(self, obj, full_name, alias,
                                               tango_class_name=class_name)
        self.__objects[full_name.lower()] = tango_object
        if self._phase > Server.Phase1:
            import PyTango
            util = PyTango.Util.instance()
            util.create_device(class_name, name)
        return tango_object

    def run(self, timeout=None):
        self.log.debug("run")
        gevent_mode = self.gevent_mode
        running = self.__running
        if not running:
            self.__prepare()
            self.__initialize()
        else:
            if not gevent_mode:
                raise RuntimeError("Server is already running")
        self.__run(timeout=timeout)
