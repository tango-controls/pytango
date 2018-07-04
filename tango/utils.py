# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

"""
This is an internal PyTango module.
"""

from __future__ import with_statement
from __future__ import print_function

import os
import sys
import six
import types
import numbers
import collections
import enum

from ._tango import StdStringVector, StdDoubleVector, \
    DbData, DbDevInfos, DbDevExportInfos, CmdArgType, AttrDataFormat, \
    EventData, AttrConfEventData, DataReadyEventData, DevFailed, constants, \
    DevState, CommunicationFailed, PipeEventData, DevIntrChangeEventData

from . import _tango
from .constants import AlrmValueNotSpec, StatusNotSet, TgLibVers
from .release import Release

__all__ = (
    "requires_pytango", "requires_tango",
    "is_pure_str", "is_seq", "is_non_str_seq", "is_integer",
    "is_number", "is_scalar_type", "is_array_type", "is_numerical_type",
    "is_int_type", "is_float_type", "is_bool_type", "is_binary_type",
    "is_str_type", "obj_2_str", "seqStr_2_obj",
    "scalar_to_array_type",
    "document_method", "document_static_method", "document_enum",
    "CaselessList", "CaselessDict", "EventCallBack", "get_home",
    "from_version_str_to_hex_str", "from_version_str_to_int",
    "seq_2_StdStringVector", "StdStringVector_2_seq",
    "dir2", "TO_TANGO_TYPE")

__docformat__ = "restructuredtext"

# Types

_scalar_int_types = (
    CmdArgType.DevShort,
    CmdArgType.DevUShort,
    CmdArgType.DevInt,
    CmdArgType.DevLong,
    CmdArgType.DevULong,
    CmdArgType.DevLong64,
    CmdArgType.DevULong64,)

_scalar_float_types = (
    CmdArgType.DevFloat,
    CmdArgType.DevDouble,)

_scalar_numerical_types = (
    _scalar_int_types +
    _scalar_float_types)

_scalar_str_types = (
    CmdArgType.DevString,
    CmdArgType.ConstDevString,)

_scalar_bool_types = (
    CmdArgType.DevBoolean,)

_scalar_types = (
    _scalar_numerical_types +
    _scalar_str_types +
    _scalar_bool_types +
    (CmdArgType.DevEncoded,
     CmdArgType.DevUChar,))

_array_int_types = (
    CmdArgType.DevVarShortArray,
    CmdArgType.DevVarUShortArray,
    CmdArgType.DevVarLongArray,
    CmdArgType.DevVarULongArray,
    CmdArgType.DevVarLong64Array,
    CmdArgType.DevVarULong64Array)

_array_float_types = (
    CmdArgType.DevVarFloatArray,
    CmdArgType.DevVarDoubleArray)

_array_numerical_types = (
    _array_int_types +
    _array_float_types)

_array_str_types = (
    CmdArgType.DevVarStringArray,)

_array_bool_types = (
    CmdArgType.DevVarBooleanArray,)

_array_types = (
    _array_numerical_types +
    _array_bool_types +
    _array_str_types +
    (CmdArgType.DevVarCharArray,
     CmdArgType.DevVarDoubleStringArray,
     CmdArgType.DevVarLongStringArray,))

_binary_types = (
    CmdArgType.DevEncoded,
    CmdArgType.DevVarCharArray,)


def __build_to_tango_type():
    ret = {
        int: CmdArgType.DevLong64,
        str: CmdArgType.DevString,
        bool: CmdArgType.DevBoolean,
        bytearray: CmdArgType.DevEncoded,
        float: CmdArgType.DevDouble,
        chr: CmdArgType.DevUChar,
        None: CmdArgType.DevVoid,

        'int': CmdArgType.DevLong64,
        'int16': CmdArgType.DevShort,
        'int32': CmdArgType.DevLong,
        'int64': CmdArgType.DevLong64,
        'uint': CmdArgType.DevULong64,
        'uint16': CmdArgType.DevUShort,
        'uint32': CmdArgType.DevULong,
        'uint64': CmdArgType.DevULong64,
        'str': CmdArgType.DevString,
        'string': CmdArgType.DevString,
        'text': CmdArgType.DevString,
        'bool': CmdArgType.DevBoolean,
        'boolean': CmdArgType.DevBoolean,
        'bytes': CmdArgType.DevEncoded,
        'bytearray': CmdArgType.DevEncoded,
        'float': CmdArgType.DevDouble,
        'float32': CmdArgType.DevFloat,
        'float64': CmdArgType.DevDouble,
        'double': CmdArgType.DevDouble,
        'byte': CmdArgType.DevUChar,
        'chr': CmdArgType.DevUChar,
        'char': CmdArgType.DevUChar,
        'None': CmdArgType.DevVoid,
        'state': CmdArgType.DevState,
        'enum': CmdArgType.DevEnum,
        'blob': CmdArgType.DevPipeBlob,
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
        FROM_TANGO_TO_NUMPY_TYPE = {
            CmdArgType.DevBoolean: numpy.bool8,
            CmdArgType.DevUChar: numpy.ubyte,
            CmdArgType.DevShort: numpy.short,
            CmdArgType.DevUShort: numpy.ushort,
            CmdArgType.DevLong: numpy.int32,
            CmdArgType.DevULong: numpy.uint32,
            CmdArgType.DevLong64: numpy.int64,
            CmdArgType.DevULong64: numpy.uint64,
            CmdArgType.DevString: numpy.str,
            CmdArgType.DevDouble: numpy.float64,
            CmdArgType.DevFloat: numpy.float32,
        }

        for key, value in FROM_TANGO_TO_NUMPY_TYPE.items():
            ret[value] = key
    return ret


TO_TANGO_TYPE = __build_to_tango_type()

_scalar_to_array_type = {
    CmdArgType.DevBoolean: CmdArgType.DevVarBooleanArray,
    CmdArgType.DevUChar: CmdArgType.DevVarCharArray,
    CmdArgType.DevShort: CmdArgType.DevVarShortArray,
    CmdArgType.DevUShort: CmdArgType.DevVarUShortArray,
    CmdArgType.DevInt: CmdArgType.DevVarLongArray,
    CmdArgType.DevLong: CmdArgType.DevVarLongArray,
    CmdArgType.DevULong: CmdArgType.DevVarULongArray,
    CmdArgType.DevLong64: CmdArgType.DevVarLong64Array,
    CmdArgType.DevULong64: CmdArgType.DevVarULong64Array,
    CmdArgType.DevFloat: CmdArgType.DevVarFloatArray,
    CmdArgType.DevDouble: CmdArgType.DevVarDoubleArray,
    CmdArgType.DevString: CmdArgType.DevVarStringArray,
    CmdArgType.ConstDevString: CmdArgType.DevVarStringArray,
}

# add derived scalar types to scalar to array map
for k, v in TO_TANGO_TYPE.items():
    if v in _scalar_to_array_type:
        _scalar_to_array_type[k] = _scalar_to_array_type[v]

__NO_STR_VALUE = AlrmValueNotSpec, StatusNotSet

__device_classes = None

bool_ = lambda value_str: value_str.lower() == "true"


def __import(name):
    __import__(name)
    return sys.modules[name]


def __requires(package_name, min_version=None, conflicts=(),
               software_name="Software"):
    from distutils.version import LooseVersion
    package_name_l = package_name.lower()
    if package_name_l == 'pytango':
        curr_version = LooseVersion(Release.version)
    elif package_name_l == 'tango':
        curr_version = LooseVersion(TgLibVers)
    else:
        try:
            package = __import(package_name)
            curr_version = LooseVersion(package.__version__)
        except ImportError:
            msg = "Could not find package {0} required by {1}".format(
                package_name, software_name)
            raise Exception(msg)
        except:
            msg = "Error importing package {0} required by {1}".format(
                package_name, software_name)
            raise Exception(msg)

    if min_version is not None:
        min_version = LooseVersion(min_version)
        if min_version > curr_version:
            msg = "{0} requires {1} {2} but {3} installed".format(
                software_name, package_name, min_version, curr_version)
            raise Exception(msg)

    conflicts = map(LooseVersion, conflicts)
    if curr_version in conflicts:
        msg = "{0} cannot run with {1} {2}".format(
            software_name, package_name, curr_version)
        raise Exception(msg)
    return True


def requires_pytango(min_version=None, conflicts=(),
                     software_name="Software"):
    """
    Determines if the required PyTango version for the running
    software is present. If not an exception is thrown.
    Example usage::

        from tango import requires_pytango

        requires_pytango('7.1', conflicts=['8.1.1'], software='MyDS')

    :param min_version:
        minimum PyTango version [default: None, meaning no minimum
        required]. If a string is given, it must be in the valid
        version number format
        (see: :class:`~distutils.version.LooseVersion`)
    :type min_version:
        None, str, :class:`~distutils.version.LooseVersion`
    :param conflics:
        a sequence of PyTango versions which conflict with the
        software using it
    :type conflics:
        seq<str|LooseVersion>
    :param software_name:
        software name using tango. Used in the exception message
    :type software_name: str

    :raises Exception: if the required PyTango version is not met

    New in PyTango 8.1.4
    """
    return __requires("pytango", min_version=min_version,
                      conflicts=conflicts, software_name=software_name)


def requires_tango(min_version=None, conflicts=(), software_name="Software"):
    """
    Determines if the required Tango version for the running
    software is present. If not an exception is thrown.
    Example usage::

        from tango import requires_tango

        requires_tango('7.1', conflicts=['8.1.1'], software='MyDS')

    :param min_version:
        minimum Tango version [default: None, meaning no minimum
        required]. If a string is given, it must be in the valid
        version number format
        (see: :class:`~distutils.version.LooseVersion`)
    :type min_version:
        None, str, :class:`~distutils.version.LooseVersion`
    :param conflics:
        a sequence of Tango versions which conflict with the
        software using it
    :type conflics:
        seq<str|LooseVersion>
    :param software_name:
        software name using Tango. Used in the exception message
    :type software_name: str

    :raises Exception: if the required Tango version is not met

    New in PyTango 8.1.4
    """
    return __requires("Tango", min_version=min_version,
                      conflicts=conflicts, software_name=software_name)


def get_tango_device_classes():
    global __device_classes
    if __device_classes is None:

        __device_classes = [_tango.DeviceImpl]
        i = 2
        while True:
            dc = "Device_{0}Impl".format(i)
            try:
                __device_classes.append(getattr(_tango, dc))
                i = i + 1
            except AttributeError:
                break
    return __device_classes


def get_latest_device_class():
    return get_tango_device_classes()[-1]


try:
    __str_klasses = basestring,
except NameError:
    __str_klasses = str,

__int_klasses = int,
__number_klasses = numbers.Number,
__seq_klasses = collections.Sequence, bytearray, StdStringVector

__use_unicode = False
try:
    unicode
    __use_unicode = True
    __str_klasses = tuple(list(__str_klasses) + [unicode])
except NameError:
    pass

__use_long = False
try:
    long
    __use_long = True
    __int_klasses = tuple(list(__int_klasses) + [long])
except NameError:
    pass

if constants.NUMPY_SUPPORT:
    import numpy

    __int_klasses = tuple(list(__int_klasses) + [numpy.integer])
    __number_klasses = tuple(list(__number_klasses) + [numpy.number])
    __seq_klasses = tuple(list(__seq_klasses) + [numpy.ndarray])

__str_klasses = tuple(__str_klasses)
__int_klasses = tuple(__int_klasses)
__number_klasses = tuple(__number_klasses)
__seq_klasses = tuple(__seq_klasses)


def __get_tango_type(obj):
    if is_non_str_seq(obj):
        tg_type, tg_format = get_tango_type(obj[0])
        tg_format = AttrDataFormat(int(tg_format) + 1)
        return tg_type, tg_format
    elif is_pure_str(obj):
        r = CmdArgType.DevString
    elif isinstance(obj, DevState):
        r = CmdArgType.DevState
    elif isinstance(obj, bool):
        r = CmdArgType.DevBoolean
    elif isinstance(obj, __int_klasses):
        r = CmdArgType.DevLong64
    elif isinstance(obj, __number_klasses):
        r = CmdArgType.DevDouble
    return r, AttrDataFormat.SCALAR


def __get_tango_type_numpy_support(obj):
    try:
        ndim, dtype = obj.ndim, str(obj.dtype)
        if ndim > 2:
            raise TypeError('cannot translate numpy array with {0} '
                            'dimensions to tango type'.format(obj.ndim))
        return TO_TANGO_TYPE[dtype], AttrDataFormat(ndim)
    except AttributeError:
        return __get_tango_type(obj)


def get_tango_type(obj):
    if constants.NUMPY_SUPPORT:
        return __get_tango_type_numpy_support(obj)
    return __get_tango_type(obj)


class EnumTypeError(Exception):
    """Invalid Enum class for use with DEV_ENUM."""


def get_enum_labels(enum_cls):
    """
    Return list of enumeration labels from Enum class.

    The list is useful when creating an attribute, for the
    `enum_labels` parameter.  The enumeration values are checked
    to ensure they are unique, start at zero, and increment by one.

    :param enum_cls: the Enum class to be inspected
    :type enum_cls: :py:obj:`enum.Enum`

    :return: List of label strings
    :rtype: :py:obj:`list`

    :raises EnumTypeError: in case the given class is invalid
    """
    if not issubclass(enum_cls, enum.Enum):
        raise EnumTypeError("Input class '%s' must be derived from enum.Enum"
                            % enum_cls)

    # Check there are no duplicate labels
    try:
        enum.unique(enum_cls)
    except ValueError as exc:
        raise EnumTypeError("Input class '%s' must be unique - %s"
                            % (enum_cls, exc))

    # Check the values start at 0, and increment by 1, since that is
    # assumed by tango's DEV_ENUM implementation.
    values = [member.value for member in enum_cls]
    if not values:
        raise EnumTypeError("Input class '%s' has no members!" % enum_cls)
    expected_value = 0
    for value in values:
        if value != expected_value:
            raise EnumTypeError("Enum values for '%s' must start at 0 and "
                                "increment by 1.  Values: %s"
                                % (enum_cls, values))
        expected_value += 1

    return [member.name for member in enum_cls]


def is_pure_str(obj):
    """
    Tells if the given object is a python string.

    In python 2.x this means any subclass of basestring.
    In python 3.x this means any subclass of str.

    :param obj: the object to be inspected
    :type obj: :py:obj:`object`

    :return: True is the given obj is a string or False otherwise
    :rtype: :py:obj:`bool`
    """
    return isinstance(obj, __str_klasses)


def is_seq(obj):
    """
    Tells if the given object is a python sequence.

    It will return True for any collections.Sequence (list, tuple,
    str, bytes, unicode), bytearray and (if numpy is enabled)
    numpy.ndarray

    :param obj: the object to be inspected
    :type obj: :py:obj:`object`

    :return: True is the given obj is a sequence or False otherwise
    :rtype: :py:obj:`bool`
    """
    return isinstance(obj, __seq_klasses)


def is_non_str_seq(obj):
    """
    Tells if the given object is a python sequence (excluding string
    sequences).

    It will return True for any collections.Sequence (list, tuple (and
    bytes in python3)), bytearray and (if numpy is enabled)
    numpy.ndarray

    :param obj: the object to be inspected
    :type obj: :py:obj:`object`

    :return: True is the given obj is a sequence or False otherwise
    :rtype: :py:obj:`bool`
    """
    return is_seq(obj) and not is_pure_str(obj)


def is_integer(obj):
    """
    Tells if the given object is a python integer.

    It will return True for any int, long (in python 2) and
    (if numpy is enabled) numpy.integer

    :param obj: the object to be inspected
    :type obj: :py:obj:`object`

    :return:
        True is the given obj is a python integer or False otherwise
    :rtype: :py:obj:`bool`
    """
    return isinstance(obj, __int_klasses)


def is_number(obj):
    """
    Tells if the given object is a python number.

    It will return True for any numbers.Number and (if numpy is
    enabled) numpy.number

    :param obj: the object to be inspected
    :type obj: :py:obj:`object`

    :return:
        True is the given obj is a python number or False otherwise
    :rtype: :py:obj:`bool`
    """
    return isinstance(obj, __number_klasses)


def is_scalar(tg_type):
    """Tells if the given tango type is a scalar

    :param tg_type: tango type
    :type tg_type: :class:`tango.CmdArgType`

    :return: True if the given tango type is a scalar or False otherwise
    :rtype: :py:obj:`bool`
    """

    global _scalar_types
    return tg_type in _scalar_types


is_scalar_type = is_scalar


def is_array(tg_type):
    """Tells if the given tango type is an array type

    :param tg_type: tango type
    :type tg_type: :class:`tango.CmdArgType`

    :return: True if the given tango type is an array type or False otherwise
    :rtype: :py:obj:`bool`
    """
    global _array_types
    return tg_type in _array_types


is_array_type = is_array


def is_numerical(tg_type, inc_array=False):
    """Tells if the given tango type is numerical

    :param tg_type: tango type
    :type tg_type: :class:`tango.CmdArgType`
    :param inc_array: (optional, default is False) determines if include array
                      in the list of checked types
    :type inc_array: :py:obj:`bool`

    :return: True if the given tango type is a numerical or False otherwise
    :rtype: :py:obj:`bool`
    """
    global _scalar_numerical_types, _array_numerical_types
    if tg_type in _scalar_numerical_types:
        return True
    if not inc_array:
        return False
    return tg_type in _array_numerical_types


is_numerical_type = is_numerical


def is_int(tg_type, inc_array=False):
    """Tells if the given tango type is integer

    :param tg_type: tango type
    :type tg_type: :class:`tango.CmdArgType`
    :param inc_array: (optional, default is False) determines if include array
                      in the list of checked types
    :type inc_array: :py:obj:`bool`

    :return: True if the given tango type is integer or False otherwise
    :rtype: :py:obj:`bool`
    """
    global _scalar_int_types, _array_int_types
    if tg_type in _scalar_int_types:
        return True
    if not inc_array:
        return False
    return tg_type in _array_int_types


is_int_type = is_int


def is_float(tg_type, inc_array=False):
    """Tells if the given tango type is float

    :param tg_type: tango type
    :type tg_type: :class:`tango.CmdArgType`
    :param inc_array: (optional, default is False) determines if include array
                      in the list of checked types
    :type inc_array: :py:obj:`bool`

    :return: True if the given tango type is float or False otherwise
    :rtype: :py:obj:`bool`
    """
    global _scalar_float_types, _array_float_types
    if tg_type in _scalar_float_types:
        return True
    if not inc_array:
        return False
    return tg_type in _array_float_types


is_float_type = is_float


def is_bool(tg_type, inc_array=False):
    """Tells if the given tango type is boolean

    :param tg_type: tango type
    :type tg_type: :class:`tango.CmdArgType`
    :param inc_array: (optional, default is False) determines if include array
                      in the list of checked types
    :type inc_array: :py:obj:`bool`

    :return: True if the given tango type is boolean or False otherwise
    :rtype: :py:obj:`bool`
    """
    global _scalar_bool_types, _array_bool_types
    if tg_type in _scalar_bool_types:
        return True
    if not inc_array:
        return False
    return tg_type in _array_bool_types


is_bool_type = is_bool


def is_str(tg_type, inc_array=False):
    """Tells if the given tango type is string

    :param tg_type: tango type
    :type tg_type: :class:`tango.CmdArgType`
    :param inc_array: (optional, default is False) determines if include array
                      in the list of checked types
    :type inc_array: :py:obj:`bool`

    :return: True if the given tango type is string or False otherwise
    :rtype: :py:obj:`bool`
    """
    global _scalar_str_types, _array_str_types
    if tg_type in _scalar_str_types:
        return True
    if not inc_array:
        return False
    return tg_type in _array_str_types


is_str_type = is_str


def is_binary(tg_type, inc_array=False):
    """Tells if the given tango type is binary

    :param tg_type: tango type
    :type tg_type: :class:`tango.CmdArgType`
    :param inc_array: (optional, default is False) determines if include array
                      in the list of checked types
    :type inc_array: :py:obj:`bool`

    :return: True if the given tango type is binary or False otherwise
    :rtype: :py:obj:`bool`
    """
    global _binary_types
    return tg_type in _binary_types


is_binary_type = is_binary


def seq_2_StdStringVector(seq, vec=None):
    """Converts a python sequence<str> object to a :class:`tango.StdStringVector`

        :param seq: the sequence of strings
        :type seq: sequence<:py:obj:`str`>
        :param vec: (optional, default is None) an :class:`tango.StdStringVector`
                    to be filled. If None is given, a new :class:`tango.StdStringVector`
                    is created
        :return: a :class:`tango.StdStringVector` filled with the same contents as seq
        :rtype: :class:`tango.StdStringVector`
    """
    if vec is None:
        if isinstance(seq, StdStringVector):
            return seq
        vec = StdStringVector()
    if not isinstance(vec, StdStringVector):
        raise TypeError('vec must be a tango.StdStringVector')
    for e in seq:
        vec.append(str(e))
    return vec


def StdStringVector_2_seq(vec, seq=None):
    """Converts a :class:`tango.StdStringVector` to a python sequence<str>

        :param seq: the :class:`tango.StdStringVector`
        :type seq: :class:`tango.StdStringVector`
        :param vec: (optional, default is None) a python sequence to be filled.
                     If None is given, a new list is created
        :return: a python sequence filled with the same contents as seq
        :rtype: sequence<str>
    """
    if seq is None:
        seq = []
    if not isinstance(vec, StdStringVector):
        raise TypeError('vec must be a tango.StdStringVector')
    for e in vec:
        seq.append(str(e))
    return seq


def seq_2_StdDoubleVector(seq, vec=None):
    """Converts a python sequence<float> object to a :class:`tango.StdDoubleVector`

        :param seq: the sequence of floats
        :type seq: sequence<:py:obj:`float`>
        :param vec: (optional, default is None) an :class:`tango.StdDoubleVector`
                    to be filled. If None is given, a new :class:`tango.StdDoubleVector`
                    is created
        :return: a :class:`tango.StdDoubleVector` filled with the same contents as seq
        :rtype: :class:`tango.StdDoubleVector`
    """
    if vec is None:
        if isinstance(seq, StdDoubleVector):
            return seq
        vec = StdDoubleVector()
    if not isinstance(vec, StdDoubleVector):
        raise TypeError('vec must be a tango.StdDoubleVector')
    for e in seq:
        vec.append(str(e))
    return vec


def StdDoubleVector_2_seq(vec, seq=None):
    """Converts a :class:`tango.StdDoubleVector` to a python sequence<float>

        :param seq: the :class:`tango.StdDoubleVector`
        :type seq: :class:`tango.StdDoubleVector`
        :param vec: (optional, default is None) a python sequence to be filled.
                     If None is given, a new list is created
        :return: a python sequence filled with the same contents as seq
        :rtype: sequence<float>
    """
    if seq is None:
        seq = []
    if not isinstance(vec, StdDoubleVector):
        raise TypeError('vec must be a tango.StdDoubleVector')
    for e in vec:
        seq.append(float(e))
    return seq


def seq_2_DbDevInfos(seq, vec=None):
    """Converts a python sequence<DbDevInfo> object to a :class:`tango.DbDevInfos`

        :param seq: the sequence of DbDevInfo
        :type seq: sequence<DbDevInfo>
        :param vec: (optional, default is None) an :class:`tango.DbDevInfos`
                    to be filled. If None is given, a new :class:`tango.DbDevInfos`
                    is created
        :return: a :class:`tango.DbDevInfos` filled with the same contents as seq
        :rtype: :class:`tango.DbDevInfos`
    """
    if vec is None:
        if isinstance(seq, DbDevInfos):
            return seq
        vec = DbDevInfos()
    if not isinstance(vec, DbDevInfos):
        raise TypeError('vec must be a tango.DbDevInfos')
    for e in seq:
        vec.append(e)
    return vec


def seq_2_DbDevExportInfos(seq, vec=None):
    """Converts a python sequence<DbDevExportInfo> object to a :class:`tango.DbDevExportInfos`

        :param seq: the sequence of DbDevExportInfo
        :type seq: sequence<DbDevExportInfo>
        :param vec: (optional, default is None) an :class:`tango.DbDevExportInfos`
                    to be filled. If None is given, a new :class:`tango.DbDevExportInfos`
                    is created
        :return: a :class:`tango.DbDevExportInfos` filled with the same contents as seq
        :rtype: :class:`tango.DbDevExportInfos`
    """
    if vec is None:
        if isinstance(seq, DbDevExportInfos):
            return seq
        vec = DbDevExportInfos()
    if not isinstance(vec, DbDevExportInfos):
        raise TypeError('vec must be a tango.DbDevExportInfos')
    for e in seq:
        vec.append(e)
    return vec


def seq_2_DbData(seq, vec=None):
    """Converts a python sequence<DbDatum> object to a :class:`tango.DbData`

        :param seq: the sequence of DbDatum
        :type seq: sequence<DbDatum>
        :param vec: (optional, default is None) an :class:`tango.DbData`
                    to be filled. If None is given, a new :class:`tango.DbData`
                    is created
        :return: a :class:`tango.DbData` filled with the same contents as seq
        :rtype: :class:`tango.DbData`
    """
    if vec is None:
        if isinstance(seq, DbData):
            return seq
        vec = DbData()
    if not isinstance(vec, DbData):
        raise TypeError('vec must be a tango.DbData')
    for e in seq:
        vec.append(e)
    return vec


def DbData_2_dict(db_data, d=None):
    if d is None:
        d = {}
    if not isinstance(db_data, DbData):
        raise TypeError(
            'db_data must be a tango.DbData. A %s found instead' %
            type(db_data))
    for db_datum in db_data:
        d[db_datum.name] = db_datum.value_string
    return d


def seqStr_2_obj(seq, tg_type, tg_format=None):
    """Translates a sequence<str> to a sequence of objects of give type and format

        :param seq: the sequence
        :type seq: sequence<str>
        :param tg_type: tango type
        :type tg_type: :class:`tango.CmdArgType`
        :param tg_format: (optional, default is None, meaning SCALAR) tango format
        :type tg_format: :class:`tango.AttrDataFormat`

        :return: a new sequence
    """
    if tg_format:
        return _seqStr_2_obj_from_type_format(seq, tg_type, tg_format)
    return _seqStr_2_obj_from_type(seq, tg_type)


def _seqStr_2_obj_from_type(seq, tg_type):
    if is_pure_str(seq):
        seq = seq,

    # Scalar cases
    global _scalar_int_types
    if tg_type in _scalar_int_types:
        return int(seq[0])

    global _scalar_float_types
    if tg_type in _scalar_float_types:
        return float(seq[0])

    global _scalar_str_types
    if tg_type in _scalar_str_types:
        return seq[0]

    if tg_type == CmdArgType.DevBoolean:
        return seq[0].lower() == 'true'

    # sequence cases
    if tg_type in (CmdArgType.DevVarCharArray, CmdArgType.DevVarStringArray):
        return seq

    global _array_int_types
    if tg_type in _array_int_types:
        argout = []
        for x in seq:
            argout.append(int(x))
        return argout

    global _array_float_types
    if tg_type in _array_float_types:
        argout = []
        for x in seq:
            argout.append(float(x))
        return argout

    if tg_type == CmdArgType.DevVarBooleanArray:
        argout = []
        for x in seq:
            argout.append(x.lower() == 'true')
        return argout

    return []


def _seqStr_2_obj_from_type_format(seq, tg_type, tg_format):
    if tg_format == AttrDataFormat.SCALAR:
        return _seqStr_2_obj_from_type(tg_type, seq)
    elif tg_format == AttrDataFormat.SPECTRUM:
        return _seqStr_2_obj_from_type(_scalar_to_array_type[tg_type], seq)
    elif tg_format == AttrDataFormat.IMAGE:
        if tg_type == CmdArgType.DevString:
            return seq

        global _scalar_int_types
        if tg_type in _scalar_int_types:
            argout = []
            for x in seq:
                tmp = []
                for y in x:
                    tmp.append(int(y))
                argout.append(tmp)
            return argout

        global _scalar_float_types
        if tg_type in _scalar_float_types:
            argout = []
            for x in seq:
                tmp = []
                for y in x:
                    tmp.append(float(y))
                argout.append(tmp)
            return argout

    # UNKNOWN_FORMAT
    return _seqStr_2_obj_from_type(tg_type, seq)


def scalar_to_array_type(tg_type):
    """
    Gives the array tango type corresponding to the given tango
    scalar type. Example: giving DevLong will return DevVarLongArray.

    :param tg_type: tango type
    :type tg_type: :class:`tango.CmdArgType`

    :return: the array tango type for the given scalar tango type
    :rtype: :class:`tango.CmdArgType`

    :raises ValueError: in case the given dtype is not a tango scalar type
    """
    try:
        return _scalar_to_array_type[tg_type]
    except KeyError:
        raise ValueError("Invalid tango scalar type: {0}".format(tg_type))


def str_2_obj(obj_str, tg_type=None):
    """Converts a string into an object according to the given tango type

           :param obj_str: the string to be converted
           :type obj_str: :py:obj:`str`
           :param tg_type: tango type
           :type tg_type: :class:`tango.CmdArgType`
           :return: an object calculated from the given string
           :rtype: :py:obj:`object`
    """
    if tg_type is None:
        return obj_str
    f = str
    if is_scalar_type(tg_type):
        if is_numerical_type(tg_type):
            if obj_str in __NO_STR_VALUE:
                return None
        if is_int_type(tg_type):
            f = int
        elif is_float_type(tg_type):
            f = float
        elif is_bool_type(tg_type):
            f = bool_
    return f(obj_str)


def obj_2_str(obj, tg_type=None):
    """Converts a python object into a string according to the given tango type

           :param obj: the object to be converted
           :type obj: :py:obj:`object`
           :param tg_type: tango type
           :type tg_type: :class:`tango.CmdArgType`
           :return: a string representation of the given object
           :rtype: :py:obj:`str`
    """
    if tg_type is None:
        return obj
    if tg_type in _scalar_types:
        # scalar cases
        if is_pure_str(obj):
            return obj
        elif is_non_str_seq(obj):
            if not len(obj):
                return ""
            obj = obj[0]
        return str(obj)
    # sequence cases
    if obj is None:
        return ''
    return '\n'.join([str(i) for i in obj])


def __get_meth_func(klass, method_name):
    meth = getattr(klass, method_name)
    func = meth
    if hasattr(meth, '__func__'):
        func = meth.__func__
    elif hasattr(meth, 'im_func'):
        func = meth.im_func
    return meth, func


def copy_doc(klass, fnname):
    """Copies documentation string of a method from the super class into the
    rewritten method of the given class"""
    base_meth, base_func = __get_meth_func(klass.__base__, fnname)
    meth, func = __get_meth_func(klass, fnname)
    func.__doc__ = base_func.__doc__


def document_method(klass, method_name, d, add=True):
    meth, func = __get_meth_func(klass, method_name)
    if add:
        cpp_doc = meth.__doc__
        if cpp_doc:
            func.__doc__ = "%s\n%s" % (d, cpp_doc)
            return
    func.__doc__ = d

    if func.__name__ != method_name:
        try:
            func.__name__ = method_name
        except AttributeError:
            pass


def document_static_method(klass, method_name, d, add=True):
    meth, func = __get_meth_func(klass, method_name)
    if add:
        cpp_doc = meth.__doc__
        if cpp_doc:
            meth.__doc__ = "%s\n%s" % (d, cpp_doc)
            return
    meth.__doc__ = d


def document_enum(klass, enum_name, desc, append=True):
    # derived = type(base)('derived', (base,), {'__doc__': 'desc'})

    # Get the original enum type
    base = getattr(klass, enum_name)

    # Prepare the new docstring
    if append and base.__doc__ is not None:
        desc = base.__doc__ + "\n" + desc

    # Create a new type, derived from the original. Only difference
    # is the docstring.
    derived = type(base)(enum_name, (base,), {'__doc__': desc})

    # Replace the original enum type with the new one
    setattr(klass, enum_name, derived)


class CaselessList(list):
    """A case insensitive lists that has some caseless methods. Only allows
    strings as list members. Most methods that would normally return a list,
    return a CaselessList. (Except list() and lowercopy())
    Sequence Methods implemented are :
    __contains__, remove, count, index, append, extend, insert,
    __getitem__, __setitem__, __getslice__, __setslice__
    __add__, __radd__, __iadd__, __mul__, __rmul__
    Plus Extra methods:
    findentry, copy , lowercopy, list
    Inherited methods :
    __imul__, __len__, __iter__, pop, reverse, sort
    """

    def __init__(self, inlist=[]):
        list.__init__(self)
        for entry in inlist:
            if not isinstance(entry, str):
                raise TypeError(
                    'Members of this object must be strings. '
                    'You supplied \"%s\" which is \"%s\"' %
                    (entry, type(entry)))
            self.append(entry)

    def findentry(self, item):
        """A caseless way of checking if an item is in the list or not.
        It returns None or the entry."""
        if not isinstance(item, str):
            raise TypeError(
                'Members of this object must be strings. '
                'You supplied \"%s\"' % type(item))
        for entry in self:
            if item.lower() == entry.lower():
                return entry
        return None

    def __contains__(self, item):
        """A caseless way of checking if a list has a member in it or not."""
        for entry in self:
            if item.lower() == entry.lower():
                return True
        return False

    def remove(self, item):
        """Remove the first occurence of an item, the caseless way."""
        for entry in self:
            if item.lower() == entry.lower():
                list.remove(self, entry)
                return
        raise ValueError(': list.remove(x): x not in list')

    def copy(self):
        """Return a CaselessList copy of self."""
        return CaselessList(self)

    def list(self):
        """Return a normal list version of self."""
        return list(self)

    def lowercopy(self):
        """Return a lowercase (list) copy of self."""
        return [entry.lower() for entry in self]

    def append(self, item):
        """Adds an item to the list and checks it's a string."""
        if not isinstance(item, str):
            raise TypeError(
                'Members of this object must be strings. '
                'You supplied \"%s\"' % type(item))
        list.append(self, item)

    def extend(self, item):
        """Extend the list with another list. Each member of the list must be
        a string."""
        if not isinstance(item, list):
            raise TypeError(
                'You can only extend lists with lists. '
                'You supplied \"%s\"' % type(item))
        for entry in item:
            if not isinstance(entry, str):
                raise TypeError(
                    'Members of this object must be strings. '
                    'You supplied \"%s\"' % type(entry))
            list.append(self, entry)

    def count(self, item):
        """Counts references to 'item' in a caseless manner.
        If item is not a string it will always return 0."""
        if not isinstance(item, str):
            return 0
        count = 0
        for entry in self:
            if item.lower() == entry.lower():
                count += 1
        return count

    def index(self, item, minindex=0, maxindex=None):
        """Provide an index of first occurence of item in the list. (or raise
        a ValueError if item not present)
        If item is not a string, will raise a TypeError.
        minindex and maxindex are also optional arguments
        s.index(x[, i[, j]]) return smallest k such that s[k] == x and i <= k < j
        """
        if maxindex is None:
            maxindex = len(self)
        minindex = max(0, minindex) - 1
        maxindex = min(len(self), maxindex)
        if not isinstance(item, str):
            raise TypeError(
                'Members of this object must be strings. '
                'You supplied \"%s\"' % type(item))
        index = minindex
        while index < maxindex:
            index += 1
            if item.lower() == self[index].lower():
                return index
        raise ValueError(': list.index(x): x not in list')

    def insert(self, i, x):
        """s.insert(i, x) same as s[i:i] = [x]
        Raises TypeError if x isn't a string."""
        if not isinstance(x, str):
            raise TypeError(
                'Members of this object must be strings. '
                'You supplied \"%s\"' % type(x))
        list.insert(self, i, x)

    def __setitem__(self, index, value):
        """For setting values in the list.
        index must be an integer or (extended) slice object. (__setslice__ used
        for simple slices)
        If index is an integer then value must be a string.
        If index is a slice object then value must be a list of strings - with
        the same length as the slice object requires.
        """
        if isinstance(index, int):
            if not isinstance(value, str):
                raise TypeError(
                    'Members of this object must be strings. '
                    'You supplied \"%s\"' % type(value))
            list.__setitem__(self, index, value)
        elif isinstance(index, slice):
            if not hasattr(value, '__len__'):
                raise TypeError(
                    'Value given to set slice is not a sequence object.')
            for entry in value:
                if not isinstance(entry, str):
                    raise TypeError(
                        'Members of this object must be strings. '
                        'You supplied \"%s\"' % type(entry))
            list.__setitem__(self, index, value)
        else:
            raise TypeError('Indexes must be integers or slice objects.')

    def __setslice__(self, i, j, sequence):
        """Called to implement assignment to self[i:j]."""
        for entry in sequence:
            if not isinstance(entry, str):
                raise TypeError(
                    'Members of this object must be strings. '
                    'You supplied \"%s\"' % type(entry))
        list.__setslice__(self, i, j, sequence)

    def __getslice__(self, i, j):
        """Called to implement evaluation of self[i:j].
        Although the manual says this method is deprecated - if I don't define
        it the list one is called.
        (Which returns a list - this returns a CaselessList)"""
        return CaselessList(list.__getslice__(self, i, j))

    def __getitem__(self, index):
        """For fetching indexes.
        If a slice is fetched then the list returned is a CaselessList."""
        if not isinstance(index, slice):
            return list.__getitem__(self, index)
        else:
            return CaselessList(list.__getitem__(self, index))

    def __add__(self, item):
        """To add a list, and return a CaselessList.
        Every element of item must be a string."""
        return CaselessList(list.__add__(self, item))

    def __radd__(self, item):
        """To add a list, and return a CaselessList.
        Every element of item must be a string."""
        return CaselessList(list.__add__(self, item))

    def __iadd__(self, item):
        """To add a list in place."""
        for entry in item:
            self.append(entry)

    def __mul__(self, item):
        """To multiply itself, and return a CaselessList.
        Every element of item must be a string."""
        return CaselessList(list.__mul__(self, item))

    def __rmul__(self, item):
        """To multiply itself, and return a CaselessList.
        Every element of item must be a string."""
        return CaselessList(list.__rmul__(self, item))


class CaselessDict(dict):
    def __init__(self, other=None):
        if other:
            # Doesn't do keyword args
            if isinstance(other, dict):
                for k, v in other.items():
                    dict.__setitem__(self, k.lower(), v)
            else:
                for k, v in other:
                    dict.__setitem__(self, k.lower(), v)

    def __getitem__(self, key):
        return dict.__getitem__(self, key.lower())

    def __setitem__(self, key, value):
        dict.__setitem__(self, key.lower(), value)

    def __contains__(self, key):
        return dict.__contains__(self, key.lower())

    def __delitem__(self, k):
        dict.__delitem__(self, k.lower())

    def has_key(self, key):
        return key.lower() in self

    def get(self, key, def_val=None):
        return dict.get(self, key.lower(), def_val)

    def setdefault(self, key, def_val=None):
        return dict.setdefault(self, key.lower(), def_val)

    def update(self, other):
        for k, v in other.items():
            dict.__setitem__(self, k.lower(), v)

    def fromkeys(self, iterable, value=None):
        d = CaselessDict()
        for k in iterable:
            dict.__setitem__(d, k.lower(), value)
        return d

    def pop(self, key, def_val=None):
        return dict.pop(self, key.lower(), def_val)

    def keys(self):
        return CaselessList(dict.keys(self))


__DEFAULT_FACT_IOR_FILE = "/tmp/rdifact.ior"
__BASE_LINE = "notifd"
__END_NOTIFD_LINE = "/DEVICE/notifd:"
__NOTIFD_FACTORY_PREFIX = "notifd/factory/"


def notifd2db(notifd_ior_file=__DEFAULT_FACT_IOR_FILE,
              files=None, host=None, out=sys.stdout):
    ior_string = ""
    with open(notifd_ior_file) as ior_file:
        ior_string = ior_file.read()

    if files is None:
        return _notifd2db_real_db(ior_string, host=host, out=out)
    else:
        return _notifd2db_file_db(ior_string, files, out=out)


def _notifd2db_file_db(ior_string, files, out=sys.stdout):
    raise RuntimeError("Not implemented yet")

    print("going to export notification service event factory to "
          "device server property file(s) ...", file=out)
    for f in files:
        with open(f, "w"):
            pass
    return


def _notifd2db_real_db(ior_string, host=None, out=sys.stdout):
    from . import Database

    print("going to export notification service event factory to "
          "Tango database ...", file=out)

    num_retries = 3
    while num_retries > 0:
        try:
            db = Database()
            db.set_timeout_millis(10000)
            num_retries = 0
        except DevFailed as df:
            num_retries -= 1
            if num_retries == 0:
                print("Can't create Tango database object", file=out)
                print(str(df), file=out)
                return
            print("Can't create Tango database object, retrying....", file=out)

    if host is None:
        import socket
        host_name = socket.getfqdn()

    global __NOTIFD_FACTORY_PREFIX
    notifd_factory_name = __NOTIFD_FACTORY_PREFIX + host_name

    args = notifd_factory_name, ior_string, host_name, str(os.getpid()), "1"

    num_retries = 3
    while num_retries > 0:
        try:
            db.command_inout("DbExportEvent", args)
            print("Successfully exported notification service event "
                  "factory for host", host_name, "to Tango database !",
                  file=out)
            break
        except CommunicationFailed as cf:
            if len(cf.errors) >= 2:
                if cf.errors[1].reason == "API_DeviceTimedOut":
                    if num_retries > 0:
                        num_retries -= 1
                else:
                    num_retries = 0
            else:
                num_retries = 0
        except Exception:
            num_retries = 0

    if num_retries == 0:
        print("Failed to export notification service event factory "
              "to TANGO database", file=out)


class EventCallback(object):
    """
    Useful event callback for test purposes

    Usage::

        >>> dev = tango.DeviceProxy(dev_name)
        >>> cb = tango.utils.EventCallback()
        >>> id = dev.subscribe_event("state", tango.EventType.CHANGE_EVENT, cb, [])
        2011-04-06 15:33:18.910474 sys/tg_test/1 STATE CHANGE [ATTR_VALID] ON

    Allowed format keys are:

        - date (event timestamp)
        - reception_date (event reception timestamp)
        - type (event type)
        - dev_name (device name)
        - name (attribute name)
        - value (event value)

    New in PyTango 7.1.4
    """

    def __init__(self, format="{date} {dev_name} {name} {type} {value}",
                 fd=sys.stdout, max_buf=100):

        self._msg = format
        self._fd = fd
        self._evts = []
        self._max_buf = max_buf

    def get_events(self):
        """Returns the list of events received by this callback

           :return: the list of events received by this callback
           :rtype: sequence<obj>
        """
        return self._evts

    def push_event(self, evt):
        """Internal usage only"""
        try:
            self._push_event(evt)
        except Exception as e:
            print("Unexpected error in callback for %s: %s"
                  % (str(evt), str(e)), file=self._fd)

    def _push_event(self, evt):
        """Internal usage only"""
        self._append(evt)
        import datetime
        now = datetime.datetime.now()
        try:
            date = self._get_date(evt)
        except:
            date = now
        try:
            reception_date = evt.reception_date.todatetime()
        except:
            reception_date = now
        try:
            evt_type = evt.event.upper()
        except:
            evt_type = "<UNKNOWN>"
        try:
            dev_name = evt.device.dev_name().upper()
        except:
            dev_name = "<UNKNOWN>"
        try:
            attr_name = evt.attr_name.split("/")[-1].upper()
        except:
            attr_name = "<UNKNOWN>"
        try:
            value = self._get_value(evt)
        except Exception as e:
            value = "Unexpected exception in getting event value: %s" % str(e)
        d = {"date": date, "reception_date": reception_date,
             "type": evt_type, "dev_name": dev_name, "name": attr_name,
             "value": value}
        print(self._msg.format(**d), file=self._fd)

    def _append(self, evt):
        """Internal usage only"""
        evts = self._evts
        if len(evts) == self._max_buf:
            evts.pop(0)
        evts.append(evt)

    def _get_date(self, evt):
        if isinstance(evt, EventData):
            return evt.attr_value.time.todatetime()
        elif isinstance(evt, PipeEventData):
            return evt.pipe_value.time.todatetime()
        else:
            return evt.get_date().todatetime()

    def _get_value(self, evt):
        """Internal usage only"""
        if evt.err:
            e = evt.errors[0]
            return "[%s] %s" % (e.reason, e.desc)

        if isinstance(evt, EventData):
            return "[%s] %s" % (
                evt.attr_value.quality, str(evt.attr_value.value))
        elif isinstance(evt, AttrConfEventData):
            cfg = evt.attr_conf
            return "label='%s'; unit='%s'" % (cfg.label, cfg.unit)
        elif isinstance(evt, DataReadyEventData):
            return ""
        elif isinstance(evt, PipeEventData):
            return evt.pipe_value
        elif isinstance(evt, DevIntrChangeEventData):
            print("utils::_get_value()")
            return


def get_home():
    """
    Find user's home directory if possible. Otherwise raise error.

    :return: user's home directory
    :rtype: :py:obj:`str`

    New in PyTango 7.1.4
    """
    path = ''
    try:
        path = os.path.expanduser("~")
    except:
        pass
    if not os.path.isdir(path):
        for evar in ('HOME', 'USERPROFILE', 'TMP'):
            try:
                path = os.environ[evar]
                if os.path.isdir(path):
                    break
            except:
                pass
    if path:
        return path
    else:
        raise RuntimeError('please define environment variable $HOME')


def _get_env_var(env_var_name):
    """
    Returns the value for the given environment name

    Search order:

        * a real environ var
        * HOME/.tangorc
        * /etc/tangorc

    :param env_var_name: the environment variable name
    :type env_var_name: str
    :return: the value for the given environment name
    :rtype: str

    New in PyTango 7.1.4
    """

    if env_var_name in os.environ:
        return os.environ[env_var_name]

    fname = os.path.join(get_home(), '.tangorc')
    if not os.path.exists(fname):
        if os.name == 'posix':
            fname = "/etc/tangorc"
    if not os.path.exists(fname):
        return None

    for line in open(fname):
        strippedline = line.split('#', 1)[0].strip()

        if not strippedline:
            # empty line
            continue

        tup = strippedline.split('=', 1)
        if len(tup) != 2:
            # illegal line!
            continue

        key, val = map(str.strip, tup)
        if key == env_var_name:
            return val


def from_version_str_to_hex_str(version_str):
    v = map(int, version_str.split('.'))
    return "0x%02d%02d%02d00" % (v[0], v[1], v[2])


def from_version_str_to_int(version_str):
    return int(from_version_str_to_hex_str(version_str), 16)


def info():
    # Compile and Runtime are set by `tango.pytango_init.init`
    from .constants import Compile, Runtime
    msg = """\
PyTango {0.version} {0.version_info}
PyTango compiled with:
    Python : {1.PY_VERSION}
    Numpy  : {1.NUMPY_VERSION}
    Tango  : {1.TANGO_VERSION}
    Boost  : {1.BOOST_VERSION}

PyTango runtime is:
    Python : {2.PY_VERSION}
    Numpy  : {2.NUMPY_VERSION}
    Tango  : {2.TANGO_VERSION}
    Boost  : {2.BOOST_VERSION}

PyTango running on:
{2.UNAME}
"""
    msg = msg.format(Release, Compile, Runtime)
    return msg


def get_attrs(obj):
    """Helper for dir2 implementation."""
    if not hasattr(obj, '__dict__'):
        return []  # slots only
    proxy_type = types.MappingProxyType if six.PY3 else types.DictProxyType
    if not isinstance(obj.__dict__, (dict, proxy_type)):
        print(type(obj.__dict__), obj)
        raise TypeError("%s.__dict__ is not a dictionary" % obj.__name__)
    return obj.__dict__.keys()


def dir2(obj):
    """Default dir implementation.

    Inspired by gist: katyukha/dirmixin.py
    https://gist.github.com/katyukha/c6e5e2b829e247c9b009
    """
    attrs = set()

    if not hasattr(obj, '__bases__'):
        # obj is an instance
        if not hasattr(obj, '__class__'):
            # slots
            return sorted(get_attrs(obj))
        klass = obj.__class__
        attrs.update(get_attrs(klass))
    else:
        # obj is a class
        klass = obj

    for cls in klass.__bases__:
        attrs.update(get_attrs(cls))
        attrs.update(dir2(cls))
    attrs.update(get_attrs(obj))
    return list(attrs)
