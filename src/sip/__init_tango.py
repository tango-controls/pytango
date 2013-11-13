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

import collections

def __inc_param(obj, name):
    ret = not name.startswith('_')
    ret &= not name in ('except_flags',)
    ret &= not isinstance(getattr(obj, name), collections.Callable)
    return ret

def __single_param(obj, param_name, f=repr, fmt='%s = %s'):
    param_value = getattr(obj, param_name)
    return fmt % (param_name, f(param_value))

def __struct_params_s(obj, separator=', ', f=repr, fmt='%s = %s'):
    """method wrapper for printing all elements of a struct"""
    s = separator.join([__single_param(obj, n, f, fmt) for n in dir(obj) if __inc_param(obj, n)])
    return s

def __struct_params_repr(obj):
    """method wrapper for representing all elements of a struct"""
    return __struct_params_s(obj)

def __struct_params_str(obj, fmt, f=repr):
    """method wrapper for printing all elements of a struct."""
    return __struct_params_s(obj, '\n', f=f, fmt=fmt)

def __repr__Struct(self):
    """repr method for struct"""
    return '%s(%s)' % (self.__class__.__name__, __struct_params_repr(self))

def __str__Struct_Helper(self, f=repr):
    """str method for struct"""
    attrs = [ n for n in dir(self) if __inc_param(self, n)]
    fmt = attrs and '%%%ds=%%s' % max(map(len, attrs)) or "%s = %s"
    return '%s(\n%s)\n' % (self.__class__.__name__, __struct_params_str(self, fmt, f))

def __str__Struct(self):
    return __str__Struct_Helper(self, f=repr)

def __registerStructStr(Tango):
    """helper method to register str and repr methods for structures"""
    structs = (Tango.DeviceInfo, Tango.DbDevImportInfo, Tango.DbDatum,
        Tango.AttributeInfo)

    for struct in structs:
        struct.__str__ = __str__Struct
        struct.__repr__ = __repr__Struct

def __pprint_init(Tango):
    __registerStructStr(Tango)


def init(Tango, Tangodict):
    __pprint_init(Tango)
    return 1
