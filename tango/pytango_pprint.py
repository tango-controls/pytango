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

__all__ = ("pytango_pprint_init",)

__docformat__ = "restructuredtext"

from ._tango import (StdStringVector, StdLongVector, CommandInfoList,
                     AttributeInfoList, AttributeInfoListEx, PipeInfoList,
                     DeviceDataHistoryList,
                     GroupReplyList, GroupAttrReplyList, GroupCmdReplyList,
                     DbData, DbDevInfos, DbDevExportInfos, DbDevImportInfos, DbHistoryList,
                     LockerInfo, DevCommandInfo, AttributeDimension, CommandInfo, PipeInfo,
                     DeviceInfo, DeviceAttributeConfig, AttributeInfo, AttributeAlarmInfo,
                     ChangeEventInfo, PeriodicEventInfo, ArchiveEventInfo,
                     AttributeEventInfo, AttributeInfoEx,
                     DeviceAttribute, DeviceAttributeHistory, DeviceData, DeviceDataHistory,
                     DevicePipe, DbDatum, DbDevInfo, DbDevImportInfo, DbDevExportInfo,
                     DbServerInfo, GroupReply, GroupAttrReply, GroupCmdReply,
                     DevError, EventData, AttrConfEventData, DataReadyEventData,
                     TimeVal, DevFailed, CmdArgType)

from .device_server import AttributeAlarm, EventProperties
from .device_server import ChangeEventProp, PeriodicEventProp, ArchiveEventProp
from .device_server import AttributeConfig, AttributeConfig_2
from .device_server import AttributeConfig_3, AttributeConfig_5
import collections


def __inc_param(obj, name):
    ret = not name.startswith('_')
    ret &= name not in ('except_flags',)
    ret &= not isinstance(getattr(obj, name), collections.Callable)
    return ret


def __single_param(obj, param_name, f=repr, fmt='%s = %s'):
    param_value = getattr(obj, param_name)
    if param_name is 'data_type':
        param_value = CmdArgType.values.get(param_value, param_value)
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
    attrs = [n for n in dir(self) if __inc_param(self, n)]
    fmt = attrs and '%%%ds = %%s' % max(map(len, attrs)) or "%s = %s"
    return '%s[\n%s]\n' % (self.__class__.__name__, __struct_params_str(self, fmt, f))


def __str__Struct(self):
    return __str__Struct_Helper(self, f=repr)


def __str__Struct_extra(self):
    return __str__Struct_Helper(self, f=str)


def __registerSeqStr():
    """helper function to make internal sequences printable"""
    _SeqStr = lambda self: (self and "[%s]" % (", ".join(map(repr, self)))) or "[]"
    _SeqRepr = lambda self: (self and "[%s]" % (", ".join(map(repr, self)))) or "[]"

    seqs = (StdStringVector, StdLongVector, CommandInfoList,
            AttributeInfoList, AttributeInfoListEx, PipeInfoList,
            DeviceDataHistoryList,
            GroupReplyList, GroupAttrReplyList, GroupCmdReplyList,
            DbData, DbDevInfos, DbDevExportInfos, DbDevImportInfos, DbHistoryList)

    for seq in seqs:
        seq.__str__ = _SeqStr
        seq.__repr__ = _SeqRepr


def __str__DevFailed(self):
    if isinstance(self.args, collections.Sequence):
        return 'DevFailed[\n%s]' % '\n'.join(map(str, self.args))
    return 'DevFailed[%s]' % (self.args)


def __repr__DevFailed(self):
    return 'DevFailed(args = %s)' % repr(self.args)


def __str__DevError(self):
    desc = self.desc.replace("\n", "\n           ")
    s = """DevError[
    desc = %s
  origin = %s
  reason = %s
severity = %s]\n""" % (desc, self.origin, self.reason, self.severity)
    return s


def __registerStructStr():
    """helper method to register str and repr methods for structures"""
    structs = (LockerInfo, DevCommandInfo, AttributeDimension, CommandInfo,
               DeviceInfo, DeviceAttributeConfig, AttributeInfo, AttributeAlarmInfo,
               ChangeEventInfo, PeriodicEventInfo, ArchiveEventInfo,
               AttributeEventInfo, AttributeInfoEx, PipeInfo,
               DeviceAttribute, DeviceAttributeHistory, DeviceData, DeviceDataHistory,
               DevicePipe, DbDatum, DbDevInfo, DbDevImportInfo, DbDevExportInfo,
               DbServerInfo, GroupReply, GroupAttrReply, GroupCmdReply,
               DevError, EventData, AttrConfEventData, DataReadyEventData,
               AttributeConfig, AttributeConfig_2, AttributeConfig_3,
               AttributeConfig_5,
               ChangeEventProp, PeriodicEventProp, ArchiveEventProp,
               AttributeAlarm, EventProperties)

    for struct in structs:
        struct.__str__ = __str__Struct
        struct.__repr__ = __repr__Struct

    # special case for TimeVal: it already has a str representation itself
    TimeVal.__repr__ = __repr__Struct

    # special case for DevFailed: we want a better pretty print
    # also, because it is an Exception it has the message attribute which
    # generates a Deprecation warning in python 2.6
    DevFailed.__str__ = __str__DevFailed
    DevFailed.__repr__ = __repr__DevFailed

    DevError.__str__ = __str__DevError


def pytango_pprint_init(doc=True):
    __registerSeqStr()
    __registerStructStr()
