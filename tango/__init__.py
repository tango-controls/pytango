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
This is the main PyTango package file.
Documentation for this package can be found online:

http://pytango.readthedocs.io
"""

from __future__ import print_function

__all__ = (
    'AccessControlType', 'ApiUtil', 'ArchiveEventInfo',
    'ArchiveEventProp', 'ArgType', 'AsynCall', 'AsynReplyNotArrived', 'AttReqType',
    'Attr', 'AttrConfEventData', 'AttrData', 'AttrDataFormat', 'AttrList',
    'AttrProperty', 'AttrQuality', 'AttrReadEvent', 'AttrSerialModel',
    'AttrWriteType', 'AttrWrittenEvent', 'Attribute', 'AttributeAlarm',
    'AttributeAlarmInfo', 'AttributeConfig', 'AttributeConfig_2',
    'AttributeConfig_3', 'AttributeDimension', 'AttributeEventInfo',
    'AttributeInfo', 'AttributeInfoEx', 'AttributeInfoList', 'AttributeInfoListEx',
    'AttributeList', 'AttributeProxy', 'ChangeEventInfo', 'ChangeEventProp',
    'Pipe', 'PipeConfig', 'PipeWriteType', 'PipeEventData', 'DevIntrChangeEventData',
    'CmdArgType', 'CmdDoneEvent', 'CommandInfo', 'CommandInfoList',
    'CommunicationFailed', 'Connection', 'ConnectionFailed',
    'ConstDevString', 'DServer', 'DataReadyEventData', 'Database', 'DbData',
    'DbDatum', 'DbDevExportInfo', 'DbDevExportInfos', 'DbDevImportInfo',
    'DbDevImportInfos', 'DbDevFullInfo', 'DbDevInfo', 'DbDevInfos', 'DbHistory',
    'DbHistoryList', 'DbServerInfo', 'DbServerData', 'DebugIt', 'DevBoolean', 'DevCommandInfo',
    'DevDouble', 'DevEncoded', 'DevError', 'DevFailed', 'DevFloat', 'DevInt',
    'DevLong', 'DevLong64', 'DevShort', 'DevSource', 'DevState', 'DevString',
    'DevUChar', 'DevULong', 'DevULong64', 'DevUShort', 'DevEnum', 'DevVarBooleanArray',
    'DevVarCharArray', 'DevVarDoubleArray', 'DevVarDoubleStringArray',
    'DevVarFloatArray', 'DevVarLong64Array', 'DevVarLongArray',
    'DevVarLongStringArray', 'DevVarShortArray', 'DevVarStringArray',
    'DevVarULong64Array', 'DevVarULongArray', 'DevVarUShortArray',
    'DevVoid', 'DeviceAttribute', 'DeviceAttributeConfig',
    'DeviceAttributeHistory', 'DeviceClass', 'DeviceData', 'DeviceDataList',
    'DeviceDataHistory', 'DeviceDataHistoryList',
    'DeviceImpl', 'DeviceInfo', 'DeviceProxy', 'DeviceUnlocked',
    'DispLevel', 'EncodedAttribute', 'ErrSeverity', 'ErrorIt',
    'EventData', 'EventProperties', 'EventSystemFailed', 'EventType',
    'Except', 'FMT_UNKNOWN', 'FatalIt', 'GreenMode', 'Group',
    'GroupAttrReply',
    'GroupAttrReplyList', 'GroupCmdReply', 'GroupCmdReplyList',
    'GroupReply', 'GroupReplyList', 'IMAGE', 'ImageAttr', 'InfoIt',
    'KeepAliveCmdCode', 'Level', 'LockCmdCode', 'LockerInfo', 'LockerLanguage',
    'LogIt', 'LogLevel', 'LogTarget', 'Logger', 'Logging', 'MessBoxType',
    'MultiAttribute', 'MultiAttrProp', 'MultiClassAttribute', 'NamedDevFailed',
    'NamedDevFailedList', 'NonDbDevice', 'NonSupportedFeature',
    'NotAllowed', 'PeriodicEventInfo', 'PeriodicEventProp',
    'PollCmdCode', 'PollDevice',
    'PollObjType', 'READ', 'READ_WITH_WRITE', 'READ_WRITE', 'Release', 'SCALAR',
    'SPECTRUM', 'SerialModel', 'SpectrumAttr', 'StdDoubleVector',
    #'StdGroupAttrReplyVector', 'StdGroupCmdReplyVector', 'StdGroupReplyVector',
    'StdLongVector', 'StdNamedDevFailedVector', 'StdStringVector', 'SubDevDiag',
    'TangoStream', 'TimeVal', 'UserDefaultAttrProp', 'UserDefaultPipeProp', 'Util',
    'WAttribute', 'WRITE', 'WarnIt', 'WrongData', 'WrongNameSyntax', '__version__',
    '__version_description__', '__version_info__', '__version_long__',
    '__version_number__', 'alarm_flags', 'asyn_req_type', 'cb_sub_model',
    'class_factory', 'class_list', 'constants', 'constructed_class',
    'cpp_class_list', 'delete_class_list', 'get_class', 'get_classes',
    'get_constructed_class', 'get_constructed_classes', 'get_cpp_class',
    'get_cpp_classes', 'raise_asynch_exception', 'AutoTangoMonitor',
    'AutoTangoAllowThreads', 'LatestDeviceImpl', 'Interceptors',
    'get_attribute_proxy', 'requires_tango', 'requires_pytango',
    'set_green_mode', 'get_green_mode', 'get_device_proxy',
    'is_scalar_type', 'is_array_type', 'is_numerical_type',
    'is_int_type', 'is_float_type', 'is_bool_type', 'is_str_type',
    'obj_2_str', 'str_2_obj', #'seqStr_2_obj'
    )

__docformat__ = "restructuredtext"


# Prepare windows import

def __prepare_nt():
    import os
    import sys
    import struct

    if os.name != 'nt':
        return

    try:
        from . import _tango  # noqa: F401
    except ImportError:
        pass
    else:
        return

    PATH = os.environ.get('PATH')
    if PATH is None:
        os.environ["PATH"] = PATH = ""
    tango_root = os.environ.get("TANGO_ROOT")
    if tango_root is None:
        tango_root = os.path.join(os.environ["ProgramFiles"], "tango")
    tango_root = tango_root.lower()

    if sys.hexversion < 0x03030000:
        vc = "vc9_dll"
    else:
        vc = "vc10_dll"
    is64 = 8 * struct.calcsize("P") == 64
    if is64:
        arch = "win64"
    else:
        arch = "win32"
    tango_dll_path = os.path.join(tango_root, arch, "lib", vc)
    tango_dll_path = tango_dll_path.lower()
    if os.path.exists(tango_dll_path) and \
       tango_dll_path not in PATH.lower():
        os.environ['PATH'] += ";" + tango_dll_path
    else:
        # Tango C++ could not be found on the system...
        # ... use PyTango's private Tango C++ library
        tango_dll_path = os.path.dirname(os.path.abspath(__file__))
        tango_dll_path = os.path.join(tango_dll_path, "_tango_dll_")
        if os.path.exists(tango_dll_path):
            os.environ['PATH'] += ";" + tango_dll_path


__prepare_nt()


# Extension imports

from ._tango import AccessControlType
from ._tango import ApiUtil
from ._tango import ArchiveEventInfo
from ._tango import AsynCall
from ._tango import AsynReplyNotArrived
from ._tango import AttReqType
from ._tango import Attr
from ._tango import AttrConfEventData
from ._tango import AttrDataFormat
from ._tango import AttrList
from ._tango import AttrProperty
from ._tango import AttrQuality
from ._tango import AttrReadEvent
from ._tango import AttrSerialModel
from ._tango import AttrWriteType
from ._tango import AttrWrittenEvent
from ._tango import Attribute
from ._tango import AttributeAlarmInfo
from ._tango import AttributeDimension
from ._tango import AttributeEventInfo
from ._tango import AttributeInfo
from ._tango import AttributeInfoEx
from ._tango import AttributeInfoList
from ._tango import AttributeInfoListEx
from ._tango import AttributeList
from ._tango import AutoTangoAllowThreads
from ._tango import AutoTangoMonitor
from ._tango import ChangeEventInfo
from ._tango import CmdArgType
from ._tango import CmdDoneEvent
from ._tango import CommandInfo
from ._tango import CommandInfoList
from ._tango import CommunicationFailed
from ._tango import ConnectionFailed
#from ._tango import ConstDevString
from ._tango import DServer
from ._tango import DataReadyEventData
from ._tango import Database
from ._tango import DbData
from ._tango import DbDatum
from ._tango import DbDevExportInfo
from ._tango import DbDevExportInfos
from ._tango import DbDevFullInfo
from ._tango import DbDevImportInfo
from ._tango import DbDevImportInfos
from ._tango import DbDevInfo
from ._tango import DbDevInfos
from ._tango import DbHistory
from ._tango import DbHistoryList
from ._tango import DbServerData
from ._tango import DbServerInfo
#from ._tango import DevBoolean
from ._tango import DevCommandInfo
#from ._tango import DevDouble
#from ._tango import DevEncoded
#from ._tango import DevEnum
from ._tango import DevError
from ._tango import DevFailed
#from ._tango import DevFloat
#from ._tango import DevInt
from ._tango import DevIntrChangeEventData
#from ._tango import DevLong
#from ._tango import DevLong64
#from ._tango import DevShort
from ._tango import DevSource
from ._tango import DevState
#from ._tango import DevString
#from ._tango import DevUChar
#from ._tango import DevULong
#from ._tango import DevULong64
#from ._tango import DevUShort
#from ._tango import DevVarBooleanArray
#from ._tango import DevVarCharArray
#from ._tango import DevVarDoubleArray
#from ._tango import DevVarDoubleStringArray
#from ._tango import DevVarFloatArray
#from ._tango import DevVarLong64Array
#from ._tango import DevVarLongArray
#from ._tango import DevVarLongStringArray
#from ._tango import DevVarShortArray
#from ._tango import DevVarStringArray
#from ._tango import DevVarULong64Array
#from ._tango import DevVarULongArray
#from ._tango import DevVarUShortArray
#from ._tango import DevVoid
from ._tango import DeviceAttribute
from ._tango import DeviceAttributeConfig
from ._tango import DeviceAttributeHistory
from ._tango import DeviceData
from ._tango import DeviceDataHistory
from ._tango import DeviceDataHistoryList
from ._tango import DeviceDataList
from ._tango import DeviceImpl
from ._tango import DeviceInfo
from ._tango import DeviceProxy
from ._tango import DeviceUnlocked
from ._tango import DispLevel
#from ._tango import EncodedAttribute
from ._tango import ErrSeverity
from ._tango import EventData
from ._tango import EventSystemFailed
from ._tango import EventType
from ._tango import Except
from ._tango import FMT_UNKNOWN
from ._tango import GreenMode
from ._tango import Group
from ._tango import GroupAttrReply
from ._tango import GroupAttrReplyList
from ._tango import GroupCmdReply
from ._tango import GroupCmdReplyList
from ._tango import GroupReply
from ._tango import GroupReplyList
from ._tango import IMAGE
from ._tango import ImageAttr
from ._tango import Interceptors
from ._tango import KeepAliveCmdCode
from ._tango import Level
from ._tango import LockCmdCode
from ._tango import LockerInfo
from ._tango import LockerLanguage
from ._tango import LogLevel
from ._tango import LogTarget
from ._tango import Logger
from ._tango import Logging
from ._tango import MessBoxType
from ._tango import MultiAttribute
from ._tango import MultiClassAttribute
#from ._tango import NamedDevFailed
#from ._tango import NamedDevFailedList
from ._tango import NonDbDevice
from ._tango import NonSupportedFeature
from ._tango import NotAllowed
from ._tango import PeriodicEventInfo
from ._tango import Pipe
from ._tango import PipeEventData
from ._tango import PipeWriteType
from ._tango import PollCmdCode
from ._tango import PollDevice
from ._tango import PollObjType
from ._tango import READ
from ._tango import READ_WITH_WRITE
from ._tango import READ_WRITE
from ._tango import SCALAR
from ._tango import SPECTRUM
from ._tango import SerialModel
from ._tango import SpectrumAttr
from ._tango import StdDoubleVector
#from ._tango import StdGroupAttrReplyVector
#from ._tango import StdGroupCmdReplyVector
#from ._tango import StdGroupReplyVector
from ._tango import StdLongVector
#from ._tango import StdNamedDevFailedVector
from ._tango import StdStringVector
from ._tango import SubDevDiag
from ._tango import TimeVal
from ._tango import UserDefaultAttrProp
from ._tango import UserDefaultPipeProp
from ._tango import WAttribute
from ._tango import WRITE
from ._tango import WrongData
from ._tango import WrongNameSyntax
from ._tango import alarm_flags
from ._tango import asyn_req_type
from ._tango import cb_sub_model
from ._tango import constants
#from ._tango import raise_asynch_exception

# Aliases

ArgType = CmdArgType


# Release

from .release import Release

__author__ = Release.author_lines
__version_info__ = Release.version_info
__version__ = Release.version
__version_long__ = Release.version_long
__version_number__ = Release.version_number
__version_description__ = Release.version_description
__doc__ = Release.long_description

# Pytango imports

from .attr_data import AttrData

from .log4tango import (
    TangoStream, LogIt, DebugIt, InfoIt, WarnIt, ErrorIt, FatalIt)

from .device_server import (
    ChangeEventProp, PeriodicEventProp, ArchiveEventProp, AttributeAlarm,
    EventProperties, AttributeConfig, AttributeConfig_2, AttributeConfig_3,
    MultiAttrProp, LatestDeviceImpl)

from .pipe import PipeConfig

from .attribute_proxy import AttributeProxy, get_attribute_proxy

#from .group import Group

from .pyutil import Util

from .device_class import DeviceClass

from .globals import (
    get_class, get_classes, get_cpp_class, get_cpp_classes,
    get_constructed_class, get_constructed_classes, class_factory,
    delete_class_list, class_list, cpp_class_list, constructed_class)

from .utils import (requires_pytango, requires_tango,
#    is_scalar_type, is_array_type, is_numerical_type,
#    is_int_type, is_float_type, is_bool_type, is_str_type,
#    obj_2_str, str_2_obj, seqStr_2_obj
)

from .green import set_green_mode, get_green_mode

from .device_proxy import get_device_proxy


# Pytango initialization

from .pytango_init import init as __init
__init()
