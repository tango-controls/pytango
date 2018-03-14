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

    PATH = os.environ.get(PATH)
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
        os.environ[PATH] += ";" + tango_dll_path
    else:
        # Tango C++ could not be found on the system...
        # ... use PyTangos private Tango C++ library
        tango_dll_path = os.path.dirname(os.path.abspath(__file__))
        tango_dll_path = os.path.join(tango_dll_path, "_tango_dll_")
        if os.path.exists(tango_dll_path):
            os.environ[PATH] += ";" + tango_dll_path


__prepare_nt()


# Extension imports

from ._tango import (
AccessControlType, ApiUtil, ArchiveEventInfo, AttReqType, Attr, AttrConfEventData,
AttrDataFormat, AttrList, AttrMemorizedType, AttrProperty, AttrQuality, AttrSerialModel,
AttrWriteType, Attribute, AttributeAlarmInfo, AttributeDimension, AttributeEventInfo,
AttributeInfo, AttributeInfoEx, AttributeInfoList, AttributeInfoListEx, AttributeList,
AutoTangoAllowThreads, AutoTangoMonitor, ChangeEventInfo, CmdArgType, CommandInfo,
CommandInfoList, Connection, DataReadyEventData, Database, DbData, DbDatum, DbDevExportInfo,
DbDevExportInfos, DbDevFullInfo, DbDevImportInfo, DbDevImportInfos, DbDevInfo, DbDevInfos,
DbHistory, DbHistoryList, DbServerData, DbServerInfo, DevCommandInfo, DevError, DevSource,
DevState, DeviceAttribute, DeviceAttributeConfig, DeviceAttributeHistory, DeviceData,
DeviceDataHistory, DeviceDataHistoryList, DeviceDataList, DeviceImpl, DeviceInfo, DevicePipe,
DeviceProxy, DispLevel, ErrSeverity, EventData, EventType, ExtractAs, FMT_UNKNOWN,
FwdAttr, GreenMode, Group, IMAGE, ImageAttr, Interceptors, KeepAliveCmdCode, Level,
LevelLevel, LockCmdCode, LockerInfo, LockerLanguage, LockingThread, LogLevel, LogTarget,
Logger, Logging, MessBoxType, MultiAttribute, MultiClassAttribute, PeriodicEventInfo, Pipe,
PipeInfo, PipeInfoList, PipeList, PipeSerialModel, PipeWriteType, PollCmdCode, PollDevice,
PollObjType, READ, READ_WITH_WRITE, READ_WRITE, SCALAR, SPECTRUM, SerialModel, SpectrumAttr,
StdDoubleVector, StdGroupAttrReplyVector, StdGroupCmdReplyVector, StdGroupReplyVector, StdLongVector,
StdStringVector, SubDevDiag, TimeVal, UserDefaultAttrProp, UserDefaultFwdAttrProp,
UserDefaultPipeProp, Util, WAttribute, WPipe, WRITE, WT_UNKNOWN, _ImageFormat,
__AttributeProxy, __tangolib_version__, alarm_flags,
asyn_req_type, cb_sub_model, constants, DevFailed
)

# Aliases

# ArgType = CmdArgType


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

# from .attr_data import AttrData

from .log4tango import (
    TangoStream, LogIt, DebugIt, InfoIt, WarnIt, ErrorIt, FatalIt)

# from .device_server import (
#     ChangeEventProp, PeriodicEventProp, ArchiveEventProp, AttributeAlarm,
#     EventProperties, AttributeConfig, AttributeConfig_2, AttributeConfig_3,
#     MultiAttrProp, LatestDeviceImpl)
# 
# from .pipe import PipeConfig
# 
# from .attribute_proxy import AttributeProxy, get_attribute_proxy
# 
#from .group import Group

from .pyutil import Util

#from .device_class import DeviceClass

from .globals import (
    get_class, get_classes, get_cpp_class, get_cpp_classes,
    get_constructed_class, get_constructed_classes, class_factory,
    delete_class_list, class_list, cpp_class_list, constructed_class)

# from .utils import requires_pytango, requires_tango

from .green import set_green_mode, get_green_mode

from .device_proxy import get_device_proxy

# Pytango initialization
from .pytango_init import init as __init
__init()
