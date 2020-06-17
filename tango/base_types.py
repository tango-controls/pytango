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

__all__ = ("base_types_init",)

__docformat__ = "restructuredtext"

from ._tango import (StdStringVector, StdLongVector, StdDoubleVector,
                     CommandInfoList, AttributeInfoList, AttributeInfoListEx, DbData,
                     DbDevInfos, DbDevExportInfos, DbDevImportInfos, DbHistoryList,
                     DeviceDataHistoryList, StdGroupReplyVector,
                     StdGroupCmdReplyVector, StdGroupAttrReplyVector,
                     ArchiveEventInfo, EventData, AttrConfEventData, AttributeAlarmInfo,
                     AttributeDimension, AttributeEventInfo, DeviceAttributeConfig,
                     AttributeInfo, AttributeInfoEx, ChangeEventInfo, PeriodicEventInfo,
                     DevCommandInfo, CommandInfo, DataReadyEventData, DeviceInfo,
                     LockerInfo, PollDevice, TimeVal, AttrWriteType, AttrDataFormat, DispLevel)

from .utils import document_method, is_integer
from .utils import document_enum as __document_enum
from .utils import seq_2_StdStringVector, StdStringVector_2_seq


def __StdVector__add(self, seq):
    ret = seq.__class__(self)
    ret.extend(seq)
    return ret


def __StdVector__mul(self, n):
    ret = self.__class__()
    for _ in range(n):
        ret.extend(self)
    return ret


def __StdVector__imul(self, n):
    ret = self.__class__()
    for _ in range(n):
        ret.extend(self)
    return ret


def __StdVector__getitem(self, key):
    if is_integer(key) or key.step is None:
        return self.__original_getitem(key)

    res = self.__class__()
    nb = len(self)
    start = key.start or 0
    stop = key.stop or nb
    if start >= nb:
        return res
    if stop > nb:
        stop = nb

    for i in range(start, stop, key.step or 1):
        res.append(self[i])

    return res


def __fillVectorClass(klass):
    klass.__add__ = __StdVector__add
    klass.__mul__ = __StdVector__mul
    klass.__imul__ = __StdVector__imul
    klass.__original_getitem = klass.__getitem__
    klass.__getitem__ = __StdVector__getitem


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# DeviceAttributeConfig pickle
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def __DeviceAttributeConfig__getinitargs__(self):
    return ()


def __DeviceAttributeConfig__getstate__(self):
    return (
        self.name,
        int(self.writable),
        int(self.data_format),
        self.data_type,
        self.max_dim_x,
        self.max_dim_y,
        self.description,
        self.label,
        self.unit,
        self.standard_unit,
        self.display_unit,
        self.format,
        self.min_value,
        self.max_value,
        self.min_alarm,
        self.max_alarm,
        self.writable_attr_name,
        StdStringVector_2_seq(self.extensions))


def __DeviceAttributeConfig__setstate__(self, state):
    self.name = state[0]
    self.writable = AttrWriteType(state[1])
    self.data_format = AttrDataFormat(state[2])
    self.data_type = state[3]
    self.max_dim_x = state[4]
    self.max_dim_y = state[5]
    self.description = state[6]
    self.label = state[7]
    self.unit = state[8]
    self.standard_unit = state[9]
    self.display_unit = state[10]
    self.format = state[11]
    self.min_value = state[12]
    self.max_value = state[13]
    self.min_alarm = state[14]
    self.max_alarm = state[15]
    self.writable_attr_name = state[16]
    self.extensions = seq_2_StdStringVector(state[17])


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# AttributeInfo pickle
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def __AttributeInfo__getinitargs__(self):
    return ()


def __AttributeInfo__getstate__(self):
    ret = list(__DeviceAttributeConfig__getstate__(self))
    ret.append(int(self.disp_level))
    return tuple(ret)


def __AttributeInfo__setstate__(self, state):
    __DeviceAttributeConfig__setstate__(self, state)
    self.disp_level = DispLevel(state[18])


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# AttributeAlarmInfo pickle
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def __AttributeAlarmInfo__getinitargs__(self):
    return ()


def __AttributeAlarmInfo__getstate__(self):
    return (
        self.min_alarm,
        self.max_alarm,
        self.min_warning,
        self.max_warning,
        self.delta_t,
        self.delta_val,
        StdStringVector_2_seq(self.extensions))


def __AttributeAlarmInfo__setstate__(self, state):
    self.min_alarm = state[0]
    self.max_alarm = state[1]
    self.min_warning = state[2]
    self.max_warning = state[3]
    self.delta_t = state[4]
    self.delta_val = state[5]
    self.extensions = seq_2_StdStringVector(state[6])


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ChangeEventInfo pickle
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def __ChangeEventInfo__getinitargs__(self):
    return ()


def __ChangeEventInfo__getstate__(self):
    return (
        self.rel_change,
        self.abs_change,
        StdStringVector_2_seq(self.extensions))


def __ChangeEventInfo__setstate__(self, state):
    self.rel_change = state[0]
    self.abs_change = state[1]
    self.extensions = seq_2_StdStringVector(state[2])


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# PeriodicEventInfo pickle
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def __PeriodicEventInfo__getinitargs__(self):
    return ()


def __PeriodicEventInfo__getstate__(self):
    return (
        self.period,
        StdStringVector_2_seq(self.extensions))


def __PeriodicEventInfo__setstate__(self, state):
    self.period = state[0]
    self.extensions = seq_2_StdStringVector(state[1])


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ArchiveEventInfo pickle
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def __ArchiveEventInfo__getinitargs__(self):
    return ()


def __ArchiveEventInfo__getstate__(self):
    return (
        self.archive_rel_change,
        self.archive_abs_change,
        self.archive_period,
        StdStringVector_2_seq(self.extensions))


def __ArchiveEventInfo__setstate__(self, state):
    self.archive_rel_change = state[0]
    self.archive_abs_change = state[1]
    self.archive_period = state[2]
    self.extensions = seq_2_StdStringVector(state[3])


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# AttributeEventInfo pickle
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def __AttributeEventInfo__getinitargs__(self):
    return ()


def __AttributeEventInfo__getstate__(self):
    return (
        self.ch_event,
        self.per_event,
        self.arch_event)


def __AttributeEventInfo__setstate__(self, state):
    self.ch_event = state[0]
    self.per_event = state[1]
    self.arch_event = state[2]


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# AttributeInfoEx pickle
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def __AttributeInfoEx__getinitargs__(self):
    return ()


def __AttributeInfoEx__getstate__(self):
    ret = list(__AttributeInfo__getstate__(self))
    ret.append(self.alarms)
    ret.append(self.events)
    ret.append(StdStringVector_2_seq(self.sys_extensions))
    return tuple(ret)


def __AttributeInfoEx__setstate__(self, state):
    __AttributeInfo__setstate__(self, state)
    self.alarms = state[19]
    self.events = state[20]
    self.sys_extensions = seq_2_StdStringVector(state[21])


def __init_base_types():
    v_klasses = (
        StdStringVector, StdLongVector, StdDoubleVector, CommandInfoList,
        AttributeInfoList, AttributeInfoListEx, DbData, DbDevInfos,
        DbDevExportInfos, DbDevImportInfos, DbHistoryList,
        DeviceDataHistoryList, StdGroupReplyVector,
        StdGroupCmdReplyVector, StdGroupAttrReplyVector)

    for v_klass in v_klasses:
        __fillVectorClass(v_klass)

    DeviceAttributeConfig.__getinitargs__ = __DeviceAttributeConfig__getinitargs__
    DeviceAttributeConfig.__getstate__ = __DeviceAttributeConfig__getstate__
    DeviceAttributeConfig.__setstate__ = __DeviceAttributeConfig__setstate__

    AttributeInfo.__getinitargs__ = __AttributeInfo__getinitargs__
    AttributeInfo.__getstate__ = __AttributeInfo__getstate__
    AttributeInfo.__setstate__ = __AttributeInfo__setstate__

    AttributeAlarmInfo.__getinitargs__ = __AttributeAlarmInfo__getinitargs__
    AttributeAlarmInfo.__getstate__ = __AttributeAlarmInfo__getstate__
    AttributeAlarmInfo.__setstate__ = __AttributeAlarmInfo__setstate__

    ChangeEventInfo.__getinitargs__ = __ChangeEventInfo__getinitargs__
    ChangeEventInfo.__getstate__ = __ChangeEventInfo__getstate__
    ChangeEventInfo.__setstate__ = __ChangeEventInfo__setstate__

    PeriodicEventInfo.__getinitargs__ = __PeriodicEventInfo__getinitargs__
    PeriodicEventInfo.__getstate__ = __PeriodicEventInfo__getstate__
    PeriodicEventInfo.__setstate__ = __PeriodicEventInfo__setstate__

    ArchiveEventInfo.__getinitargs__ = __ArchiveEventInfo__getinitargs__
    ArchiveEventInfo.__getstate__ = __ArchiveEventInfo__getstate__
    ArchiveEventInfo.__setstate__ = __ArchiveEventInfo__setstate__

    AttributeEventInfo.__getinitargs__ = __AttributeEventInfo__getinitargs__
    AttributeEventInfo.__getstate__ = __AttributeEventInfo__getstate__
    AttributeEventInfo.__setstate__ = __AttributeEventInfo__setstate__

    AttributeInfoEx.__getinitargs__ = __AttributeInfoEx__getinitargs__
    AttributeInfoEx.__getstate__ = __AttributeInfoEx__getstate__
    AttributeInfoEx.__setstate__ = __AttributeInfoEx__setstate__


def __doc_base_types():
    def document_enum(enum_name, desc):
        import tango
        __document_enum(tango, enum_name, desc)

    document_enum("ExtractAs", """
    Defines what will go into value field of DeviceAttribute, or what will
    Attribute.get_write_value() return... Not all the possible values are valid
    in all the cases.

    Valid possible values are:

        - Numpy    : Value will be stored in [value, w_value]. If the
          attribute is an scalar, they will contain a value. If it's
          an SPECTRUM or IMAGE it will be exported as a numpy array.
        - Tuple    : Value will be stored in [value, w_value]. If the
          attribute is an scalar, they will contain a value. If it's
          an SPECTRUM or IMAGE it will be exported as a tuple or
          tuple of tuples.
        - List     : Value will be stored in [value, w_value]. If the
          attribute is an scalar, they will contain a value. If it's
          an SPECTRUM or IMAGE it will be exported as a list or list
          of lists
        - String   : The data will be stored 'as is', the binary data
          as it comes from TangoC++ in 'value'.
        - Nothing  : The value will not be extracted from DeviceAttribute
    """)

    document_enum("CmdArgType", """
    An enumeration representing the command argument type.

        - DevVoid
        - DevBoolean
        - DevShort
        - DevLong
        - DevFloat
        - DevDouble
        - DevUShort
        - DevULong
        - DevString
        - DevVarCharArray
        - DevVarShortArray
        - DevVarLongArray
        - DevVarFloatArray
        - DevVarDoubleArray
        - DevVarUShortArray
        - DevVarULongArray
        - DevVarStringArray
        - DevVarLongStringArray
        - DevVarDoubleStringArray
        - DevState
        - ConstDevString
        - DevVarBooleanArray
        - DevUChar
        - DevLong64
        - DevULong64
        - DevVarLong64Array
        - DevVarULong64Array
        - DevInt
        - DevEncoded
        - DevEnum
        - DevPipeBlob
    """)

    document_enum("LockerLanguage", """
    An enumeration representing the programming language in which the
    client application who locked is written.

        - CPP : C++/Python language
        - JAVA : Java language

    New in PyTango 7.0.0
    """)

    document_enum("MessBoxType", """
    An enumeration representing the MessBoxType

        - STOP
        - INFO

    New in PyTango 7.0.0
    """)

    document_enum("PollObjType", """
    An enumeration representing the PollObjType

        - POLL_CMD
        - POLL_ATTR
        - EVENT_HEARTBEAT
        - STORE_SUBDEV

    New in PyTango 7.0.0
    """)

    document_enum("PollCmdCode", """
    An enumeration representing the PollCmdCode

        - POLL_ADD_OBJ
        - POLL_REM_OBJ
        - POLL_START
        - POLL_STOP
        - POLL_UPD_PERIOD
        - POLL_REM_DEV
        - POLL_EXIT
        - POLL_REM_EXT_TRIG_OBJ
        - POLL_ADD_HEARTBEAT
        - POLL_REM_HEARTBEAT

    New in PyTango 7.0.0
    """)

    document_enum("SerialModel", """
    An enumeration representing the type of serialization performed by the device server

        - BY_DEVICE
        - BY_CLASS
        - BY_PROCESS
        - NO_SYNC
    """)

    document_enum("AttReqType", """
    An enumeration representing the type of attribute request

        - READ_REQ
        - WRITE_REQ
    """)

    document_enum("LockCmdCode", """
    An enumeration representing the LockCmdCode

        - LOCK_ADD_DEV
        - LOCK_REM_DEV
        - LOCK_UNLOCK_ALL_EXIT
        - LOCK_EXIT

    New in PyTango 7.0.0
    """)

    document_enum("LogLevel", """
    An enumeration representing the LogLevel

        - LOG_OFF
        - LOG_FATAL
        - LOG_ERROR
        - LOG_WARN
        - LOG_INFO
        - LOG_DEBUG

    New in PyTango 7.0.0
    """)

    document_enum("LogTarget", """
    An enumeration representing the LogTarget

        - LOG_CONSOLE
        - LOG_FILE
        - LOG_DEVICE

    New in PyTango 7.0.0
    """)

    document_enum("EventType", """
    An enumeration representing event type

        - CHANGE_EVENT
        - QUALITY_EVENT
        - PERIODIC_EVENT
        - ARCHIVE_EVENT
        - USER_EVENT
        - ATTR_CONF_EVENT
        - DATA_READY_EVENT
        - INTERFACE_CHANGE_EVENT
        - PIPE_EVENT

        *DATA_READY_EVENT - New in PyTango 7.0.0*
        *INTERFACE_CHANGE_EVENT - New in PyTango 9.2.2*
        *PIPE_EVENT - New in PyTango 9.2.2*

    """)

    document_enum("AttrSerialModel", """
    An enumeration representing the AttrSerialModel

        - ATTR_NO_SYNC
        - ATTR_BY_KERNEL
        - ATTR_BY_USER

    New in PyTango 7.1.0
    """)

    document_enum("KeepAliveCmdCode", """
    An enumeration representing the KeepAliveCmdCode

        - EXIT_TH

    New in PyTango 7.0.0
    """)

    document_enum("AccessControlType", """
    An enumeration representing the AccessControlType

        - ACCESS_READ
        - ACCESS_WRITE

    New in PyTango 7.0.0
    """)

    document_enum("asyn_req_type", """
    An enumeration representing the asynchronous request type

        - POLLING
        - CALLBACK
        - ALL_ASYNCH
    """)

    document_enum("cb_sub_model", """
    An enumeration representing callback sub model

        - PUSH_CALLBACK
        - PULL_CALLBACK
    """)

    document_enum("AttrQuality", """
    An enumeration representing the attribute quality

        - ATTR_VALID
        - ATTR_INVALID
        - ATTR_ALARM
        - ATTR_CHANGING
        - ATTR_WARNING
    """)

    document_enum("AttrWriteType", """
    An enumeration representing the attribute type

        - READ
        - READ_WITH_WRITE
        - WRITE
        - READ_WRITE
    """)

    document_enum("AttrDataFormat", """
    An enumeration representing the attribute format

        - SCALAR
        - SPECTRUM
        - IMAGE
        - FMT_UNKNOWN
    """)

    document_enum("PipeWriteType", """
    An enumeration representing the pipe type

        - PIPE_READ
        - PIPE_READ_WRITE
    """)

    document_enum("DevSource", """
    An enumeration representing the device source for data

        - DEV
        - CACHE
        - CACHE_DEV
    """)

    document_enum("ErrSeverity", """
    An enumeration representing the error severity

        - WARN
        - ERR
        - PANIC
    """)

    document_enum("DevState", """
    An enumeration representing the device state

        - ON
        - OFF
        - CLOSE
        - OPEN
        - INSERT
        - EXTRACT
        - MOVING
        - STANDBY
        - FAULT
        - INIT
        - RUNNING
        - ALARM
        - DISABLE
        - UNKNOWN
    """)

    document_enum("DispLevel", """
    An enumeration representing the display level

        - OPERATOR
        - EXPERT
    """)

    document_enum("GreenMode", """
    An enumeration representing the GreenMode

        - Synchronous
        - Futures
        - Gevent

    New in PyTango 8.1.0
    """)

    ArchiveEventInfo.__doc__ = """
    A structure containing available archiving event information for an attribute
    with the folowing members:

        - archive_rel_change : (str) relative change that will generate an event
        - archive_abs_change : (str) absolute change that will generate an event
        - archive_period : (str) archive period
        - extensions : (sequence<str>) extensions (currently not used)"""

    EventData.__doc__ = """
    This class is used to pass data to the callback method when an event
    is sent to the client. It contains the following public fields:

         - device : (DeviceProxy) The DeviceProxy object on which the call was
           executed.
         - attr_name : (str) The attribute name
         - event : (str) The event name
         - attr_value : (DeviceAttribute) The attribute data (DeviceAttribute)
         - err : (bool) A boolean flag set to true if the request failed. False
           otherwise
         - errors : (sequence<DevError>) The error stack
         - reception_date: (TimeVal)
    """

    document_method(EventData, "get_date", """
    get_date(self) -> TimeVal

            Returns the timestamp of the event.

        Parameters : None
        Return     : (TimeVal) the timestamp of the event

        New in PyTango 7.0.0
    """)

    AttrConfEventData.__doc__ = """
    This class is used to pass data to the callback method when a
    configuration event is sent to the client. It contains the
    following public fields:

        - device : (DeviceProxy) The DeviceProxy object on which the call was executed
        - attr_name : (str) The attribute name
        - event : (str) The event name
        - attr_conf : (AttributeInfoEx) The attribute data
        - err : (bool) A boolean flag set to true if the request failed. False
          otherwise
        - errors : (sequence<DevError>) The error stack
        - reception_date: (TimeVal)
    """

    document_method(AttrConfEventData, "get_date", """
    get_date(self) -> TimeVal

            Returns the timestamp of the event.

        Parameters : None
        Return     : (TimeVal) the timestamp of the event

        New in PyTango 7.0.0
    """)

    AttributeAlarmInfo.__doc__ = """
    A structure containing available alarm information for an attribute
    with the folowing members:

        - min_alarm : (str) low alarm level
        - max_alarm : (str) high alarm level
        - min_warning : (str) low warning level
        - max_warning : (str) high warning level
        - delta_t : (str) time delta
        - delta_val : (str) value delta
        - extensions : (StdStringVector) extensions (currently not used)"""

    AttributeDimension.__doc__ = """
    A structure containing x and y attribute data dimensions with
    the following members:

        - dim_x : (int) x dimension
        - dim_y : (int) y dimension"""

    AttributeEventInfo.__doc__ = """
    A structure containing available event information for an attribute
    with the folowing members:

        - ch_event : (ChangeEventInfo) change event information
        - per_event : (PeriodicEventInfo) periodic event information
        - arch_event :  (ArchiveEventInfo) archiving event information"""

    DeviceAttributeConfig.__doc__ = """
    A base structure containing available information for an attribute
    with the following members:

        - name : (str) attribute name
        - writable : (AttrWriteType) write type (R, W, RW, R with W)
        - data_format : (AttrDataFormat) data format (SCALAR, SPECTRUM, IMAGE)
        - data_type : (int) attribute type (float, string,..)
        - max_dim_x : (int) first dimension of attribute (spectrum or image attributes)
        - max_dim_y : (int) second dimension of attribute(image attribute)
        - description : (int) attribute description
        - label : (str) attribute label (Voltage, time, ...)
        - unit : (str) attribute unit (V, ms, ...)
        - standard_unit : (str) standard unit
        - display_unit : (str) display unit
        - format : (str) how to display the attribute value (ex: for floats could be '%6.2f')
        - min_value : (str) minimum allowed value
        - max_value : (str) maximum allowed value
        - min_alarm : (str) low alarm level
        - max_alarm : (str) high alarm level
        - writable_attr_name : (str) name of the writable attribute
        - extensions : (StdStringVector) extensions (currently not used)"""

    AttributeInfo.__doc__ = """
    A structure (inheriting from :class:`DeviceAttributeConfig`) containing
    available information for an attribute with the following members:

        - disp_level : (DispLevel) display level (OPERATOR, EXPERT)

        Inherited members are:

            - name : (str) attribute name
            - writable : (AttrWriteType) write type (R, W, RW, R with W)
            - data_format : (AttrDataFormat) data format (SCALAR, SPECTRUM, IMAGE)
            - data_type : (int) attribute type (float, string,..)
            - max_dim_x : (int) first dimension of attribute (spectrum or image attributes)
            - max_dim_y : (int) second dimension of attribute(image attribute)
            - description : (int) attribute description
            - label : (str) attribute label (Voltage, time, ...)
            - unit : (str) attribute unit (V, ms, ...)
            - standard_unit : (str) standard unit
            - display_unit : (str) display unit
            - format : (str) how to display the attribute value (ex: for floats could be '%6.2f')
            - min_value : (str) minimum allowed value
            - max_value : (str) maximum allowed value
            - min_alarm : (str) low alarm level
            - max_alarm : (str) high alarm level
            - writable_attr_name : (str) name of the writable attribute
            - extensions : (StdStringVector) extensions (currently not used)"""

    AttributeInfoEx.__doc__ = """
    A structure (inheriting from :class:`AttributeInfo`) containing
    available information for an attribute with the following members:

        - alarms : object containing alarm information (see AttributeAlarmInfo).
        - events : object containing event information (see AttributeEventInfo).
        - sys_extensions : StdStringVector

        Inherited members are:

            - name : (str) attribute name
            - writable : (AttrWriteType) write type (R, W, RW, R with W)
            - data_format : (AttrDataFormat) data format (SCALAR, SPECTRUM, IMAGE)
            - data_type : (int) attribute type (float, string,..)
            - max_dim_x : (int) first dimension of attribute (spectrum or image attributes)
            - max_dim_y : (int) second dimension of attribute(image attribute)
            - description : (int) attribute description
            - label : (str) attribute label (Voltage, time, ...)
            - unit : (str) attribute unit (V, ms, ...)
            - standard_unit : (str) standard unit
            - display_unit : (str) display unit
            - format : (str) how to display the attribute value (ex: for floats could be '%6.2f')
            - min_value : (str) minimum allowed value
            - max_value : (str) maximum allowed value
            - min_alarm : (str) low alarm level
            - max_alarm : (str) high alarm level
            - writable_attr_name : (str) name of the writable attribute
            - extensions : (StdStringVector) extensions (currently not used)
            - disp_level : (DispLevel) display level (OPERATOR, EXPERT)"""

    ChangeEventInfo.__doc__ = """
    A structure containing available change event information for an attribute
    with the folowing members:

        - rel_change : (str) relative change that will generate an event
        - abs_change : (str) absolute change that will generate an event
        - extensions : (StdStringVector) extensions (currently not used)"""

    PeriodicEventInfo.__doc__ = """
    A structure containing available periodic event information for an attribute
    with the folowing members:

        - period : (str) event period
        - extensions : (StdStringVector) extensions (currently not used)"""

    DevCommandInfo.__doc__ = """
    A device command info with the following members:

        - cmd_name : (str) command name
        - cmd_tag : command as binary value (for TACO)
        - in_type : (CmdArgType) input type
        - out_type : (CmdArgType) output type
        - in_type_desc : (str) description of input type
        - out_type_desc : (str) description of output type

    New in PyTango 7.0.0"""

    CommandInfo.__doc__ = """
    A device command info (inheriting from :class:`DevCommandInfo`) with the following members:

        - disp_level : (DispLevel) command display level

        Inherited members are (from :class:`DevCommandInfo`):

            - cmd_name : (str) command name
            - cmd_tag : (str) command as binary value (for TACO)
            - in_type : (CmdArgType) input type
            - out_type : (CmdArgType) output type
            - in_type_desc : (str) description of input type
            - out_type_desc : (str) description of output type"""

    DataReadyEventData.__doc__ = """
    This class is used to pass data to the callback method when an
    attribute data ready event is sent to the clien. It contains the
    following public fields:

        - device : (DeviceProxy) The DeviceProxy object on which the call was executed
        - attr_name : (str) The attribute name
        - event : (str) The event name
        - attr_data_type : (int) The attribute data type
        - ctr : (int) The user counter. Set to 0 if not defined when sent by the
          server
        - err : (bool) A boolean flag set to true if the request failed. False
          otherwise
        - errors : (sequence<DevError>) The error stack
        - reception_date: (TimeVal)

        New in PyTango 7.0.0"""

    DeviceInfo.__doc__ = """
    A structure containing available information for a device with the"
    following members:

        - dev_class : (str) device class
        - server_id : (str) server ID
        - server_host : (str) host name
        - server_version : (str) server version
        - doc_url : (str) document url"""

    LockerInfo.__doc__ = """
    A structure with information about the locker with the folowing members:

        - ll : (tango.LockerLanguage) the locker language
        - li : (pid_t / UUID) the locker id
        - locker_host : (str) the host
        - locker_class : (str) the class

        pid_t should be an int, UUID should be a tuple of four numbers.

        New in PyTango 7.0.0"""

    PollDevice.__doc__ = """
    A structure containing PollDevice information with the folowing members:

        - dev_name : (str) device name
        - ind_list : (sequence<int>) index list

        New in PyTango 7.0.0"""

    document_method(DataReadyEventData, "get_date", """
    get_date(self) -> TimeVal

            Returns the timestamp of the event.

        Parameters : None
        Return     : (TimeVal) the timestamp of the event

        New in PyTango 7.0.0
    """)

    TimeVal.__doc__ = """
    Time value structure with the following members:

        - tv_sec : seconds
        - tv_usec : microseconds
        - tv_nsec : nanoseconds"""


def base_types_init(doc=True):
    __init_base_types()
    if doc:
        __doc_base_types()
