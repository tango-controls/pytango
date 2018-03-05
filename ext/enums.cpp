/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void export_enums(py::module &m) {
    py::enum_<Tango::LockerLanguage>(m, "LockerLanguage")
        .value("CPP", Tango::CPP)
        .value("JAVA", Tango::JAVA)
    ;

    py::enum_<Tango::CmdArgType>(m, "CmdArgType")
        .value(Tango::CmdArgTypeName[Tango::DEV_VOID], Tango::DEV_VOID)
        .value(Tango::CmdArgTypeName[Tango::DEV_BOOLEAN], Tango::DEV_BOOLEAN)
        .value(Tango::CmdArgTypeName[Tango::DEV_SHORT], Tango::DEV_SHORT)
        .value(Tango::CmdArgTypeName[Tango::DEV_LONG], Tango::DEV_LONG)
        .value(Tango::CmdArgTypeName[Tango::DEV_FLOAT], Tango::DEV_FLOAT)
        .value(Tango::CmdArgTypeName[Tango::DEV_DOUBLE], Tango::DEV_DOUBLE)
        .value(Tango::CmdArgTypeName[Tango::DEV_USHORT], Tango::DEV_USHORT)
        .value(Tango::CmdArgTypeName[Tango::DEV_ULONG], Tango::DEV_ULONG)
        .value(Tango::CmdArgTypeName[Tango::DEV_STRING], Tango::DEV_STRING)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_CHARARRAY], Tango::DEVVAR_CHARARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_SHORTARRAY], Tango::DEVVAR_SHORTARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_LONGARRAY], Tango::DEVVAR_LONGARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_FLOATARRAY], Tango::DEVVAR_FLOATARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_DOUBLEARRAY], Tango::DEVVAR_DOUBLEARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_USHORTARRAY], Tango::DEVVAR_USHORTARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_ULONGARRAY], Tango::DEVVAR_ULONGARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_STRINGARRAY], Tango::DEVVAR_STRINGARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_LONGSTRINGARRAY], Tango::DEVVAR_LONGSTRINGARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_DOUBLESTRINGARRAY], Tango::DEVVAR_DOUBLESTRINGARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEV_STATE], Tango::DEV_STATE)
        .value(Tango::CmdArgTypeName[Tango::CONST_DEV_STRING], Tango::CONST_DEV_STRING)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_BOOLEANARRAY], Tango::DEVVAR_BOOLEANARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEV_UCHAR], Tango::DEV_UCHAR)
        .value(Tango::CmdArgTypeName[Tango::DEV_LONG64], Tango::DEV_LONG64)
        .value(Tango::CmdArgTypeName[Tango::DEV_ULONG64], Tango::DEV_ULONG64)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_LONG64ARRAY], Tango::DEVVAR_LONG64ARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_ULONG64ARRAY], Tango::DEVVAR_ULONG64ARRAY)
        .value(Tango::CmdArgTypeName[Tango::DEV_INT], Tango::DEV_INT)
        .value(Tango::CmdArgTypeName[Tango::DEV_ENCODED], Tango::DEV_ENCODED)
        .value(Tango::CmdArgTypeName[Tango::DEV_ENUM], Tango::DEV_ENUM)
        .value(Tango::CmdArgTypeName[Tango::DEV_PIPE_BLOB], Tango::DEV_PIPE_BLOB)
        .value(Tango::CmdArgTypeName[Tango::DEVVAR_STATEARRAY], Tango::DEVVAR_STATEARRAY)
    ;

    py::enum_<Tango::MessBoxType>(m, "MessBoxType")
        .value("STOP", Tango::STOP)
        .value("INFO", Tango::INFO)
    ;

    py::enum_<Tango::PollObjType>(m, "PollObjType")
        .value("POLL_CMD", Tango::POLL_CMD)
        .value("POLL_ATTR", Tango::POLL_ATTR)
        .value("EVENT_HEARTBEAT", Tango::EVENT_HEARTBEAT)
        .value("STORE_SUBDEV", Tango::STORE_SUBDEV)
    ;

    py::enum_<Tango::PollCmdCode>(m, "PollCmdCode")
        .value("POLL_ADD_OBJ", Tango::POLL_ADD_OBJ)
        .value("POLL_REM_OBJ", Tango::POLL_REM_OBJ)
        .value("POLL_START", Tango::POLL_START)
        .value("POLL_STOP", Tango::POLL_STOP)
        .value("POLL_UPD_PERIOD", Tango::POLL_UPD_PERIOD)
        .value("POLL_REM_DEV", Tango::POLL_REM_DEV)
        .value("POLL_EXIT", Tango::POLL_EXIT)
        .value("POLL_REM_EXT_TRIG_OBJ", Tango::POLL_REM_EXT_TRIG_OBJ)
        .value("POLL_ADD_HEARTBEAT", Tango::POLL_ADD_HEARTBEAT)
        .value("POLL_REM_HEARTBEAT", Tango::POLL_REM_HEARTBEAT)
    ;

    py::enum_<Tango::SerialModel>(m, "SerialModel")
        .value("BY_DEVICE",Tango::BY_DEVICE)
        .value("BY_CLASS",Tango::BY_CLASS)
        .value("BY_PROCESS",Tango::BY_PROCESS)
        .value("NO_SYNC",Tango::NO_SYNC)
    ;

    py::enum_<Tango::AttReqType>(m, "AttReqType")
        .value("READ_REQ",Tango::READ_REQ)
        .value("WRITE_REQ",Tango::WRITE_REQ)
    ;

    py::enum_<Tango::LockCmdCode>(m, "LockCmdCode")
        .value("LOCK_ADD_DEV", Tango::LOCK_ADD_DEV)
        .value("LOCK_REM_DEV", Tango::LOCK_REM_DEV)
        .value("LOCK_UNLOCK_ALL_EXIT", Tango::LOCK_UNLOCK_ALL_EXIT)
        .value("LOCK_EXIT", Tango::LOCK_EXIT)
    ;

#ifdef TANGO_HAS_LOG4TANGO

    py::enum_<Tango::LogLevel>(m, "LogLevel")
        .value("LOG_OFF", Tango::LOG_OFF)
        .value("LOG_FATAL", Tango::LOG_FATAL)
        .value("LOG_ERROR", Tango::LOG_ERROR)
        .value("LOG_WARN", Tango::LOG_WARN)
        .value("LOG_INFO", Tango::LOG_INFO)
        .value("LOG_DEBUG", Tango::LOG_DEBUG)
    ;

    py::enum_<Tango::LogTarget>(m, "LogTarget")
        .value("LOG_CONSOLE", Tango::LOG_CONSOLE)
        .value("LOG_FILE", Tango::LOG_FILE)
        .value("LOG_DEVICE", Tango::LOG_DEVICE)
    ;

#endif // TANGO_HAS_LOG4TANGO

    py::enum_<Tango::EventType>(m, "EventType")
        .value("CHANGE_EVENT", Tango::CHANGE_EVENT)
        .value("QUALITY_EVENT", Tango::QUALITY_EVENT)
        .value("PERIODIC_EVENT", Tango::PERIODIC_EVENT)
        .value("ARCHIVE_EVENT", Tango::ARCHIVE_EVENT)
        .value("USER_EVENT", Tango::USER_EVENT)
        .value("ATTR_CONF_EVENT", Tango::ATTR_CONF_EVENT)
        .value("DATA_READY_EVENT", Tango::DATA_READY_EVENT)
        .value("INTERFACE_CHANGE_EVENT", Tango::INTERFACE_CHANGE_EVENT)
        .value("PIPE_EVENT", Tango::PIPE_EVENT)
    ;

    py::enum_<Tango::AttrSerialModel>(m, "AttrSerialModel")
        .value("ATTR_NO_SYNC", Tango::ATTR_NO_SYNC)
        .value("ATTR_BY_KERNEL", Tango::ATTR_BY_KERNEL)
        .value("ATTR_BY_USER", Tango::ATTR_BY_USER)
    ;
    
    py::enum_<Tango::KeepAliveCmdCode>(m, "KeepAliveCmdCode")
        .value("EXIT_TH", Tango::EXIT_TH)
    ;

    py::enum_<Tango::AccessControlType>(m, "AccessControlType")
        .value("ACCESS_READ", Tango::ACCESS_READ)
        .value("ACCESS_WRITE", Tango::ACCESS_WRITE)
    ;

    py::enum_<Tango::asyn_req_type>(m, "asyn_req_type")
        .value("POLLING", Tango::POLLING)
        .value("CALLBACK", Tango::CALL_BACK)
        .value("ALL_ASYNCH", Tango::ALL_ASYNCH)
    ;

    py::enum_<Tango::cb_sub_model>(m, "cb_sub_model")
        .value("PUSH_CALLBACK", Tango::PUSH_CALLBACK)
        .value("PULL_CALLBACK", Tango::PULL_CALLBACK)
    ;

    //
    // Tango IDL
    //

    py::enum_<Tango::AttrQuality>(m, "AttrQuality")
        .value("ATTR_VALID", Tango::ATTR_VALID)
        .value("ATTR_INVALID", Tango::ATTR_INVALID)
        .value("ATTR_ALARM", Tango::ATTR_ALARM)
        .value("ATTR_CHANGING", Tango::ATTR_CHANGING)
        .value("ATTR_WARNING", Tango::ATTR_WARNING)
    ;

    py::enum_<Tango::AttrWriteType>(m, "AttrWriteType")
        .value("READ", Tango::READ)
        .value("READ_WITH_WRITE", Tango::READ_WITH_WRITE)
        .value("WRITE", Tango::WRITE)
        .value("READ_WRITE", Tango::READ_WRITE)
        .value("WT_UNKNOWN", Tango::READ_WRITE)
        .export_values()
    ;

    py::enum_<Tango::AttrDataFormat>(m, "AttrDataFormat")
        .value("SCALAR", Tango::SCALAR)
        .value("SPECTRUM", Tango::SPECTRUM)
        .value("IMAGE", Tango::IMAGE)
        .value("FMT_UNKNOWN", Tango::FMT_UNKNOWN)
        .export_values()
    ;

    py::enum_<Tango::DevSource>(m, "DevSource")
        .value("DEV", Tango::DEV)
        .value("CACHE", Tango::CACHE)
        .value("CACHE_DEV", Tango::CACHE_DEV)
    ;

    py::enum_<Tango::ErrSeverity>(m, "ErrSeverity")
        .value("WARN", Tango::WARN)
        .value("ERR", Tango::ERR)
        .value("PANIC", Tango::PANIC)
    ;

    py::enum_<Tango::DevState>(m, "DevState")
        .value(Tango::DevStateName[Tango::ON], Tango::ON)
        .value(Tango::DevStateName[Tango::OFF], Tango::OFF)
        .value(Tango::DevStateName[Tango::CLOSE], Tango::CLOSE)
        .value(Tango::DevStateName[Tango::OPEN], Tango::OPEN)
        .value(Tango::DevStateName[Tango::INSERT], Tango::INSERT)
        .value(Tango::DevStateName[Tango::EXTRACT], Tango::EXTRACT)
        .value(Tango::DevStateName[Tango::MOVING], Tango::MOVING)
        .value(Tango::DevStateName[Tango::STANDBY], Tango::STANDBY)
        .value(Tango::DevStateName[Tango::FAULT], Tango::FAULT)
        .value(Tango::DevStateName[Tango::INIT], Tango::INIT)
        .value(Tango::DevStateName[Tango::RUNNING], Tango::RUNNING)
        .value(Tango::DevStateName[Tango::ALARM], Tango::ALARM)
        .value(Tango::DevStateName[Tango::DISABLE], Tango::DISABLE)
        .value(Tango::DevStateName[Tango::UNKNOWN], Tango::UNKNOWN)
    ;

    py::enum_<Tango::DispLevel>(m, "DispLevel")
        .value("OPERATOR", Tango::OPERATOR)
        .value("EXPERT", Tango::EXPERT)
        .value("DL_UNKNOWN", Tango::DL_UNKNOWN)
    ;

    py::enum_<Tango::PipeWriteType>(m, "PipeWriteType")
        .value("PIPE_READ", Tango::PIPE_READ)
        .value("PIPE_READ_WRITE", Tango::PIPE_READ_WRITE)
        .value("PIPE_WT_UNKNOWN", Tango::PIPE_WT_UNKNOWN)
    ;

    py::enum_<Tango::PipeSerialModel>(m, "PipeSerialModel")
        .value("PIPE_NO_SYNC", Tango::PIPE_NO_SYNC)
        .value("PIPE_BY_KERNEL", Tango::PIPE_BY_KERNEL)
        .value("PIPE_BY_USER", Tango::PIPE_BY_USER)
    ;

//    m.attr("PipeReqType") = m.attr("AttrReqType");

    py::enum_<Tango::AttrMemorizedType>(m, "AttrMemorizedType")
        .value("NOT_KNOWN", Tango::NOT_KNOWN)
        .value("NONE", Tango::NONE)
        .value("MEMORIZED", Tango::MEMORIZED)
        .value("MEMORIZED_WRITE_INIT", Tango::MEMORIZED_WRITE_INIT)
    ;
}
