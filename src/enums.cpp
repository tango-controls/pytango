/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
   
   PyTango is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   PyTango is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

#include <boost/python.hpp>
#include <tango.h>

using namespace boost::python;

void export_enums()
{
    enum_<Tango::LockerLanguage>("LockerLanguage")
        .value("CPP", Tango::CPP)
        .value("JAVA", Tango::JAVA)
    ;

    enum_<Tango::CmdArgType>("CmdArgType")
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
        .value("DevEncoded", Tango::DEV_ENCODED)
        .export_values()
    ;

    enum_<Tango::MessBoxType>("MessBoxType")
        .value("STOP", Tango::STOP)
        .value("INFO", Tango::INFO)
    ;

    enum_<Tango::PollObjType>("PollObjType")
        .value("POLL_CMD", Tango::POLL_CMD)
        .value("POLL_ATTR", Tango::POLL_ATTR)
        .value("EVENT_HEARTBEAT", Tango::EVENT_HEARTBEAT)
        .value("STORE_SUBDEV", Tango::STORE_SUBDEV)
    ;

    enum_<Tango::PollCmdCode>("PollCmdCode")
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

    enum_<Tango::SerialModel>("SerialModel")
        .value("BY_DEVICE",Tango::BY_DEVICE)
        .value("BY_CLASS",Tango::BY_CLASS)
        .value("BY_PROCESS",Tango::BY_PROCESS)
        .value("NO_SYNC",Tango::NO_SYNC)
    ;

    enum_<Tango::AttReqType>("AttReqType")
        .value("READ_REQ",Tango::READ_REQ)
        .value("WRITE_REQ",Tango::WRITE_REQ)
    ;

    enum_<Tango::LockCmdCode>("LockCmdCode")
        .value("LOCK_ADD_DEV", Tango::LOCK_ADD_DEV)
        .value("LOCK_REM_DEV", Tango::LOCK_REM_DEV)
        .value("LOCK_UNLOCK_ALL_EXIT", Tango::LOCK_UNLOCK_ALL_EXIT)
        .value("LOCK_EXIT", Tango::LOCK_EXIT)
    ;

#ifdef TANGO_HAS_LOG4TANGO

    enum_<Tango::LogLevel>("LogLevel")
        .value("LOG_OFF", Tango::LOG_OFF)
        .value("LOG_FATAL", Tango::LOG_FATAL)
        .value("LOG_ERROR", Tango::LOG_ERROR)
        .value("LOG_WARN", Tango::LOG_WARN)
        .value("LOG_INFO", Tango::LOG_INFO)
        .value("LOG_DEBUG", Tango::LOG_DEBUG)
    ;

    enum_<Tango::LogTarget>("LogTarget")
        .value("LOG_CONSOLE", Tango::LOG_CONSOLE)
        .value("LOG_FILE", Tango::LOG_FILE)
        .value("LOG_DEVICE", Tango::LOG_DEVICE)
    ;

#endif // TANGO_HAS_LOG4TANGO

    enum_<Tango::EventType>("EventType")
        .value("CHANGE_EVENT", Tango::CHANGE_EVENT)
        .value("QUALITY_EVENT", Tango::QUALITY_EVENT)
        .value("PERIODIC_EVENT", Tango::PERIODIC_EVENT)
        .value("ARCHIVE_EVENT", Tango::ARCHIVE_EVENT)
        .value("USER_EVENT", Tango::USER_EVENT)
        .value("ATTR_CONF_EVENT", Tango::ATTR_CONF_EVENT)
        .value("DATA_READY_EVENT", Tango::DATA_READY_EVENT)
    ;

    enum_<Tango::AttrSerialModel>("AttrSerialModel")
        .value("ATTR_NO_SYNC", Tango::ATTR_NO_SYNC)
        .value("ATTR_BY_KERNEL", Tango::ATTR_BY_KERNEL)
        .value("ATTR_BY_USER", Tango::ATTR_BY_USER)
    ;
    
    enum_<Tango::KeepAliveCmdCode>("KeepAliveCmdCode")
        .value("EXIT_TH", Tango::EXIT_TH)
    ;

    enum_<Tango::AccessControlType>("AccessControlType")
        .value("ACCESS_READ", Tango::ACCESS_READ)
        .value("ACCESS_WRITE", Tango::ACCESS_WRITE)
    ;

    enum_<Tango::asyn_req_type>("asyn_req_type")
        .value("POLLING", Tango::POLLING)
        .value("CALLBACK", Tango::CALL_BACK)
        .value("ALL_ASYNCH", Tango::ALL_ASYNCH)
    ;

    enum_<Tango::cb_sub_model>("cb_sub_model")
        .value("PUSH_CALLBACK", Tango::PUSH_CALLBACK)
        .value("PULL_CALLBACK", Tango::PULL_CALLBACK)
    ;

    //
    // Tango IDL
    //

    enum_<Tango::AttrQuality>("AttrQuality")
        .value("ATTR_VALID", Tango::ATTR_VALID)
        .value("ATTR_INVALID", Tango::ATTR_INVALID)
        .value("ATTR_ALARM", Tango::ATTR_ALARM)
        .value("ATTR_CHANGING", Tango::ATTR_CHANGING)
        .value("ATTR_WARNING", Tango::ATTR_WARNING)
    ;

    enum_<Tango::AttrWriteType>("AttrWriteType")
        .value("READ", Tango::READ)
        .value("READ_WITH_WRITE", Tango::READ_WITH_WRITE)
        .value("WRITE", Tango::WRITE)
        .value("READ_WRITE", Tango::READ_WRITE)
        .export_values()
    ;

    enum_<Tango::AttrDataFormat>("AttrDataFormat")
        .value("SCALAR", Tango::SCALAR)
        .value("SPECTRUM", Tango::SPECTRUM)
        .value("IMAGE", Tango::IMAGE)
        .value("FMT_UNKNOWN", Tango::FMT_UNKNOWN)
        .export_values()
    ;

    enum_<Tango::DevSource>("DevSource")
        .value("DEV", Tango::DEV)
        .value("CACHE", Tango::CACHE)
        .value("CACHE_DEV", Tango::CACHE_DEV)
    ;

    enum_<Tango::ErrSeverity>("ErrSeverity")
        .value("WARN", Tango::WARN)
        .value("ERR", Tango::ERR)
        .value("PANIC", Tango::PANIC)
    ;

    enum_<Tango::DevState>("DevState")
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

    enum_<Tango::DispLevel>("DispLevel")
        .value("OPERATOR", Tango::OPERATOR)
        .value("EXPERT", Tango::EXPERT)
    ;

}
