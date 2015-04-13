/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

%ModuleHeaderCode
#include <tango.h>
using namespace Tango;
%End

enum CmdArgType {
	DEV_VOID,
	DEV_BOOLEAN,
	DEV_SHORT,
	DEV_LONG,
	DEV_FLOAT,
	DEV_DOUBLE,
	DEV_USHORT,
	DEV_ULONG,
	DEV_STRING,
	DEVVAR_CHARARRAY,
	DEVVAR_SHORTARRAY,
	DEVVAR_LONGARRAY,
	DEVVAR_FLOATARRAY,
	DEVVAR_DOUBLEARRAY,
	DEVVAR_USHORTARRAY,
	DEVVAR_ULONGARRAY,
	DEVVAR_STRINGARRAY,
	DEVVAR_LONGSTRINGARRAY,
	DEVVAR_DOUBLESTRINGARRAY,
	DEV_STATE,
	CONST_DEV_STRING,
	DEVVAR_BOOLEANARRAY,
	DEV_UCHAR,
	DEV_LONG64,
	DEV_ULONG64,
	DEVVAR_LONG64ARRAY,
	DEVVAR_ULONG64ARRAY,
	DEV_INT,
	DEV_ENCODED
};

enum MessBoxType {
        STOP,
        INFO
};

enum PollObjType {
	POLL_CMD,
	POLL_ATTR,
	EVENT_HEARTBEAT,
	STORE_SUBDEV
};

enum PollCmdCode {
	POLL_ADD_OBJ,
	POLL_REM_OBJ,
	POLL_START,
	POLL_STOP,
	POLL_UPD_PERIOD,
	POLL_REM_DEV,
	POLL_EXIT,
	POLL_REM_EXT_TRIG_OBJ,
	POLL_ADD_HEARTBEAT,
	POLL_REM_HEARTBEAT
};

enum SerialModel {
	BY_DEVICE,
	BY_CLASS,
	BY_PROCESS,
	NO_SYNC
};

enum AttReqType {
	READ_REQ,
	WRITE_REQ
};

enum LockCmdCode {
	LOCK_ADD_DEV,
	LOCK_REM_DEV,
	LOCK_UNLOCK_ALL_EXIT,
	LOCK_EXIT
};

enum LogLevel {
  	LOG_OFF,
  	LOG_FATAL,
  	LOG_ERROR,
  	LOG_WARN,
  	LOG_INFO,
  	LOG_DEBUG
} ;

enum LogTarget {
  	LOG_CONSOLE,
  	LOG_FILE,
  	LOG_DEVICE
};

enum EventType {
	CHANGE_EVENT,
	QUALITY_EVENT,
	PERIODIC_EVENT,
	ARCHIVE_EVENT,
	USER_EVENT,
	ATTR_CONF_EVENT,
	DATA_READY_EVENT,
	numEventType
};

enum AttrSerialModel
{
	ATTR_NO_SYNC,
	ATTR_BY_KERNEL,
	ATTR_BY_USER
};

enum KeepAliveCmdCode
{
	EXIT_TH
};

enum AccessControlType
{
	ACCESS_READ,
	ACCESS_WRITE
};

enum MinMaxValueCheck
{
    MIN,
    MAX
};

enum ChannelType
{
    ZMQ,
    NOTIFD
};

// IDL

enum AttrQuality {
     ATTR_VALID,
     ATTR_INVALID,
     ATTR_ALARM,
     ATTR_CHANGING,
     ATTR_WARNING
     /*, __max_AttrQuality=0xffffffff */
};

enum AttrWriteType
{
    READ,
    READ_WITH_WRITE,
    WRITE,
    READ_WRITE
    /*, __max_AttrWriteType=0xffffffff */
};

enum AttrDataFormat
{
    SCALAR,
    SPECTRUM,
    IMAGE,
    FMT_UNKNOWN
    /*, __max_AttrDataFormat=0xffffffff */
};

enum DevSource
{
    DEV,
    CACHE,
    CACHE_DEV
    /*, __max_DevSource=0xffffffff */
};

enum ErrSeverity
{
    WARN,
    ERR,
    PANIC
    /*, __max_ErrSeverity=0xffffffff */
};

enum DevState
{
    ON,
    OFF,
    CLOSE,
    OPEN,
    INSERT,
    EXTRACT,
    MOVING,
    STANDBY,
    FAULT,
    INIT,
    RUNNING,
    ALARM,
    DISABLE,
    UNKNOWN
    /*, __max_DevState=0xffffffff */
};

enum DispLevel
{
    OPERATOR,
    EXPERT
    /*, __max_DispLevel=0xffffffff */
};
