################################################################################
##
## This file is part of Taurus, a Tango User Interface Library
## 
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
## 
## PyTango is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## PyTango is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################

import ctypes
import ctypes.util
import enumeration
import time

_tango_lib_name = ctypes.util.find_library("c_tango")

if _tango_lib_name is None:
    raise RuntimeError("Failed to find c_tango shared library")

_ref      = ctypes.byref
String    = ctypes.c_char_p
StringPtr = ctypes.POINTER(String)
Int       = ctypes.c_int
IntPtr    = ctypes.POINTER(Int) 
Enum      = ctypes.c_int
Length    = ctypes.c_uint
Bool      = ctypes.c_short

c_tango = ctypes.CDLL(_tango_lib_name)

TangoDataType = Enum
TangoDataTypeEnum = enumeration.Enumeration("TangoDataTypeEnum", (
    "DEV_VOID",
    "DEV_BOOLEAN",
    "DEV_SHORT",
    "DEV_LONG",
    "DEV_FLOAT",
    "DEV_DOUBLE",
    "DEV_USHORT",
    "DEV_ULONG",
    "DEV_STRING",
    "DEVVAR_CHARARRAY",
    "DEVVAR_SHORTARRAY",
    "DEVVAR_LONGARRAY",
    "DEVVAR_FLOATARRAY",
    "DEVVAR_DOUBLEARRAY",
    "DEVVAR_USHORTARRAY",
    "DEVVAR_ULONGARRAY",
    "DEVVAR_STRINGARRAY",
    "DEVVAR_LONGSTRINGARRAY",
    "DEVVAR_DOUBLESTRINGARRAY",
    "DEV_STATE",
    "CONST_DEV_STRING",
    "DEVVAR_BOOLEANARRAY",
    "DEV_UCHAR",
    "DEV_LONG64",
    "DEV_ULONG64",
    "DEVVAR_LONG64ARRAY",
    "DEVVAR_ULONG64ARRAY",
    "DEV_INT" ) )
locals().update(TangoDataTypeEnum.lookup)
TangoDataTypePtr = ctypes.POINTER(TangoDataType)

def _is_scalar(data_type):
    if data_type <= TangoDataTypeEnum.DEV_STRING: return True
    if data_type > TangoDataTypeEnum.DEV_STRING and data_type < TangoDataTypeEnum.DEV_STATE: return False
    if data_type == TangoDataTypeEnum.DEVVAR_BOOLEANARRAY or \
       data_type == TangoDataTypeEnum.DEVVAR_LONG64ARRAY or \
       data_type == TangoDataTypeEnum.DEVVAR_ULONG64ARRAY:
       return False
    return True

TangoDataTypeEnum.is_scalar = _is_scalar

TangoDevState = Enum
TangoDevStateEnum = enumeration.Enumeration("TangoDevStateEnum", (
    "ON",
    "OFF",
    "CLOSE",
    "OPEN",
    "INSERT",
    "EXTRACT",
    "MOVING",
    "STANDBY",
    "FAULT",
    "INIT",
    "RUNNING",
    "ALARM",
    "DISABLE",
    "UNKNOWN") )
locals().update(TangoDevStateEnum.lookup)
TangoDevStatePtr = ctypes.POINTER(TangoDevState)

AttrQuality = Enum
AttrQualityEnum = enumeration.Enumeration("AttrQualityEnum", (
    "ATTR_VALID",
    "ATTR_INVALID",
    "ATTR_ALARM",
    "ATTR_CHANGING", 
    "ATTR_WARNING" ) )
locals().update(AttrQualityEnum.lookup)
AttrQualityPtr = ctypes.POINTER(AttrQuality)

AttrWriteType = Enum
AttrWriteTypeEnum = enumeration.Enumeration("AttrWriteTypeEnum", (
   "READ",
   "READ_WITH_WRITE",
   "WRITE",
   "READ_WRITE" ) )
locals().update(AttrWriteTypeEnum.lookup)
AttrWriteTypePtr = ctypes.POINTER(AttrWriteType)

AttrDataFormat = Enum
AttrDataFormatEnum = enumeration.Enumeration("AttrDataFormatEnum", (
    "SCALAR",
    "SPECTRUM",
    "IMAGE" ) )
locals().update(AttrDataFormatEnum.lookup)
AttrDataFormatPtr = ctypes.POINTER(AttrDataFormat)

DispLevel = Enum
DispLevelEnum = enumeration.Enumeration("DispLevelEnum", (
    "OPERATOR",
    "EXPERT" ) )
locals().update(DispLevelEnum.lookup)
DispLevelPtr = ctypes.POINTER(DispLevel)

ErrSeverity = Enum
ErrSeverityEnum = enumeration.Enumeration("ErrSeverityEnum", (
   "WARN",
   "ERR",
   "PANIC" ) )
locals().update(ErrSeverityEnum.lookup)   
ErrSeverityPtr = ctypes.POINTER(ErrSeverity)

DevSource = Enum
DevSourceEnum = enumeration.Enumeration("DevSourceEnum", (
    "DEV",
    "CACHE",
    "CACHE_DEV" ) )
locals().update(DevSourceEnum.lookup)  
DevSourcePtr = ctypes.POINTER(DevSource)
    
TangoDevLong = ctypes.c_int32
TangoDevLongPtr = ctypes.POINTER(TangoDevLong)
TangoDevULong = ctypes.c_uint32
TangoDevULongPtr = ctypes.POINTER(TangoDevULong)
TangoDevLong64 = ctypes.c_int64
TangoDevLong64Ptr = ctypes.POINTER(TangoDevLong64)
TangoDevULong64 = ctypes.c_uint64
TangoDevULong64Ptr = ctypes.POINTER(TangoDevULong64)


class VarArray(ctypes.Structure):
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        if not isinstance(i,int): raise TypeError("tuple indices must be integers")
        if i < 0 or i > self.length-1: raise IndexError("tuple index out of range")
        return self.sequence[i]


class VarBoolArray(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", ctypes.POINTER(ctypes.c_int16)) 
        

class VarCharArray(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", ctypes.POINTER(ctypes.c_char)) 


class VarShortArray(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", ctypes.POINTER(ctypes.c_int16)) 


class VarUShortArray(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", ctypes.POINTER(ctypes.c_uint16)) 


class VarLongArray(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", TangoDevLongPtr) 


class VarULongArray(VarArray):
    _fields_ =  \
        ("length", Length), \
        ("sequence", TangoDevULongPtr) 


class VarLong64Array(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", TangoDevLong64Ptr) 


class VarULong64Array(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", TangoDevULong64Ptr) 


class VarFloatArray(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", ctypes.POINTER(ctypes.c_float)) 


class VarDoubleArray(VarArray):
    _fields_ =  \
        ("length", Length), \
        ("sequence", ctypes.POINTER(ctypes.c_double)) 
    

class VarStringArray(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", StringPtr)
    
    def __str__(self):
        l = self.length
        if l == 1:
            return self.sequence[0]
        return str(list(self.sequence[:10]))
VarStringArrayPtr = ctypes.POINTER(VarStringArray)

class VarStateArray(VarArray):
    _fields_ = \
        ("length", Length), \
        ("sequence", TangoDevStatePtr) 
    
    def __str__(self):
        l = self.length
        if l == 1:
            return TangoDevStateEnum.whatis(self.sequence[0])
        return map(TangoDevStateEnum.whatis, self.sequence[:10])
        

class TangoAttributeData(ctypes.Union):
    _fields_ =  \
        ("bool_arr", VarBoolArray), \
        ("char_arr", VarCharArray), \
        ("short_arr", VarShortArray), \
        ("ushort_arr", VarUShortArray), \
        ("long_arr", VarLongArray), \
        ("ulong_arr", VarULongArray), \
        ("long64_arr", VarLong64Array), \
        ("ulong64_arr", VarULong64Array), \
        ("float_arr", VarFloatArray), \
        ("double_arr", VarDoubleArray), \
        ("string_arr", VarStringArray), \
        ("state_arr", VarStateArray)
    
    def get_raw(self, type):
        if type == DEV_BOOLEAN:   return self.bool_arr
        elif type == DEV_UCHAR:    return self.char_arr
        elif type == DEV_SHORT:   return self.short_arr
        elif type == DEV_USHORT:  return self.ushort_arr
        elif type == DEV_LONG:    return self.long_arr
        elif type == DEV_ULONG:   return self.ulong_arr
        elif type == DEV_LONG64:  return self.long64_arr
        elif type == DEV_ULONG64: return self.ulong64_arr
        elif type == DEV_FLOAT:   return self.float_arr
        elif type == DEV_DOUBLE:  return self.double_arr
        elif type == DEV_STRING:  return self.string_arr
        elif type == DEV_STATE:   return self.state_arr

    def get(self, type):
        raw = self.get_raw(type)
        if TangoDataTypeEnum.is_scalar(type):
            return raw[0]
        return raw

    def representation(self, type):
        return str(self.get_raw(type))
TangoAttributeDataPtr = ctypes.POINTER(TangoAttributeData)


class TangoCommandData(ctypes.Union):
    _fields_ = \
        ("bool_val", Bool), \
        ("short_val", ctypes.c_short), \
        ("ushort_val", ctypes.c_ushort), \
        ("long_val", ctypes.c_int32), \
        ("ulong_val", ctypes.c_uint32), \
        ("float_val", ctypes.c_float), \
        ("double_val", ctypes.c_double), \
        ("string_val", ctypes.c_char_p), \
        ("state_val", TangoDevState), \
        ("long64_val", ctypes.c_int64), \
        ("ulong64_val", ctypes.c_uint64), \
        ("bool_arr", VarBoolArray), \
        ("char_arr", VarCharArray), \
        ("short_arr", VarShortArray), \
        ("ushort_arr", VarUShortArray), \
        ("long_arr", VarLongArray), \
        ("ulong_arr", VarULongArray), \
        ("long64_arr", VarLong64Array), \
        ("ulong64_arr", VarULong64Array), \
        ("float_arr", VarFloatArray), \
        ("double_arr", VarDoubleArray), \
        ("string_arr", VarStringArray), \
        ("state_arr", VarStateArray),
TangoCommandDataPtr = ctypes.POINTER(TangoCommandData)


class TangoPropertyData(ctypes.Union):
    _fields_ = \
        ("bool_val", Bool), \
        ("char_val", ctypes.c_char), \
        ("short_val", ctypes.c_short), \
        ("ushort_val", ctypes.c_ushort), \
        ("long_val", ctypes.c_int32), \
        ("ulong_val", ctypes.c_uint32), \
        ("float_val", ctypes.c_float), \
        ("double_val", ctypes.c_double), \
        ("string_val", ctypes.c_char_p), \
        ("long64_val", ctypes.c_int64), \
        ("ulong64_val", ctypes.c_uint64), \
        ("short_arr", VarShortArray), \
        ("ushort_arr", VarUShortArray), \
        ("long_arr", VarLongArray), \
        ("ulong_arr", VarULongArray), \
        ("long64_arr", VarLong64Array), \
        ("ulong64_arr", VarULong64Array), \
        ("float_arr", VarFloatArray), \
        ("double_arr", VarDoubleArray), \
        ("string_arr", VarStringArray),
TangoPropertyDataPtr = ctypes.POINTER(TangoPropertyData)


class CommandData(ctypes.Structure):
    _fields_ = \
        ("arg_type",TangoDataType), \
        ("cmd_data",TangoCommandData)
CommandDataPtr = ctypes.POINTER(CommandData)


_time_t = ctypes.c_long
_suseconds_t = ctypes.c_long

class timeval(ctypes.Structure):
    _fields_ = \
        ("tv_sec", _time_t), \
        ("tv_usec", _suseconds_t),
    
    def __str__(self):
        return time.ctime(self.tv_sec + 1E-6 * self.tv_usec)
timevalPtr = ctypes.POINTER(timeval)


class AttributeData(ctypes.Structure):
    _fields_ = \
        ("data_type", TangoDataType), \
        ("attr_data", TangoAttributeData), \
        ("quality", AttrQuality), \
        ("name", String), \
        ("dim_x", Int), \
        ("dim_y", Int), \
        ("time_stamp", timeval)
    
    def __str__(self):
        s = "AttributeData[\n"
        s += "name: %s\n" % self.name
        s += "data_type: %s\n" % TangoDataTypeEnum.whatis(self.data_type)
        s += "quality: %s\n" % AttrQualityEnum.whatis(self.quality)
        s += "dim_x: %d\n" % self.dim_x
        s += "dim_y: %d\n" % self.dim_y
        s += "time_stamp: %s\n" % self.time_stamp
        s += "attr_data: %s\n" % str(self.attr_data.representation(self.data_type))
        s += "]\n"
        return s
    
    def get_raw_data(self):
        return self.attr_data.get_raw(self.data_type)

    def get_data(self):
        return self.attr_data.get(self.data_type)
AttributeDataPtr = ctypes.POINTER(AttributeData)


class AttributeDataList(ctypes.Structure):
    _fields_ = \
        ("length", Length), \
        ("sequence", AttributeDataPtr)

    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        if not isinstance(i,int): raise TypeError("tuple indices must be integers")
        if i < 0 or i > self.length-1: raise IndexError("tuple index out of range")
        return self.sequence[i]
    
    def __str__(self):
        s = "AttributeDataList[\n"
        for attr in self: s += attr
        return s
AttributeDataListPtr = ctypes.POINTER(AttributeDataList)

                        
class DevFailed(ctypes.Structure):
    _fields_ = \
        ("desc", String), \
        ("reason", String), \
        ("origin", String), \
        ("severity", ErrSeverity)
    
    def __str__(self):
        s  = "Severity    : %d\n" % self.severity
        s += "Reason      : %s\n" % self.reason
        s += "Description : %s\n" % self.desc
        s += "Origin      : %s\n\n" % self.origin
        return s    

    def __repr__(self):
        return self.__str__()
DevFailedPtr = ctypes.POINTER(DevFailed)


class ErrorStack(ctypes.Structure):
    _fields_ = \
        ("length", Length), \
        ("sequence", DevFailedPtr) 

    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        if not isinstance(i,int): raise TypeError("tuple indices must be integers")
        if i < 0 or i > self.length-1: raise IndexError("tuple index out of range")
        return self.sequence[i]
    
    def __str__(self):
        s = "\nTango exception:\n"
        for i in xrange(self.length):
            s += str(self.sequence[i])
        return s
        
    def __repr__(self):
        return self.__str__()
ErrorStackPtr = ctypes.POINTER(ErrorStack)


class CommandInfo(ctypes.Structure):
    _fields_ = \
        ("cmd_name", String), \
        ("cmd_tag", Int), \
        ("in_type", Int), \
        ("out_type", Int), \
        ("in_type_desc", String), \
        ("out_type_desc", String), \
        ("disp_level", DispLevel) 
CommandInfoPtr = ctypes.POINTER(CommandInfo)


class CommandInfoList(ctypes.Structure):
    _fields_ = \
        ("length", Length), \
        ("sequence", CommandInfoPtr)
CommandInfoListPtr = ctypes.POINTER(CommandInfoList)


class AttributeInfo(ctypes.Structure):
    _fields_ = \
        ("name", String), \
        ("writable", AttrWriteType), \
        ("data_format", AttrDataFormat), \
        ("data_type", TangoDataType), \
        ("max_dim_x", Int), \
        ("max_dim_y", Int), \
        ("description", String), \
        ("label", String), \
        ("unit", String), \
        ("standard_unit", String), \
        ("display_unit", String), \
        ("format", String), \
        ("min_value", String), \
        ("max_value", String), \
        ("min_alarm", String), \
        ("max_alarm", String), \
        ("writable_attr_name", String), \
        ("disp_level", DispLevel)
AttributeInfoPtr = ctypes.POINTER(AttributeInfo)


class AttributeInfoList(ctypes.Structure):
    _fields_ = \
        ("length", Length), \
        ("sequence", AttributeInfoPtr)
AttributeInfoListPtr = ctypes.POINTER(AttributeInfoList)


class DbDatum(ctypes.Structure):
    _fields_ = \
        ("property_name", String), \
        ("data_type", TangoDataType), \
        ("prop_data", TangoPropertyData), \
        ("is_empty", Bool), \
        ("wrong_data_type", Bool)
DbDatumPtr = ctypes.POINTER(DbDatum)


class DbData(ctypes.Structure):
    _fields_ = \
        ("length", Length), \
        ("sequence", DbDatumPtr)
DbDataPtr = ctypes.POINTER(DbData)


DeviceProxyPtr    = ctypes.c_void_p
DeviceProxyPtrPtr = ctypes.POINTER(DeviceProxyPtr)
DatabasePtr       = ctypes.c_void_p
DatabasePtrPtr    = ctypes.POINTER(DatabasePtr)


c_tango.tango_create_device_proxy.argtypes = (String, DeviceProxyPtrPtr, ErrorStackPtr, )
c_tango.tango_delete_device_proxy.argtypes = (DeviceProxyPtrPtr, ErrorStackPtr, )
c_tango.tango_set_timeout_millis.argtypes = (DeviceProxyPtr, Int, ErrorStackPtr, )
c_tango.tango_get_timeout_millis.argtypes = (DeviceProxyPtr, IntPtr, ErrorStackPtr, )
c_tango.tango_set_source.argtypes = (DeviceProxyPtr, DevSource, ErrorStackPtr, )
c_tango.tango_get_source.argtypes = (DeviceProxyPtr, DevSourcePtr, ErrorStackPtr, )
c_tango.tango_command_query.argtypes = (DeviceProxyPtr, String, CommandInfoPtr, ErrorStackPtr, )
c_tango.tango_command_list_query.argtypes = (DeviceProxyPtr, CommandInfoListPtr, ErrorStackPtr, )
c_tango.tango_command_inout.argtypes = (DeviceProxyPtr, String, CommandDataPtr, CommandDataPtr, ErrorStackPtr, )
c_tango.tango_free_CommandData.argtypes = (CommandDataPtr, )
c_tango.tango_free_CommandInfo.argtypes = (CommandInfoPtr, )
c_tango.tango_free_CommandInfoList.argtypes = (CommandInfoListPtr, )
c_tango.tango_get_attribute_list.argtypes = (DeviceProxyPtr, VarStringArrayPtr, ErrorStackPtr, )
c_tango.tango_get_attribute_config.argtypes = (DeviceProxyPtr, VarStringArrayPtr, AttributeInfoListPtr, ErrorStackPtr, )
c_tango.tango_attribute_list_query.argtypes = (DeviceProxyPtr, AttributeInfoListPtr, ErrorStackPtr, )
c_tango.tango_read_attribute.argtypes = (DeviceProxyPtr, String, AttributeDataPtr, ErrorStackPtr, )
c_tango.tango_write_attribute.argtypes = (DeviceProxyPtr, String, AttributeDataPtr, ErrorStackPtr, )
c_tango.tango_read_attributes.argtypes = (DeviceProxyPtr, VarStringArrayPtr, AttributeDataListPtr, ErrorStackPtr, )
c_tango.tango_write_attributes.argtypes = (DeviceProxyPtr, AttributeDataListPtr, ErrorStackPtr, )
c_tango.tango_free_AttributeData.argtypes = (AttributeDataPtr, )
c_tango.tango_free_AttributeDataList.argtypes = (AttributeDataListPtr, )
c_tango.tango_free_VarStringArray.argtypes = (VarStringArrayPtr, )
c_tango.tango_print_ErrorStack.argtypes = (ErrorStackPtr, )
c_tango.tango_free_ErrorStack.argtypes = (ErrorStackPtr, )
c_tango.tango_create_database_proxy.argtypes = (DatabasePtrPtr, ErrorStackPtr, )
c_tango.tango_delete_database_proxy.argtypes = (DatabasePtrPtr, ErrorStackPtr, )
c_tango.tango_get_device_exported.argtypes = (DatabasePtr, String, DbDatumPtr, ErrorStackPtr, )
c_tango.tango_get_device_exported_for_class.argtypes = (DatabasePtr, String, DbDatumPtr, ErrorStackPtr, )
c_tango.tango_get_object_list.argtypes = (DatabasePtr, String, DbDatumPtr, ErrorStackPtr, )
c_tango.tango_get_object_property_list.argtypes = (DatabasePtr, String, String, DbDatumPtr, ErrorStackPtr, )
c_tango.tango_get_property.argtypes = (DatabasePtr, String, DbDataPtr, ErrorStackPtr, )
c_tango.tango_put_property.argtypes = (DatabasePtr, String, DbDataPtr, ErrorStackPtr, )
c_tango.tango_delete_property.argtypes = (DatabasePtr, String, DbDataPtr, ErrorStackPtr, )
c_tango.tango_get_device_property.argtypes = (DeviceProxyPtr, DbDataPtr, ErrorStackPtr, )
c_tango.tango_put_device_property.argtypes = (DeviceProxyPtr, DbDataPtr, ErrorStackPtr, )
c_tango.tango_delete_device_property.argtypes = (DeviceProxyPtr, DbDataPtr, ErrorStackPtr, )
c_tango.tango_free_DbDatum.argtypes = (DbDatumPtr, )
c_tango.tango_free_DbData.argtypes = (DbDataPtr, )


def tango_create_device_proxy(dev_name):
    dev_name = ctypes.create_string_buffer(dev_name)
    dev_ptr = ctypes.c_void_p()
    err_stack = ErrorStack()
    result = c_tango.tango_create_device_proxy(dev_name, _ref(dev_ptr), _ref(err_stack))
    if result:
        return dev_ptr
    raise Exception(err_stack)

def tango_delete_device_proxy(dev_ptr):
    err_stack = ErrorStack()
    result = c_tango.tango_delete_device_proxy(_ref(dev_ptr), _ref(err_stack))
    if result:
        return True
    raise Exception(err_stack)

def tango_set_timeout_millis(dev_ptr, millis):
    err_stack = ErrorStack()
    millis = ctypes.c_int(millis)
    result = c_tango.tango_set_timeout_millis(dev_ptr, millis, _ref(err_stack))
    if result:
        return True
    raise Exception(err_stack)    

def tango_get_timeout_millis(dev_ptr):
    err_stack = ErrorStack()
    millis = ctypes.c_int()
    result = c_tango.tango_get_timeout_millis(dev_ptr, _ref(millis), _ref(err_stack))
    if result:
        return millis
    raise Exception(err_stack)      

def tango_set_source(dev_ptr, src):
    """src -> DevSource"""
    err_stack = ErrorStack()
    result = c_tango.tango_set_source(dev_ptr, src, _ref(err_stack))
    if result:
        return True
    raise Exception(err_stack)   
    
def tango_get_source(dev_ptr):
    err_stack = ErrorStack()
    src = ctypes.c_int()
    result = c_tango.tango_get_source(dev_ptr, _ref(src), _ref(err_stack))
    if result:
        return src
    raise Exception(err_stack)
    
def tango_command_query(dev_ptr, cmd_name):
    err_stack = ErrorStack()
    cmd_name = ctypes.create_string_buffer(cmd_name)
    cmd_info = CommandInfo()
    result = c_tango.tango_command_query(dev_ptr, cmd_name, _ref(cmd_info), _ref(err_stack))
    if result:
        return cmd_info
    raise Exception(err_stack)    

def tango_command_list_query(dev_ptr):
    err_stack = ErrorStack()
    cmd_info_list = CommandInfoList()
    result = c_tango.tango_command_list_query(dev_ptr, _ref(cmd_info_list), _ref(err_stack))
    if result:
        return cmd_info_list
    raise Exception(err_stack)    

def tango_command_inout(dev_ptr, cmd_name, arg_in):
    """arg_in->CommandData"""
    err_stack = ErrorStack()
    cmd_name = ctypes.create_string_buffer(cmd_name)
    arg_out = CommandData()
    result = c_tango.tango_command_inout(dev_ptr, cmd_name, _ref(arg_in), _ref(arg_out), _ref(err_stack))
    if result:
        return arg_out
    raise Exception(err_stack)  
    
def tango_free_CommandData(cmd_data):
    c_tango.tango_free_CommandData(_ref(cmd_data))

def tango_free_CommandInfo(cmd_info):
    c_tango.tango_free_CommandInfo(_ref(cmd_info))
    
def tango_free_CommandInfoList(cmd_info_list):
    c_tango.tango_free_CommandInfoList(_ref(cmd_info_list))

def tango_get_attribute_list(dev_ptr):
    err_stack = ErrorStack()
    attr_names = VarStringArray()
    result = c_tango.tango_get_attribute_list(dev_ptr, _ref(attr_names), _ref(err_stack))
    if result:
        return attr_names
    raise Exception(err_stack)  
    
def tango_get_attribute_config(dev_ptr, attr_names):
    print "TODO"
    return
    err_stack = ErrorStack()
    attr_names = VarStringArray()
    attr_info_list = AttributeInfoList()
    result = c_tango.tango_get_attribute_config(dev_ptr, _ref(attr_names), _ref(attr_info_list), _ref(err_stack))
    if result:
        return attr_info_list
    raise Exception(err_stack)  
    
def tango_attribute_list_query(dev_ptr):
    err_stack = ErrorStack()
    attr_info_list = AttributeInfoList()
    result = c_tango.tango_attribute_list_query(dev_ptr, _ref(attr_info_list), _ref(err_stack))
    if result:
        return attr_info_list
    raise Exception(err_stack) 
    
def tango_read_attribute(dev_ptr, attr_name):
    attr_name = ctypes.create_string_buffer(attr_name)
    attr_data = AttributeData()
    err_stack = ErrorStack()
    result = c_tango.tango_read_attribute(dev_ptr, attr_name, _ref(attr_data), _ref(err_stack))    
    if result:
        return attr_data
    raise Exception(err_stack)

def tango_write_attribute(dev_ptr, attr_name, value):
    print "TODO"
    return
    attr_data = AttributeData()
    attr_data.name = ctypes.create_string_buffer(attr_name)
    attr_data.attr_data = value
    err_stack = ErrorStack()
    result = c_tango.tango_write_attribute(dev_ptr, attr_name, _ref(attr_data), _ref(err_stack))    
    if result:
        return True
    raise Exception(err_stack)

def tango_read_attributes(dev_ptr, attr_names):
    print "TODO"
    return
    attr_data_list = AttributeDataList()
    attr_names = VarStringArray()
    err_stack = ErrorStack()
    result = c_tango.tango_read_attribute(dev_ptr, _ref(attr_names), _ref(attr_data_list), _ref(err_stack))    
    if result:
        return attr_data
    raise Exception(err_stack)
    
def tango_write_attributes(dev_ptr, attr_data_list):
    """attr_data_list->AttributeDataList"""
    err_stack = ErrorStack()
    result = c_tango.tango_write_attributes(dev_ptr, _ref(attr_data_list), _ref(err_stack))    
    if result:
        return True
    raise Exception(err_stack)
    
def tango_free_AttributeData(attr_data):
    c_tango.tango_free_AttributeData(_ref(attr_data))
    
def tango_free_AttributeDataList(attr_data_list):
    c_tango.tango_free_AttributeDataList(_ref(attr_data_list))
    
def tango_free_VarStringArray(str_array):
    c_tango.tango_free_VarStringArray(_ref(str_array))

def tango_print_ErrorStack(err_stack):
    """Should not be used. This function prints to STDOUT instead of sys.stdout.
       Use: 'print err_stack' instead"""
    c_tango.tango_print_ErrorStack(_ref(err_stack))

def tango_free_ErrorStack(err_stack):
    c_tango.tango_free_ErrorStack(_ref(err_stack))

def tango_create_database_proxy():
    err_stack = ErrorStack()
    db_ptr = ctypes.c_void_p()
    result = c_tango.tango_create_database_proxy(_ref(db_ptr), _ref(err_stack))    
    if result:
        return db_ptr
    raise Exception(err_stack)

def tango_delete_database_proxy(db_ptr):
    err_stack = ErrorStack()
    result = c_tango.tango_delete_database_proxy(_ref(db_ptr), _ref(err_stack))    
    if result:
        return True
    raise Exception(err_stack)

def tango_get_device_exported(db_ptr, name_filter):
    err_stack = ErrorStack()
    name_filter = ctypes.create_string_buffer(name_filter)
    db_datum = DbDatum()
    result = c_tango.tango_get_device_exported(db_ptr, name_filter, _ref(db_datum), _ref(err_stack))    
    if result:
        return db_datum
    raise Exception(err_stack)

def tango_get_device_exported_for_class(db_ptr, class_name):
    err_stack = ErrorStack()
    class_name = ctypes.create_string_buffer(class_name)
    db_datum = DbDatum()
    result = c_tango.tango_get_device_exported_for_class(db_ptr, class_name, _ref(db_datum), _ref(err_stack))    
    if result:
        return db_datum
    raise Exception(err_stack)

def tango_get_object_list(db_ptr, name_filter):
    err_stack = ErrorStack()
    name_filter = ctypes.create_string_buffer(name_filter)
    db_datum = DbDatum()
    result = c_tango.tango_get_object_list(db_ptr, name_filter, _ref(db_datum), _ref(err_stack))    
    if result:
        return db_datum
    raise Exception(err_stack)

def tango_get_object_property_list(db_ptr, obj_name, name_filter):
    err_stack = ErrorStack()
    obj_name = ctypes.create_string_buffer(obj_name)
    name_filter = ctypes.create_string_buffer(name_filter)
    db_datum = DbDatum()
    result = c_tango.tango_get_object_property_list(db_ptr, obj_name, name_filter, _ref(db_datum), _ref(err_stack))    
    if result:
        return db_datum
    raise Exception(err_stack)

def tango_get_property(db_ptr, obj_name):
    err_stack = ErrorStack()
    obj_name = ctypes.create_string_buffer(obj_name)
    db_data = DbData()
    result = c_tango.tango_get_property(db_ptr, obj_name, _ref(db_data), _ref(err_stack))    
    if result:
        return db_datum
    raise Exception(err_stack)

def tango_put_property(db_ptr, obj_name, prop_list):
    """prop_list -> DbData"""
    err_stack = ErrorStack()
    obj_name = ctypes.create_string_buffer(obj_name)
    result = c_tango.tango_put_property(db_ptr, obj_name, _ref(prop_list), _ref(err_stack))    
    if result:
        return True
    raise Exception(err_stack)

def tango_delete_property(db_ptr, obj_name, prop_list):
    """prop_list -> DbData"""
    err_stack = ErrorStack()
    obj_name = ctypes.create_string_buffer(obj_name)
    result = c_tango.tango_delete_property(db_ptr, obj_name, _ref(prop_list), _ref(err_stack))    
    if result:
        return True
    raise Exception(err_stack)
    
def tango_get_device_property(dev_ptr, prop_list):
    """prop_list -> DbData"""
    err_stack = ErrorStack()
    result = c_tango.tango_get_device_property(dev_ptr, _ref(prop_list), _ref(err_stack))    
    if result:
        return prop_list
    raise Exception(err_stack)

def tango_put_device_property(dev_ptr, prop_list):
    """prop_list -> DbData"""
    err_stack = ErrorStack()
    result = c_tango.tango_put_device_property(dev_ptr, _ref(prop_list), _ref(err_stack))    
    if result:
        return True
    raise Exception(err_stack)

def tango_delete_device_property(dev_ptr, prop_list):
    """prop_list -> DbData"""
    err_stack = ErrorStack()
    result = c_tango.tango_delete_device_property(dev_ptr, _ref(prop_list), _ref(err_stack))    
    if result:
        return True
    raise Exception(err_stack)

def tango_free_DbDatum(db_datum):
    c_tango.tango_free_DbDatum(_ref(db_datum))
    
def tango_free_DbData(db_data):
    c_tango.tango_free_DbData(_ref(db_data))


class DeviceProxy:
    def __init__(self, dev_name):
        self._dev_name = dev_name
        self._dev = tango_create_device_proxy(dev_name)
        
    def read_attribute(self, attr_name):
        return tango_read_attribute(self._dev, attr_name)

    def write_attribute(self, attr_name, value):
        return tango_read_attribute(self._dev, attr)
        
    def read_attributes(self, attr_name_list):
        return tango_read_attributes(self._dev, attr_name_list)
        
    def get_property(self, attr_name_list):
        if isinstance(attr_name_list, str):
            attr_name_list = [ attr_name_list ]
        n = len(attr_name_list)
        db_data = DbData()
        db_data.length = n
        db_data.sequence = (n*DbDatum)()
        for i in xrange(n):
            db_data.sequence[i].property_name = attr_name_list[i]
            db_data.sequence[i].data_type = DEV_STRING
        return tango_get_device_property(self._dev, db_data)

    def __del__(self):
        try:
            if self._dev:
                try:
                    tango_delete_device_proxy(self._dev)
                except Exception, e:
                    print e
        except AttributeError:
            #The error was in the constructor and therefore _dev is not defined
            pass

