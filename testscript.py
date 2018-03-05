import tango
import socket
import pytest
from time import sleep
from tango import DeviceProxy
from tango import DevState
from tango import Database
from tango import DbDatum
from tango import Release
dp = DeviceProxy('sys/tg_test/1')

assert dp.dev_name() == u'sys/tg_test/1'
assert isinstance(dp.get_device_db(), Database)
assert dp.status() == 'The device is in RUNNING state.'
assert dp.state() == DevState.RUNNING
assert dp.adm_name() == u'dserver/TangoTest/test'
assert dp.description() == u'A TANGO device'
assert dp.name() == u'sys/tg_test/1'
with pytest.raises(Exception):
    dp.alias()  # Not aliased therefore raise exception

info = dp.info()
assert info.dev_class == "TangoTest"
assert info.server_id == "TangoTest/test"
assert info.server_host == socket.gethostname()
assert info.server_version == 5

assert dp.get_tango_lib_version() == 902
duration = dp.ping(wait=True)
if Release.version_number > 923:
    assert isinstance(duration, long)
else:
    assert isinstance(duration, int)

assert len(dp.black_box(5)) == 5

assert dp.get_command_list() == [u'CrashFromDevelopperThread', u'CrashFromOmniThread',
                                 u'DevBoolean', u'DevDouble', u'DevFloat', u'DevLong',
                                 u'DevLong64', u'DevShort', u'DevString', u'DevULong',
                                 u'DevULong64', u'DevUShort', u'DevVarCharArray',
                                 u'DevVarDoubleArray', u'DevVarDoubleStringArray',
                                 u'DevVarFloatArray', u'DevVarLong64Array',
                                 u'DevVarLongArray', u'DevVarLongStringArray',
                                 u'DevVarShortArray', u'DevVarStringArray',
                                 u'DevVarULong64Array', u'DevVarULongArray',
                                 u'DevVarUShortArray', u'DevVoid', u'DumpExecutionState',
                                 u'Init', u'State', u'Status', u'SwitchStates']

info = dp.get_command_config('DevDouble')
assert info.cmd_name == u'DevDouble'
assert info.in_type == tango.CmdArgType.DevDouble
assert info.out_type == tango.CmdArgType.DevDouble
assert info.in_type_desc == u'Any DevDouble value'
assert info.out_type_desc == u'Echo of the argin value'
assert info.disp_level == tango.DispLevel.OPERATOR

info_list = dp.get_command_config(['DevDouble', 'DevLong64', 'DevVarShortArray'])
assert len(info_list) == 3
assert info_list[0].cmd_name == u'DevDouble'
assert info_list[0].in_type == tango.CmdArgType.DevDouble
assert info_list[0].out_type == tango.CmdArgType.DevDouble
assert info_list[0].in_type_desc == u'Any DevDouble value'
assert info_list[0].out_type_desc == u'Echo of the argin value'
assert info_list[0].disp_level == tango.DispLevel.OPERATOR
assert info_list[1].cmd_name == u'DevLong64'
assert info_list[1].in_type == tango.CmdArgType.DevLong64
assert info_list[1].out_type == tango.CmdArgType.DevLong64
assert info_list[1].in_type_desc == u'Any DevLong64 value'
assert info_list[1].out_type_desc == u'Echo of the argin value'
assert info_list[1].disp_level == tango.DispLevel.OPERATOR
assert info_list[2].cmd_name == u'DevVarShortArray'
assert info_list[2].in_type == tango.CmdArgType.DevVarShortArray
assert info_list[2].out_type == tango.CmdArgType.DevVarShortArray
assert info_list[2].in_type_desc == u'-'
assert info_list[2].out_type_desc == u'-'
assert info_list[2].disp_level == tango.DispLevel.OPERATOR

info = dp.command_query('DevDouble')
assert info.cmd_name == u'DevDouble'
assert info.in_type == tango.CmdArgType.DevDouble
assert info.out_type == tango.CmdArgType.DevDouble
assert info.in_type_desc == u'Any DevDouble value'
assert info.out_type_desc == u'Echo of the argin value'
assert info.disp_level == tango.DispLevel.OPERATOR

cmd_info_list = dp.command_list_query()
assert len(cmd_info_list) == 30
info = cmd_info_list[3]
assert info.cmd_name == u'DevDouble'
assert info.in_type == tango.CmdArgType.DevDouble
assert info.out_type == tango.CmdArgType.DevDouble
assert info.in_type_desc == u'Any DevDouble value'
assert info.out_type_desc == u'Echo of the argin value'
assert info.disp_level == tango.DispLevel.OPERATOR

info = dp.import_info()
assert info.name == 'sys/tg_test/1'
assert info.exported == 1
assert info.ior[:4] == 'IOR:'
assert info.version == '5'

db_datum = DbDatum('test')
db_datum.value_string = ['3.142']
dp.put_property(db_datum)
props = dp.get_property('test')
assert props.keys() == ["test"]
assert props.values() == [['3.142']]
plist = dp.get_property_list("*")
assert len(plist) == 1 and plist[0] == 'test'
db_datum = DbDatum('timeout')
db_datum.value_string = ['0.05']
dp.put_property(db_datum)
props = dp.get_property(['test', 'timeout'])
assert props.keys() == ['test', 'timeout']
assert props.values() == [['3.142'], ['0.05']]
plist = dp.get_property_list("*")
assert len(plist) == 2 and plist[0] == 'test' and plist[1] == 'timeout'
db_datum = DbDatum('test')
props = dp.get_property(db_datum)
assert props.keys() == ["test"]
assert props.values() == [['3.142']]
db_datum_list = [DbDatum('test'), DbDatum('timeout')]
props = dp.get_property(db_datum_list)
assert props.keys() == ["test", "timeout"]
assert props.values() == [['3.142'], ['0.05']]
dp.delete_property('test')
plist = dp.get_property_list("*")
assert len(plist) == 1 and plist[0] == 'timeout'
db_datum = DbDatum('test')
db_datum.value_string = ['3.142']
dp.put_property(db_datum)
dp.delete_property(['test', 'timeout'])
plist = dp.get_property_list("*")
assert len(plist) == 0
db_datum1 = DbDatum('test')
db_datum1.value_string = ['3.142']
db_datum2 = DbDatum('timeout')
db_datum2.value_string = ['0.05']
dp.put_property([db_datum1, db_datum2])
db_datum = DbDatum('test')
dp.delete_property(db_datum)
plist = dp.get_property_list("*")
assert len(plist) == 1 and plist[0] == 'timeout'
props = {'test': '3.142'}
dp.put_property(props)
db_datum_list = [DbDatum('test'), DbDatum('timeout')]
props = dp.get_property(db_datum_list)
dp.delete_property(db_datum_list)
plist = dp.get_property_list("*")
assert len(plist) == 0
props = {'test': ['3.142'], 'timeout': ['0.05']}
dp.put_property(props)
plist = dp.get_property_list("*")
assert len(plist) == 2 and plist[0] == 'test' and plist[1] == 'timeout'
props = {'test': ['3.142'], 'timeout': ['0.05']}
dp.delete_property(db_datum_list)
plist = dp.get_property_list("*")
assert len(plist) == 0

assert dp.get_pipe_list() == ['string_long_short_ro']
pipe_info = dp.get_pipe_config('string_long_short_ro')
assert pipe_info.description == 'Pipe example'
assert pipe_info.disp_level == tango.DispLevel.OPERATOR
assert pipe_info.extensions == []
assert pipe_info.label == 'string_long_short_ro'
assert pipe_info.name == 'string_long_short_ro'
assert pipe_info.writable == tango.PipeWriteType.PIPE_READ
pipe_info_list = dp.get_pipe_config(['string_long_short_ro', 'string_long_short_ro'])
assert len(pipe_info_list) == 2
pinfo = pipe_info_list[0]
assert pinfo.description == 'Pipe example'
assert pinfo.disp_level == tango.DispLevel.OPERATOR
assert pinfo.extensions == []
assert pinfo.label == 'string_long_short_ro'
assert pinfo.name == 'string_long_short_ro'
assert pinfo.writable == tango.PipeWriteType.PIPE_READ

# dp.set_pipe_config([])
blob = dp.read_pipe('string_long_short_ro')
assert blob == ('', [{'dtype': tango._tango.CmdArgType.DevString, 'name': 'FirstDE', 'value': 'The string'},
                     {'dtype': tango._tango.CmdArgType.DevLong, 'name': 'SecondDE', 'value': 666},
                     {'dtype': tango._tango.CmdArgType.DevShort, 'name': 'ThirdDE', 'value': 12}])
# dp.write_pipe()

assert dp.get_attribute_list() == [u'ampli', u'boolean_scalar', u'double_scalar',
                                   u'double_scalar_rww', u'double_scalar_w',
                                   u'float_scalar', u'long64_scalar', u'long_scalar',
                                   u'long_scalar_rww', u'long_scalar_w', u'no_value',
                                   u'short_scalar', u'short_scalar_ro', u'short_scalar_rww',
                                   u'short_scalar_w', u'string_scalar', u'throw_exception',
                                   u'uchar_scalar', u'ulong64_scalar', u'ushort_scalar',
                                   u'ulong_scalar', u'encoded_scalar', u'boolean_spectrum',
                                   u'boolean_spectrum_ro', u'double_spectrum',
                                   u'double_spectrum_ro', u'float_spectrum',
                                   u'float_spectrum_ro', u'long64_spectrum_ro',
                                   u'long_spectrum', u'long_spectrum_ro',
                                   u'short_spectrum', u'short_spectrum_ro',
                                   u'string_spectrum', u'string_spectrum_ro',
                                   u'uchar_spectrum', u'uchar_spectrum_ro',
                                   u'ulong64_spectrum_ro', u'ulong_spectrum_ro',
                                   u'ushort_spectrum', u'ushort_spectrum_ro', u'wave',
                                   u'boolean_image', u'boolean_image_ro', u'double_image',
                                   u'double_image_ro', u'float_image', u'float_image_ro',
                                   u'long64_image_ro', u'long_image', u'long_image_ro',
                                   u'short_image', u'short_image_ro', u'string_image',
                                   u'string_image_ro', u'uchar_image', u'uchar_image_ro',
                                   u'ulong64_image_ro', u'ulong_image_ro', u'ushort_image',
                                   u'ushort_image_ro', u'State', u'Status']

reply = dp.get_attribute_config('float_scalar')
assert type(reply) == tango._tango.AttributeInfoEx
assert type(reply.alarms) == tango._tango.AttributeAlarmInfo
assert reply.alarms.delta_t == 'Not specified'
assert reply.alarms.delta_val == 'Not specified'
assert reply.alarms.extensions == []
assert reply.alarms.max_alarm == 'Not specified'
assert reply.alarms.max_warning == 'Not specified'
assert reply.alarms.min_alarm == 'Not specified'
assert reply.alarms.min_warning == 'Not specified'
assert reply.data_format == tango._tango.AttrDataFormat.SCALAR
assert reply.data_type == tango._tango.CmdArgType.DevFloat
assert reply.description == 'A float attribute'
assert reply.disp_level == tango._tango.DispLevel.OPERATOR 
assert reply.display_unit == 'No display unit'
assert reply.enum_labels == []
assert type(reply.events) == tango._tango.AttributeEventInfo
assert type(reply.events.arch_event) == tango._tango.ArchiveEventInfo
assert reply.events.arch_event.archive_abs_change == 'Not specified'
assert reply.events.arch_event.archive_period == 'Not specified'
assert reply.events.arch_event.archive_rel_change == 'Not specified'
assert reply.events.arch_event.extensions == []
assert type(reply.events.ch_event) == tango._tango.ChangeEventInfo
assert reply.events.ch_event.abs_change == 'Not specified'
assert reply.events.ch_event.extensions == []
assert reply.events.ch_event.rel_change == 'Not specified'
assert type(reply.events.per_event) == tango._tango.PeriodicEventInfo
assert reply.events.per_event.extensions == []
assert reply.events.per_event.period == '1000'

assert reply.format == '%6.2f'
assert reply.label == 'float_scalar'
assert reply.max_alarm == 'Not specified'
assert reply.max_dim_x == 1
assert reply.max_dim_y == 0
assert reply.max_value == 'Not specified'
assert reply.memorized == tango._tango.AttrMemorizedType.NONE
assert reply.min_alarm == 'Not specified'
assert reply.min_value == 'Not specified'
assert reply.name == 'float_scalar'
assert reply.root_attr_name == u'Not specified'
assert reply.standard_unit == 'No standard unit'
assert reply.sys_extensions == []
assert reply.unit == ''
assert reply.writable == tango._tango.AttrWriteType.WT_UNKNOWN
assert reply.writable_attr_name == 'float_scalar'

reply = dp.get_attribute_config(['float_scalar','long_scalar'])
assert len(reply) == 2
assert type(reply[0]) == tango._tango.AttributeInfo
assert reply[1].data_format == tango._tango.AttrDataFormat.SCALAR
assert reply[1].data_type == tango._tango.CmdArgType.DevLong
assert reply[1].description == 'No description'
assert reply[1].disp_level == tango._tango.DispLevel.OPERATOR
assert reply[1].display_unit == 'No display unit'
assert reply[1].extensions == []
assert reply[1].format == '%d'
assert reply[1].label == 'long_scalar'
assert reply[1].max_alarm == 'Not specified'
assert reply[1].max_dim_x == 1
assert reply[1].max_dim_y == 0
assert reply[1].max_value == 'Not specified'
assert reply[1].min_alarm == 'Not specified'
assert reply[1].min_value == 'Not specified'
assert reply[1].name == 'long_scalar'
assert reply[1].standard_unit == 'No standard unit'
assert reply[1].unit == ''
assert reply[1].writable == tango._tango.AttrWriteType.WT_UNKNOWN
assert reply[1].writable_attr_name == 'long_scalar'

reply1 = dp.get_attribute_config_ex('float_scalar')
assert len(reply1) == 1
assert type(reply1[0]) == tango._tango.AttributeInfoEx
reply2 = dp.get_attribute_config_ex(['float_scalar','long_scalar'])
assert len(reply2) == 2
assert type(reply2[0]) == tango._tango.AttributeInfoEx
attr_info = dp.attribute_query('long_scalar_w')
assert type(attr_info) == tango._tango.AttributeInfoEx

# dp.attribute_list_query([]);
# dp.attribute_list_query_ex([])
# dp.set_attribute_config[]
# dp.set_attribute_config[]
dp.write_attribute('long_scalar_w', 23456)
attr = dp.read_attribute('long_scalar_w')
assert attr.data_format == tango._tango.AttrDataFormat.SCALAR
assert attr.dim_x == 1
assert attr.dim_y == 0
assert attr.has_failed == False
assert attr.is_empty == False
assert attr.name == 'long_scalar_w'
assert attr.nb_read == 1
assert attr.nb_written == 1
assert attr.quality == tango._tango.AttrQuality.ATTR_VALID
assert type(attr.r_dimension) == tango._tango.AttributeDimension
assert attr.r_dimension.dim_x == 1
assert attr.r_dimension.dim_y == 0
assert type(attr.time) == tango._tango.TimeVal
assert attr.type == tango._tango.CmdArgType.DevLong
assert attr.value == 23456
assert attr.w_dim_x == 1
assert attr. w_dim_y == 0
assert type(attr.w_dimension) == tango._tango.AttributeDimension
assert attr.w_dimension.dim_x == 1
assert attr.w_dimension.dim_y == 0
assert attr.w_value == 23456

assert dp.read_attribute('status').value == 'The device is in RUNNING state.'
assert dp.read_attribute('state').value == DevState.RUNNING
assert dp.read_attribute('string_scalar').value == "Default string"
attr = dp.read_attribute('encoded_scalar')
assert attr.value[0] == 'Array of Bytes'
assert attr.value[1] == [0,1,2,3,4,5,6,7,8,9]

dp.write_attribute('double_scalar_w', 3.142)
dp.write_attribute('long_scalar_w', 1357)
dp.write_attribute('boolean_scalar', True)
attrs = dp.read_attributes(['long_scalar_w', 'double_scalar_w', 'boolean_scalar'])
vals = [attr.value for attr in attrs]
assert vals == [1357, 3.142, True]
args = [('long_scalar_w', 8642), ('double_scalar_w', 6.284), ('boolean_scalar', False)]
dp.write_attributes(args)
attrs = dp.read_attributes(['long_scalar_w', 'double_scalar_w', 'boolean_scalar'])
vals = [attr.value for attr in attrs]
assert vals == [8642, 6.284, False]
assert dp.write_read_attribute('long_scalar_w', 98765).value == 98765
args = [('long_scalar_w', 3459), ('double_scalar_w', 15.71), ('boolean_scalar', True)]
attrs = dp.write_read_attributes(args)
vals = [attr.value for attr in attrs]
assert vals == [3459, 15.71, True]

assert dp.is_command_polled('State') is False
dp.poll_command('State', 10)
sleep(0.1)
assert dp.get_command_poll_period('State') == 10
assert dp.is_command_polled('State') is True
dp.stop_poll_command('State')
sleep(0.1)
assert dp.is_command_polled('State') is False

assert dp.is_attribute_polled('double_scalar_w') is False
dp.poll_attribute('double_scalar_w', 10)
sleep(0.1)
assert dp.get_attribute_poll_period('double_scalar_w') == 10
assert dp.is_attribute_polled('double_scalar_w') is True
dp.stop_poll_attribute('double_scalar_w')
sleep(0.1)
assert dp.is_attribute_polled('double_scalar_w') is False
dp.poll_command('State', 10)
sleep(0.1)
poll_status = dp.polling_status()
assert len(poll_status) == 2
assert poll_status[0][:27] == 'Polled command name = State'
assert poll_status[1][:29] == 'Polled attribute name = State'
dp.stop_poll_command('State')

dp.poll_command('State', 10)
sleep(0.8)
history = dp.command_history("State", 5)
dp.stop_poll_command('State')
assert len(history) == 5
h0 = history[0]
assert isinstance(h0.get_date(), tango.TimeVal)
assert h0.get_type() == tango.CmdArgType.DevState
assert h0.has_failed() is False
assert h0.is_empty() is False

dp.write_attribute('double_scalar_w', 3.142)
dp.poll_attribute('double_scalar_w', 10)
sleep(0.8)
ahist = dp.attribute_history('double_scalar_w', 5)
dp.stop_poll_attribute('double_scalar_w')
assert len(ahist) == 5
attr = ahist[3]
assert attr.data_format == tango._tango.AttrDataFormat.SCALAR
assert attr.dim_x == 1
assert attr.dim_y == 0
assert attr.has_failed is False
assert attr.is_empty is False
assert attr.name == 'double_scalar_w'
assert attr.nb_read == 1
assert attr.nb_written == 1
assert attr.quality == tango._tango.AttrQuality.ATTR_VALID
assert isinstance(attr.r_dimension, tango.AttributeDimension)
assert isinstance(attr.time, tango.TimeVal)
assert attr.type == tango._tango.CmdArgType.DevDouble
# assert attr.value == 3.142
assert attr.w_dim_x == 1
assert attr.w_dim_y == 0
assert isinstance(attr.w_dimension, tango.AttributeDimension)
# assert attr.w_value == 3.142

args = [('long_scalar_w', 4567), ('double_scalar_w', 12.568), ('boolean_scalar', True)]
dp.write_attributes(args)
id = dp.read_attributes_asynch(['long_scalar_w', 'double_scalar_w', 'boolean_scalar'])
sleep(0.05)
attrs = dp.read_attributes_reply(id)
vals = [attr.value for attr in attrs]
assert vals == [4567, 12.568, True]
args = [('long_scalar_w', 1289), ('double_scalar_w', 9.426), ('boolean_scalar', False)]
dp.write_attributes(args)
id = dp.read_attributes_asynch(['long_scalar_w', 'double_scalar_w', 'boolean_scalar'])
attrs = dp.read_attributes_reply(id, 50)
vals = [attr.value for attr in attrs]
assert vals == [1289, 9.426, False]
with pytest.raises(Exception):
    id = dp.read_attributes_asynch(['long_scalar_w', 'double_scalar_w', 'boolean_scalar'])
    attrs = dp.read_attributes_reply(id, 1)  # should timeout and raise an exception
assert dp.pending_asynch_call(tango.asyn_req_type.ALL_ASYNCH) == 1

args = [('long_scalar_w', 1289), ('double_scalar_w', 9.426), ('boolean_scalar', False)]
id = dp.write_attributes_asynch(args)
sleep(0.05)
dp.write_attributes_reply(id)
args = [('long_scalar_w', 1289)]
id = dp.write_attributes_asynch(args)
with pytest.raises(Exception):
    dp.write_attributes_reply(id, 1)  # should timeout and raise an exception

# dp.read_attributes_asynch
# dp.write_attributes_asynch

dp.add_logging_target("file::/tmp/testscript-logfile")
targets = dp.get_logging_target()
assert targets[0] == "file::/tmp/testscript-logfile"
dp.remove_logging_target("file::/tmp/testscript-logfile")
targets = dp.get_logging_target()
assert len(targets) == 0
dp.set_logging_level(1)
assert dp.get_logging_level() == 1

# dp.subscribe_event
# dp.subscribe_event
# dp.unsubscribe_event
# dp.get_callback_events
# dp.get_attr_conf_events
# dp.get_data_events
# dp.get_data_ready_events
# dp.get_pipe_events
# dp.get_devintr_change_events
# dp.event_queue_size
# dp.get_last_event_date
# assert dp.is_event_queue_empty(1) is True

assert dp.is_locked() is False
dp.lock()
assert dp.is_locked() is True
assert dp.is_locked_by_me() is True
lock_status = dp.locking_status()
assert lock_status[:54] == 'Device sys/tg_test/1 is locked by CPP or Python client'
dp.unlock()
assert dp.is_locked() is False
assert dp.locking_status() == 'Device sys/tg_test/1 is not locked'

#
# Connection methods
#
assert dp.get_db_host() == socket.gethostname()
assert dp.get_db_port() == '10000'
assert dp.get_db_port_num() == 10000
assert dp.get_from_env_var() is True
# assert dp.get_fqdn() == ''
assert dp.is_dbase_used() is True
assert dp.get_dev_host() == 'IOR'
assert dp.get_dev_port() == 'IOR'
assert dp.get_idl_version() == 5
assert dp.get_timeout_millis() == 3000
dp.set_timeout_millis(4000)
assert dp.get_timeout_millis() == 4000
dp.set_timeout_millis(3000)
assert dp.get_source() == tango._tango.DevSource.CACHE_DEV
assert dp.get_transparency_reconnection() is True
dp.set_transparency_reconnection(False)
assert dp.get_transparency_reconnection() is False
dp.set_transparency_reconnection(True)
# dp.connect()
# dp.reconnect()
# dp.set_source()

assert dp.state() == DevState.RUNNING
dp.command_inout("SwitchStates")
assert dp.state() == DevState.FAULT
dp.command_inout("SwitchStates")
assert dp.state() == DevState.RUNNING

assert dp.command_inout("DevDouble", 3.142) == 3.142
assert dp.command_inout("DevBoolean", False) is False
assert dp.command_inout("DevLong64", 123456789) == 123456789
assert dp.command_inout("DevShort", 32767) == 32767

assert dp.command_inout_asynch("DevDouble", 3.142, True) == 0
id = dp.command_inout_asynch("DevLong64", 123456789)
reply = dp.command_inout_reply(id, 50)
id = dp.command_inout_asynch("DevLong", 123456)
sleep(0.05)
assert dp.command_inout_reply(id) == 123456
with pytest.raises(Exception):
    id = dp.command_inout_asynch("DevShort", 32767)
    dp.command_inout_reply(id)


# dp.command_inout_asynch_cb()
# dp.command_inout_asynch_cb()
# reply = dp.get_asynch_replies()
# dp.cancel_asynch_request(id)
# dp.cancel_all_polling_asynch_request()
assert dp.get_access_control() == tango._tango.AccessControlType.ACCESS_WRITE
# dp.set_access_control()
assert dp.get_access_right() == tango._tango.AccessControlType.ACCESS_WRITE
print("passed")