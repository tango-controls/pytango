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
assert dp.description() == u'A Tango device'
assert dp.name() == u'sys/tg_test/1'
# with pytest.raises(tango.DevFailed):
try:
    dp.alias()  # Not aliased therefore raise exception
except tango.DevFailed:
    print("got it")

info = dp.info()
assert info.dev_class == "TangoTest"
assert info.server_id == "TangoTest/test"
assert info.server_host == socket.gethostname()
assert info.server_version == 5
print("got to here")
tg_version = dp.get_tango_lib_version()
print(tg_version)
assert tg_version == 902
print("and here")
duration = dp.ping(wait=True)
if Release.version_number > 924:
    assert isinstance(duration, long)
else:
    assert isinstance(duration, int)

assert len(dp.black_box(5)) == 5
print(dp.get_command_list())
assert dp.get_command_list() == [u'ChangeState', u'DevBoolean', u'DevDouble', u'DevEncoded', u'DevFloat',
                                 u'DevLong', u'DevLong64', u'DevShort', u'DevState',
                                 u'DevString', u'DevULong', u'DevULong64', u'DevUShort',
                                 u'DevVarCharArray', u'DevVarDoubleArray', u'DevVarDoubleStringArray',
                                 u'DevVarFloatArray', u'DevVarLong64Array', u'DevVarLongArray',
                                 u'DevVarLongStringArray', u'DevVarShortArray', u'DevVarStringArray',
                                 u'DevVarULong64Array', u'DevVarULongArray', u'DevVarUShortArray',
                                 u'DevVoid', u'Init', u'PushPipeEvents', u'PushScalarArchiveEvents',
                                 u'PushScalarChangeEvents', u'Randomise', u'State', u'Status',
                                 ]
info = dp.get_command_config('DevDouble')
assert info.cmd_name == u'DevDouble'
assert info.in_type == tango.CmdArgType.DevDouble
assert info.out_type == tango.CmdArgType.DevDouble
assert info.in_type_desc == u'A DevDouble value'
assert info.out_type_desc == u'Echo of the input value'
assert info.disp_level == tango.DispLevel.OPERATOR

info_list = dp.get_command_config(['DevDouble', 'DevLong64', 'DevVarShortArray'])
assert len(info_list) == 3
assert info_list[0].cmd_name == u'DevDouble'
assert info_list[0].in_type == tango.CmdArgType.DevDouble
assert info_list[0].out_type == tango.CmdArgType.DevDouble
assert info_list[0].in_type_desc == u'A DevDouble value'
assert info_list[0].out_type_desc == u'Echo of the input value'
assert info_list[0].disp_level == tango.DispLevel.OPERATOR
assert info_list[1].cmd_name == u'DevLong64'
assert info_list[1].in_type == tango.CmdArgType.DevLong64
assert info_list[1].out_type == tango.CmdArgType.DevLong64
assert info_list[1].in_type_desc == u'A DevLong64 value'
assert info_list[1].out_type_desc == u'Echo of the input value'
assert info_list[1].disp_level == tango.DispLevel.OPERATOR
assert info_list[2].cmd_name == u'DevVarShortArray'
assert info_list[2].in_type == tango.CmdArgType.DevVarShortArray
assert info_list[2].out_type == tango.CmdArgType.DevVarShortArray
assert info_list[2].in_type_desc == u'An array of short values'
assert info_list[2].out_type_desc == u'Echo of the input values'
assert info_list[2].disp_level == tango.DispLevel.OPERATOR

info = dp.command_query('DevDouble')
assert info.cmd_name == u'DevDouble'
assert info.in_type == tango.CmdArgType.DevDouble
assert info.out_type == tango.CmdArgType.DevDouble
assert info.in_type_desc == u'A DevDouble value'
assert info.out_type_desc == u'Echo of the input value'
assert info.disp_level == tango.DispLevel.OPERATOR

cmd_info_list = dp.command_list_query()
assert len(cmd_info_list) == 33
info = cmd_info_list[2]
assert info.cmd_name == u'DevDouble'
assert info.in_type == tango.CmdArgType.DevDouble
assert info.out_type == tango.CmdArgType.DevDouble
assert info.in_type_desc == u'A DevDouble value'
assert info.out_type_desc == u'Echo of the input value'
assert info.disp_level == tango.DispLevel.OPERATOR
print("herehere")
info = dp.import_info()
print(info)
# assert info.name == 'sys/tg_test/1'
# assert info.exported == 1
# assert info.ior[:4] == 'IOR:'
# assert info.version == '5'

db_datum = DbDatum('test')
db_datum.value_string = ['3.142']
print('55')
# dp.put_property(db_datum)
# print("5")
# props = dp.get_property('test')
# print("5")
# assert props.keys() == ["test"]
# assert props.values() == [['3.142']]
# plist = dp.get_property_list("*")
# print(plist)
# assert len(plist) == 2 and plist[1] == 'test'
# db_datum = DbDatum('timeout')
# db_datum.value_string = ['0.05']
# dp.put_property(db_datum)
# props = dp.get_property(['test', 'timeout'])
# assert props.keys() == ['test', 'timeout']
# assert props.values() == [['3.142'], ['0.05']]
# plist = dp.get_property_list("*")
# assert len(plist) == 3 and plist[1] == 'test' and plist[2] == 'timeout'
# db_datum = DbDatum('test')
# props = dp.get_property(db_datum)
# assert props.keys() == ["test"]
# assert props.values() == [['3.142']]
# db_datum_list = [DbDatum('test'), DbDatum('timeout')]
# props = dp.get_property(db_datum_list)
# assert props.keys() == ["test", "timeout"]
# assert props.values() == [['3.142'], ['0.05']]
# dp.delete_property('test')
# plist = dp.get_property_list("*")
# assert len(plist) == 2 and plist[1] == 'timeout'
# db_datum = DbDatum('test')
# db_datum.value_string = ['3.142']
# dp.put_property(db_datum)
# dp.delete_property(['test', 'timeout'])
# plist = dp.get_property_list("*")
# assert len(plist) == 1
# db_datum1 = DbDatum('test')
# db_datum1.value_string = ['3.142']
# db_datum2 = DbDatum('timeout')
# db_datum2.value_string = ['0.05']
# dp.put_property([db_datum1, db_datum2])
# db_datum = DbDatum('test')
# dp.delete_property(db_datum)
# plist = dp.get_property_list("*")
# assert len(plist) == 2 and plist[1] == 'timeout'
# props = {'test': '3.142'}
# dp.put_property(props)
# db_datum_list = [DbDatum('test'), DbDatum('timeout')]
# props = dp.get_property(db_datum_list)
# dp.delete_property(db_datum_list)
# plist = dp.get_property_list("*")
# assert len(plist) == 1
# props = {'test': ['3.142'], 'timeout': ['0.05']}
# dp.put_property(props)
# plist = dp.get_property_list("*")
# assert len(plist) == 3 and plist[1] == 'test' and plist[2] == 'timeout'
# props = {'test': ['3.142'], 'timeout': ['0.05']}
# dp.delete_property(db_datum_list)
# plist = dp.get_property_list("*")
# assert len(plist) == 1

# assert dp.get_pipe_list() == ['string_long_short_ro']
# pipe_info = dp.get_pipe_config('string_long_short_ro')
# assert pipe_info.description == 'Pipe example'
# assert pipe_info.disp_level == tango.DispLevel.OPERATOR
# assert pipe_info.extensions == []
# assert pipe_info.label == 'string_long_short_ro'
# assert pipe_info.name == 'string_long_short_ro'
# assert pipe_info.writable == tango.PipeWriteType.PIPE_READ
# pipe_info_list = dp.get_pipe_config(['string_long_short_ro', 'string_long_short_ro'])
# assert len(pipe_info_list) == 2
# pinfo = pipe_info_list[0]
# assert pinfo.description == 'Pipe example'
# assert pinfo.disp_level == tango.DispLevel.OPERATOR
# assert pinfo.extensions == []
# assert pinfo.label == 'string_long_short_ro'
# assert pinfo.name == 'string_long_short_ro'
# assert pinfo.writable == tango.PipeWriteType.PIPE_READ
# 
# # dp.set_pipe_config([])
# blob = dp.read_pipe('string_long_short_ro')
# assert blob == ('', [{'dtype': tango._tango.CmdArgType.DevString, 'name': 'FirstDE', 'value': 'The string'},
#                      {'dtype': tango._tango.CmdArgType.DevLong, 'name': 'SecondDE', 'value': 666},
#                      {'dtype': tango._tango.CmdArgType.DevShort, 'name': 'ThirdDE', 'value': 12}])
# dp.write_pipe()
#
assert dp.get_attribute_list() == [u'devstate_scalar', u'encoded_string_scalar',
                                   u'long64_scalar', u'double_scalar',
                                   u'boolean_spectrum', u'long_scalar_ro',
                                   u'boolean_scalar', u'encoded_string_scalar_ro',
                                   u'ulong_spectrum', u'long_scalar',
                                   u'devstate_spectrum', u'uchar_scalar',
                                   u'uchar_spectrum', u'long64_scalar_ro',
                                   u'ulong64_spectrum', u'ulong_scalar',
                                   u'string_scalar', u'long64_spectrum',
                                   u'string_image', u'ulong64_scalar',
                                   u'ulong_image', u'encoded_byte_scalar',
                                   u'short_scalar_ro', u'ushort_image',
                                   u'ulong_scalar_ro', u'ushort_scalar',
                                   u'long_image', u'ulong64_scalar_ro',
                                   u'encoded_byte_scalar_ro', u'float_scalar',
                                   u'float_scalar_ro', u'boolean_image',
                                   u'ulong64_image', u'short_spectrum',
                                   u'ushort_spectrum', u'short_scalar',
                                   u'ushort_scalar_ro', u'float_image',
                                   u'double_scalar_ro', u'long64_image',
                                   u'string_spectrum', u'float_spectrum',
                                   u'double_spectrum', u'double_image',
                                   u'long_spectrum', u'uchar_image', u'short_image',
                                   u'State', u'Status']
print"got here7"
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
assert reply.description == 'A Tango::DevFloat scalar attribute'
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
assert float(reply.max_value) == 4096.0
assert float(reply.min_value) == 0.0
assert reply.memorized == tango._tango.AttrMemorizedType.NONE
assert reply.min_alarm == 'Not specified'
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
assert reply[1].description == 'A Tango::DevLong scalar attribute (int32)'
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
attr_info = dp.attribute_query('string_scalar')
assert type(attr_info) == tango._tango.AttributeInfoEx

# dp.attribute_list_query([]);
# dp.attribute_list_query_ex([])
# dp.set_attribute_config[]
# dp.set_attribute_config[]
dp.write_attribute('long_scalar', 23456)
attr = dp.read_attribute('long_scalar')
assert attr.data_format == tango._tango.AttrDataFormat.SCALAR
assert attr.dim_x == 1
assert attr.dim_y == 0
assert attr.has_failed == False
assert attr.is_empty == False
assert attr.name == 'long_scalar'
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
print('88')
dp.write_attribute('long64_scalar', -987654321)
assert dp.read_attribute('long64_scalar').value == -987654321
dp.write_attribute('float_scalar',45.67)
assert dp.read_attribute('float_scalar').value == pytest.approx(45.67, rel=1e-3)
dp.write_attribute('ulong_scalar',987654)
assert dp.read_attribute('ulong_scalar').value == 987654
dp.write_attribute('ulong64_scalar',987654321)
assert dp.read_attribute('ulong64_scalar').value == 987654321
dp.write_attribute('ushort_scalar',65535)
assert dp.read_attribute('ushort_scalar').value == 65535
dp.write_attribute('uchar_scalar',255)
assert dp.read_attribute('uchar_scalar').value == 255
 
assert dp.read_attribute('status').value == 'The device is in RUNNING state.'
assert dp.read_attribute('state').value == DevState.RUNNING
dp.write_attribute('string_scalar', "AbcdefghijklmnopqrstuvwxyZ")
assert dp.read_attribute('string_scalar').value == "AbcdefghijklmnopqrstuvwxyZ"
#dp.write_attribute('encoded_byte_scalar',("Array of Bytes", ([1,2,3,4,5,6,7,8,9])))
# attr = dp.read_attribute('encoded_byte_scalar')
# assert attr.value[0] == 'Array of Bytes'
# assert attr.value[1] == [1,2,3,4,5,6,7,8,9]
print('888')

dp.write_attribute('double_scalar', 3.142)
dp.write_attribute('long_scalar', 1357)
dp.write_attribute('boolean_scalar', True)
attrs = dp.read_attributes(['long_scalar', 'double_scalar', 'boolean_scalar'])
vals = [attr.value for attr in attrs]
assert vals == [1357, 3.142, True]
args = [('long_scalar', 8642), ('double_scalar', 6.284), ('boolean_scalar', False)]
dp.write_attributes(args)
attrs = dp.read_attributes(['long_scalar', 'double_scalar', 'boolean_scalar'])
vals = [attr.value for attr in attrs]
assert vals == [8642, 6.284, False]
assert dp.write_read_attribute('long_scalar', 98765).value == 98765
args = [('long_scalar', 3459), ('double_scalar', 15.71), ('boolean_scalar', True)]
attrs = dp.write_read_attributes(args)
vals = [attr.value for attr in attrs]
assert vals == [3459, 15.71, True]

# dp.stop_poll_command('State')
# assert dp.is_command_polled('State') is False
# dp.poll_command('State', 10)
# sleep(0.1)
# assert dp.get_command_poll_period('State') == 10
# assert dp.is_command_polled('State') is True
# dp.stop_poll_command('State')
# sleep(0.1)
# assert dp.is_command_polled('State') is False

# dp.stop_poll_attribute('double_scalar')
# assert dp.is_attribute_polled('double_scalar') is False
# dp.poll_attribute('double_scalar', 10)
# sleep(0.1)
# assert dp.get_attribute_poll_period('double_scalar') == 10
# assert dp.is_attribute_polled('double_scalar') is True
# dp.stop_poll_attribute('double_scalar')
# sleep(0.1)
# assert dp.is_attribute_polled('double_scalar') is False

# dp.poll_command('State', 10)
# sleep(0.1)
# poll_status = dp.polling_status()
# assert len(poll_status) == 2
# assert poll_status[0][:27] == 'Polled command name = State'
# assert poll_status[1][:29] == 'Polled attribute name = State'
# dp.stop_poll_command('State')
# 
# dp.poll_command('State', 10)
# sleep(0.8)
# history = dp.command_history("State", 5)
# dp.stop_poll_command('State')
# assert len(history) == 5
# h0 = history[0]
# assert isinstance(h0.get_date(), tango.TimeVal)
# assert h0.get_type() == tango.CmdArgType.DevState
# assert h0.has_failed() is False
# assert h0.is_empty() is False

# dp.write_attribute('double_scalar', 3.142)
# dp.poll_attribute('double_scalar', 10)
# sleep(0.8)
# ahist = dp.attribute_history('double_scalar', 5)
# dp.stop_poll_attribute('double_scalar')
# assert len(ahist) == 5
# attr = ahist[3]
# assert attr.data_format == tango._tango.AttrDataFormat.SCALAR
# assert attr.dim_x == 1
# assert attr.dim_y == 0
# assert attr.has_failed is False
# assert attr.is_empty is False
# assert attr.name == 'double_scalar'
# assert attr.nb_read == 1
# assert attr.nb_written == 1
# assert attr.quality == tango._tango.AttrQuality.ATTR_VALID
# assert isinstance(attr.r_dimension, tango.AttributeDimension)
# assert isinstance(attr.time, tango.TimeVal)
# assert attr.type == tango._tango.CmdArgType.DevDouble
# # assert attr.value == 3.142
# assert attr.w_dim_x == 1
# assert attr.w_dim_y == 0
# assert isinstance(attr.w_dimension, tango.AttributeDimension)
# # assert attr.w_value == 3.142

args = [('long_scalar', 4567), ('double_scalar', 12.568), ('boolean_scalar', True)]
dp.write_attributes(args)
id = dp.read_attributes_asynch(['long_scalar', 'double_scalar', 'boolean_scalar'])
sleep(0.05)
attrs = dp.read_attributes_reply(id)
vals = [attr.value for attr in attrs]
assert vals == [4567, 12.568, True]
args = [('long_scalar', 1289), ('double_scalar', 9.426), ('boolean_scalar', False)]
dp.write_attributes(args)
id = dp.read_attributes_asynch(['long_scalar', 'double_scalar', 'boolean_scalar'])
attrs = dp.read_attributes_reply(id, 50)
vals = [attr.value for attr in attrs]
assert vals == [1289, 9.426, False]
with pytest.raises(Exception):
    id = dp.read_attributes_asynch(['long_scalar', 'double_scalar', 'boolean_scalar'])
    attrs = dp.read_attributes_reply(id, 1)  # should timeout and raise an exception
assert dp.pending_asynch_call(tango.asyn_req_type.ALL_ASYNCH) == 1

args = [('long_scalar', 1289), ('double_scalar', 9.426), ('boolean_scalar', False)]
id = dp.write_attributes_asynch(args)
sleep(0.05)
dp.write_attributes_reply(id)
args = [('long_scalar', 1289)]
id = dp.write_attributes_asynch(args)
with pytest.raises(Exception):
    dp.write_attributes_reply(id, 1)  # should timeout and raise an exception

# dp.read_attributes_asynch
# dp.write_attributes_asynch

# dp.add_logging_target("file::/tmp/testscript-logfile")
# targets = dp.get_logging_target()
# assert targets[0] == "file::/tmp/testscript-logfile"
# dp.remove_logging_target("file::/tmp/testscript-logfile")
# targets = dp.get_logging_target()
# assert len(targets) == 0
# dp.set_logging_level(1)
# assert dp.get_logging_level() == 1

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

print("passed")