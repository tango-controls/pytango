# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2019 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

import tango
import socket
import pytest
from time import sleep
from tango import DeviceProxy
from tango import DevState
from tango import Database
from tango import DbDatum
from tango import Release
from tango import AccessControlType

dp = DeviceProxy('sys/tg_test/1')
assert dp.dev_name() == u'sys/tg_test/1'
assert dp.state() == DevState.RUNNING
assert dp.adm_name() == u'dserver/TangoTest/test'
assert dp.description() == u'A Tango device'
assert dp.name() == u'sys/tg_test/1'
assert dp.status() == u'The device is in RUNNING state.'
with pytest.raises(tango.DevFailed):
    dp.alias()  # Not aliased therefore raise exception
info = dp.info()
assert info.dev_class == "TangoTest"
assert info.server_id == "TangoTest/test"
assert info.server_host == socket.gethostname()
assert info.server_version == 5
tg_version = dp.get_tango_lib_version()
assert tg_version == 902
duration = dp.ping()
sleep(0.25)
assert duration < 500
assert len(dp.black_box(5)) == 5
#
# Connection methods
#
assert dp.get_db_host() == socket.gethostname()
assert dp.get_db_port() == '10000'
assert dp.get_db_port_num() == 10000
assert dp.get_from_env_var() is True
assert dp.get_fqdn() == ''
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

info = dp.import_info()
assert info.name == 'sys/tg_test/1'
assert info.exported == 1
assert info.ior[:4] == 'IOR:'
assert info.version == '5'

db_datum = DbDatum()
assert db_datum.is_empty() == True
assert db_datum.size() == 0
db_datum = DbDatum('test')
db_datum.value_string = ['3.142']
assert db_datum.is_empty() == False
assert db_datum.size() == 1
dp.put_property(db_datum)
props = dp.get_property('test')
assert list(props.keys()) == ["test"]
assert list(props.values()) == [['3.142']]
db_datum = DbDatum('timeout')
db_datum.value_string = ['0.05']
dp.put_property(db_datum)
props = dp.get_property(['test', 'timeout'])
assert list(props.keys()) == ['test', 'timeout']
assert list(props.values()) == [['3.142'], ['0.05']]
db_datum = DbDatum('test')
props = dp.get_property(db_datum)
assert list(props.keys()) == ["test"]
assert list(props.values()) == [['3.142']]
db_datum_list = [DbDatum('test'), DbDatum('timeout')]
props = dp.get_property(db_datum_list)
assert list(props.keys()) == ["test", "timeout"]
assert list(props.values()) == [['3.142'], ['0.05']]

dp.delete_property('test')
plist = dp.get_property_list("*")
assert len(plist) == 2 and plist[1] == 'timeout'
db_datum = DbDatum('test')
db_datum.value_string = ['3.142']
dp.put_property(db_datum)
dp.delete_property(['test', 'timeout'])
plist = dp.get_property_list("*")
assert len(plist) == 1
db_datum1 = DbDatum('test')
db_datum1.value_string = ['3.142']
db_datum2 = DbDatum('timeout')
db_datum2.value_string = ['0.05']
dp.put_property([db_datum1, db_datum2])
db_datum = DbDatum('test')
dp.delete_property(db_datum)
plist = dp.get_property_list("*")
assert len(plist) == 2 and plist[1] == 'timeout'
props = {'test': '3.142'}
dp.put_property(props)
db_datum_list = [DbDatum('test'), DbDatum('timeout')]
props = dp.get_property(db_datum_list)
dp.delete_property(db_datum_list)
plist = dp.get_property_list("*")
assert len(plist) == 1
props = {'test': ['3.142'], 'timeout': ['0.05']}
dp.put_property(props)
plist = dp.get_property_list("*")
assert len(plist) == 3 and plist[1] == 'test' and plist[2] == 'timeout'
props = {'test': ['3.142'], 'timeout': ['0.05']}
dp.delete_property(db_datum_list)
plist = dp.get_property_list("*")
assert len(plist) == 1

# Not tested yet!!!!!!!!!!!!!
# dp.attribute_list_query([]);
# dp.attribute_list_query_ex([])
# dp.set_attribute_config[]
# dp.set_attribute_config[]

with pytest.raises(tango.DevFailed):
    dp.stop_poll_command('State')
assert dp.is_command_polled('State') is False
dp.poll_command('State', 10)
sleep(0.1)
assert dp.get_command_poll_period('State') == 10
assert dp.is_command_polled('State') is True
dp.stop_poll_command('State')
sleep(0.1)
assert dp.is_command_polled('State') is False

with pytest.raises(tango.DevFailed):
    dp.stop_poll_attribute('double_scalar')
assert dp.is_attribute_polled('double_scalar') is False
dp.poll_attribute('double_scalar', 10)
sleep(0.1)
assert dp.get_attribute_poll_period('double_scalar') == 10
assert dp.is_attribute_polled('double_scalar') is True
dp.stop_poll_attribute('double_scalar')
sleep(0.1)
assert dp.is_attribute_polled('double_scalar') is False

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

dp.write_attribute('double_scalar', 3.142)
dp.poll_attribute('double_scalar', 10)
sleep(0.8)
ahist = dp.attribute_history('double_scalar', 5)
dp.stop_poll_attribute('double_scalar')
assert len(ahist) == 5
attr = ahist[3]
assert attr.data_format == tango._tango.AttrDataFormat.SCALAR
assert attr.dim_x == 1
assert attr.dim_y == 0
assert attr.has_failed is False
assert attr.is_empty is False
assert attr.name == 'double_scalar'
assert attr.nb_read == 1
assert attr.nb_written == 1
assert attr.quality == tango._tango.AttrQuality.ATTR_VALID
assert isinstance(attr.r_dimension, tango.AttributeDimension)
assert isinstance(attr.time, tango.TimeVal)
assert attr.type == tango._tango.CmdArgType.DevDouble
assert attr.value == 3.142
assert attr.w_dim_x == 1
assert attr.w_dim_y == 0
assert isinstance(attr.w_dimension, tango.AttributeDimension)
assert attr.w_value == 3.142

# Not tested yet!!!!!!!!!!!!!
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

dp.subscribe_event
dp.subscribe_event
dp.unsubscribe_event
# Not tested yet!!!!!!!!!!!!!
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

assert dp.get_access_control() == AccessControlType.ACCESS_WRITE
dp.set_access_control(tango.AccessControlType.ACCESS_READ)
assert dp.get_access_control() == AccessControlType.ACCESS_READ
dp.set_access_control(AccessControlType.ACCESS_WRITE)
assert dp.get_access_control() == AccessControlType.ACCESS_WRITE

print("passed device_proxy tests")
