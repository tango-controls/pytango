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
