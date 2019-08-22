import tango
import socket
import pytest
from time import sleep
from tango import DeviceProxy
from tango import DevState
from tango import Database
from tango import DbDatum
from tango import Release

dp = DeviceProxy('clock/tango/1')
assert dp.dev_name() == u'clock/tango/1'
assert isinstance(dp.get_device_db(), Database)
assert dp.adm_name() == u'dserver/clock/clock'
assert dp.name() == u'clock/tango/1'

assert dp.get_command_list() == [u'Init', u'State', u'Status', u'ctime', u'rubbish']
print(dp.is_command_polled("rubbish"))
dp.command_inout("rubbish")
#print(dp.command_inout("state"))
