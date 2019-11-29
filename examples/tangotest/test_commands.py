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
import pytest
import numpy as np
from time import sleep
from tango import DeviceProxy
from tango import DevState

dp = DeviceProxy('sys/tg_test/1')
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

assert dp.state() == DevState.RUNNING
dp.command_inout("ChangeState", DevState.FAULT)
assert dp.state() == DevState.FAULT
dp.command_inout("ChangeState", DevState.RUNNING)
assert dp.state() == DevState.RUNNING

assert dp.command_inout("DevBoolean", False) is False
assert dp.command_inout("DevDouble", 3.142) == 3.142
assert dp.command_inout("DevDouble", 6.284) == 6.284
assert dp.command_inout("DevDouble", 9.426) == 9.426
assert dp.command_inout("DevFloat", 12.568) == 12.568
assert dp.command_inout("DevLong64", 123456789) == 123456789
assert dp.command_inout("DevLong", 456789) == 456789
assert dp.command_inout("DevShort", 32767) == 32767
assert dp.command_inout("DevString", "abcdefgh") == "abcdefgh"
assert dp.command_inout("DevState", tango.DevState.MOVING) == tango.DevState.MOVING
assert dp.command_inout("DevEncoded", ("format", [0, 1, 2, 3, 0xfd, 0xfe, 0xff])) == ("format", [0, 1, 2, 3, 0xfd, 0xfe, 0xff])

dvda = dp.command_inout("DevVarDoubleArray", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
assert type(dvda) == np.ndarray
assert((dvda == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).all())

dvda2 = dp.command_inout("DevVarDoubleArray", np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]))
assert type(dvda2) == np.ndarray
assert((dvda2 == np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])).all())
dvl64a = dp.command_inout("DevVarLong64Array", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
assert type(dvl64a) == np.ndarray
assert((dvl64a == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]).all())
dvsa = dp.command_inout("DevVarStringArray", ["abc", "def", "ghi", "jkl", "mno"])
assert dvsa == ["abc", "def", "ghi", "jkl", "mno"]
dvlsa = dp.command_inout("DevVarLongStringArray", ([1, 2, 3, 4, 5], ["abc", "def", "ghi", "jkl", "mno"]))
assert dvlsa == ([1, 2, 3, 4, 5], ["abc", "def", "ghi", "jkl", "mno"])
dvdsa = dp.command_inout("DevVarDoubleStringArray", ([0.3, 0.4, 0.5, 0.6, 0.7], ["abc", "def", "ghi", "jkl", "mno"]))
assert dvdsa == ([0.3, 0.4, 0.5, 0.6, 0.7], ["abc", "def", "ghi", "jkl", "mno"])

assert dp.command_inout_asynch("DevDouble", 3.142, True) == 0
id = dp.command_inout_asynch("DevLong64", 1234567890)
assert dp.command_inout_reply(id, 50) == 1234567890
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

print("passed commands tests")
