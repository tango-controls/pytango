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
import numpy as np
from time import sleep
from tango import DeviceProxy
from tango import DevState
from tango import Database
from tango import DbDatum
from tango import Release
from tango import EventType

assert tango.ApiUtil.instance().is_event_consumer_created() == False
assert "tcfidell11:10000" == tango.ApiUtil.get_env_var("TANGO_HOST")
assert tango.ApiUtil.instance().get_user_connect_timeout() == -1
tango.ApiUtil.instance().set_asynch_cb_sub_model(tango.cb_sub_model.PULL_CALLBACK)
assert tango.cb_sub_model.PULL_CALLBACK == tango.ApiUtil.instance().get_asynch_cb_sub_model()
tango.ApiUtil.instance().set_asynch_cb_sub_model(tango.cb_sub_model.PUSH_CALLBACK)
assert tango.cb_sub_model.PUSH_CALLBACK == tango.ApiUtil.instance().get_asynch_cb_sub_model()
assert tango.ApiUtil.instance().pending_asynch_call(tango.asyn_req_type.ALL_ASYNCH) == 0

dp = DeviceProxy('sys/tg_test/1')

def cmd_ended():
    executed += 1
    print("cmd_ended called back")

tango.ApiUtil.instance().set_asynch_cb_sub_model(tango.cb_sub_model.PUSH_CALLBACK)
#CallBackPushEvent* cb = new CallBackPushEvent();
dp.command_inout_asynch("Status", cmd_ended)
for i in range(5):
    sleep(1)
    print("sleep")

dp.get_asynch_replies()

print("passed")