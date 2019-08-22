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