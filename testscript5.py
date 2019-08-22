import tango
import socket
import pytest
import numpy as np
from time import sleep
from tango import DeviceProxy
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
def push_event(ev):
    print("push_event called back!!!!!!!!!!!!!")
    if ev.attr_value is not None and ev.attr_value.value is not None:
        print(ev.device)
        print(ev.attr_name)
        print(ev.event)
        print(ev.attr_value.name)
        print(ev.attr_value.value)
        print(ev.attr_value.quality)
        print(ev.err)
        if ev.err:
            print(errors)
 
cb=push_event
dp.poll_attribute('double_scalar', 10)
eventId = dp.subscribe_event("double_scalar", EventType.PERIODIC_EVENT, cb)
for i in range(10):
    sleep(0.5)
dp.unsubscribe_event(eventId)
 
eventId = dp.subscribe_event("double_scalar", EventType.CHANGE_EVENT, cb)
for i in range(10):
    sleep(0.5)
dp.unsubscribe_event(eventId)
 
eventId = dp.subscribe_event("double_scalar", EventType.ARCHIVE_EVENT, cb)
for i in range(10):
    sleep(0.5)
dp.unsubscribe_event(eventId)
 
eventId1 = dp.subscribe_event("double_scalar", EventType.PERIODIC_EVENT, cb)
eventId2 = dp.subscribe_event("double_scalar", EventType.CHANGE_EVENT, cb)
for i in range(50):
    sleep(0.5)
dp.unsubscribe_event(eventId1)
dp.unsubscribe_event(eventId2)

class EventManager:
    def push_event(self, ev):
        print("push_event called back!!!!!!!!!!!!!")
        if ev.attr_value is not None and ev.attr_value.value is not None:
            print(ev.device)
            print(ev.attr_name)
            print(ev.event)
            print(ev.attr_value.name)
            print(ev.attr_value.value)
            print(ev.attr_value.quality)
            print(ev.err)
            if ev.err:
                print(errors)

cb=EventManager()
dp.poll_attribute('double_scalar', 10)
eventId = dp.subscribe_event("double_scalar", EventType.PERIODIC_EVENT, cb)
for i in range(10):
    sleep(0.5)
dp.unsubscribe_event(eventId)

print("passed")
