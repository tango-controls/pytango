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
from time import sleep
from tango import DeviceProxy
from tango import EventType


def push_event(ev):
    print("did I get an event")
    if ev.attr_value is not None and ev.attr_value.value is not None:
        assert ev.device.name() == expected_device
        assert ev.attr_name.endswith(expected_attr_name)
        assert ev.event == expected_event_type
        assert ev.attr_value.name == expected_attr_name
        assert ev.attr_value.value == expected_attr_value
        assert ev.attr_value.quality == expected_quality
        assert ev.err is False
        if ev.err:
            assert ev.errors != []


dp = DeviceProxy('sys/tg_test/1')
cb = push_event

expected_device = dp.name()
expected_event_type = "periodic"
expected_attr_name = "double_spectrum"
#expected_attr_value = 152.34
expected_quality = tango.AttrQuality.ATTR_VALID

dp.poll_attribute('double_spectrum', 100)
print("polling")
event_id = dp.subscribe_event("double_spectrum", EventType.PERIODIC_EVENT, cb)
print("subscribed")
for i in range(10):
    sleep(0.5)
dp.unsubscribe_event(event_id)
dp.stop_poll_attribute('double_spectrum')
print("passed periodic spectrum events")

expected_device = dp.name()
expected_event_type = "periodic"
expected_attr_name = "double_scalar"
expected_attr_value = 152.34
expected_quality = tango.AttrQuality.ATTR_VALID

dp.poll_attribute('double_scalar', 100)
event_id = dp.subscribe_event("double_scalar", EventType.PERIODIC_EVENT, cb)
for i in range(10):
    sleep(0.5)
dp.unsubscribe_event(event_id)
print("passed periodic scalar events")

expected_event_type = "change"
event_id = dp.subscribe_event("double_scalar", EventType.CHANGE_EVENT, cb)
for i in range(5):
    expected_attr_value += 1.0
    dp.write_attribute("double_scalar", expected_attr_value)
    for j in range(5):
        # wait for the event
        sleep(0.5)
dp.unsubscribe_event(event_id)
print("passed change scalar events")

expected_event_type = "archive"
event_id = dp.subscribe_event("double_scalar", EventType.ARCHIVE_EVENT, cb)
for i in range(5):
    expected_attr_value += 1.0
    dp.write_attribute("double_scalar", expected_attr_value)
    for j in range(5):
        # wait for the event
        sleep(0.5)
dp.unsubscribe_event(event_id)
print("passed archive scalar events")


class EventManager:
    def push_event(self, ev):
        if ev.attr_value is not None and ev.attr_value.value is not None:
            assert ev.device.name() == expected_device
            assert ev.attr_name.endswith(expected_attr_name)
            assert ev.event == expected_event_type
            assert ev.attr_value.name == expected_attr_name
            print(ev.attr_value.value)
            assert ev.attr_value.value == expected_attr_value
            assert ev.attr_value.quality == expected_quality
            assert ev.err is False
            if ev.err:
                assert ev.errors != []


expected_event_type = "periodic"
cb = EventManager()
event_id = dp.subscribe_event("double_scalar", EventType.PERIODIC_EVENT, cb)
for i in range(10):
    sleep(0.5)
dp.unsubscribe_event(event_id)
# unsubscribe before stopping poll & reset attribute
dp.stop_poll_attribute('double_scalar')
dp.write_attribute("double_scalar", 152.34)

print("passed events")
