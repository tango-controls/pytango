
# Imports

import time
import socket
from functools import partial

import pytest
from six import StringIO

from tango import EventType, GreenMode, DeviceProxy, AttrQuality
from tango.server import Device
from tango.server import command, attribute
from tango.test_utils import DeviceTestContext
from tango.utils import EventCallback

from tango.gevent import DeviceProxy as gevent_DeviceProxy
from tango.futures import DeviceProxy as futures_DeviceProxy
from tango.asyncio import DeviceProxy as asyncio_DeviceProxy

# Helpers

device_proxy_map = {
    GreenMode.Synchronous: DeviceProxy,
    GreenMode.Futures: futures_DeviceProxy,
    GreenMode.Asyncio: partial(asyncio_DeviceProxy, wait=True),
    GreenMode.Gevent: gevent_DeviceProxy}


def get_open_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


# Test device

class EventDevice(Device):

    def init_device(self):
        self.set_change_event("attr", True, False)

    @attribute
    def attr(self):
        return 0.

    @command
    def send_event(self):
        self.push_change_event("attr", 1.)

    @command
    def send_event_with_timestamp(self):
        self.push_change_event("attr", 2., 3., AttrQuality.ATTR_WARNING)

    @command(dtype_in=str)
    def add_dyn_attr(self, name):
        attr = attribute(
            name=name,
            dtype='float',
            fget=self.read)
        self.add_attribute(attr)

    @command(dtype_in=str)
    def delete_dyn_attr(self, name):
        self._remove_attribute(name)

    def read(self, attr):
        attr.set_value(1.23)


# Device fixture

@pytest.fixture(params=[GreenMode.Synchronous,
                        GreenMode.Futures,
                        GreenMode.Asyncio,
                        GreenMode.Gevent],
                scope="module")
def event_device(request):
    green_mode = request.param
    # Hack: a port have to be specified explicitely for events to work
    port = get_open_port()
    context = DeviceTestContext(EventDevice, port=port, process=True)
    with context:
        yield device_proxy_map[green_mode](context.get_device_access())


# Tests

def test_subscribe_change_event(event_device):
    results = []

    def callback(evt):
        results.append(evt.attr_value.value)

    # Subscribe
    eid = event_device.subscribe_event(
        "attr", EventType.CHANGE_EVENT, callback, wait=True)
    assert eid == 1
    # Trigger an event
    event_device.command_inout("send_event", wait=True)
    # Wait for tango event
    retries = 20
    for _ in range(retries):
        event_device.read_attribute("state", wait=True)
        if len(results) > 1:
            break
        time.sleep(0.05)
    # Test the event values
    assert results == [0., 1.]
    # Unsubscribe
    event_device.unsubscribe_event(eid)


def test_subscribe_interface_event(event_device):
    results = []

    def callback(evt):
        results.append(evt)

    # Subscribe
    eid = event_device.subscribe_event(
        "attr", EventType.INTERFACE_CHANGE_EVENT, callback, wait=True)
    assert eid == 1
    # Trigger an event
    event_device.command_inout("add_dyn_attr", 'bla', wait=True)
    event_device.read_attribute('bla', wait=True) == 1.23
    # Wait for tango event
    retries = 30
    for _ in range(retries):
        event_device.read_attribute("state", wait=True)
        if len(results) > 1:
            break
        time.sleep(0.05)
    event_device.command_inout("delete_dyn_attr", 'bla', wait=True)
    # Wait for tango event
    retries = 30
    for _ in range(retries):
        event_device.read_attribute("state", wait=True)
        if len(results) > 2:
            break
        time.sleep(0.05)
    # Test the first event value
    assert set(cmd.cmd_name for cmd in results[0].cmd_list) == \
        {'Init', 'State', 'Status',
         'add_dyn_attr', 'delete_dyn_attr',
         'send_event', 'send_event_with_timestamp'}
    assert set(att.name for att in results[0].att_list) == \
        {'attr', 'State', 'Status'}
    # Test the second event value
    assert set(cmd.cmd_name for cmd in results[1].cmd_list) == \
        {'Init', 'State', 'Status',
         'add_dyn_attr', 'delete_dyn_attr',
         'send_event', 'send_event_with_timestamp'}
    assert set(att.name for att in results[1].att_list) == \
        {'attr', 'State', 'Status', 'bla'}
    # Test the third event value
    assert set(cmd.cmd_name for cmd in results[2].cmd_list) == \
        {'Init', 'State', 'Status',
         'add_dyn_attr', 'delete_dyn_attr',
         'send_event', 'send_event_with_timestamp'}
    assert set(att.name for att in results[2].att_list) == \
        {'attr', 'State', 'Status'}
    # Unsubscribe
    event_device.unsubscribe_event(eid)


def test_push_event_with_timestamp(event_device):
    string = StringIO()
    ec = EventCallback(fd=string)
    # Subscribe
    eid = event_device.subscribe_event(
        "attr", EventType.CHANGE_EVENT, ec, wait=True)
    assert eid == 1
    # Trigger an event
    event_device.command_inout("send_event_with_timestamp", wait=True)
    # Wait for tango event
    retries = 20
    for _ in range(retries):
        event_device.read_attribute("state", wait=True)
        if len(ec.get_events()) > 1:
            break
        time.sleep(0.05)
    # Test the event values and timestamp
    results = [evt.attr_value.value for evt in ec.get_events()]
    assert results == [0., 2.]
    assert ec.get_events()[-1].attr_value.time.totime() == 3.
    # Check string
    line1 = "TEST/NODB/EVENTDEVICE ATTR#DBASE=NO CHANGE [ATTR_VALID] 0.0"
    line2 = "TEST/NODB/EVENTDEVICE ATTR#DBASE=NO CHANGE [ATTR_WARNING] 2.0"
    assert line1 in string.getvalue()
    assert line2 in string.getvalue()
    # Unsubscribe
    event_device.unsubscribe_event(eid)
