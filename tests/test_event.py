
# Imports

import time
import socket
from functools import partial

import pytest

from tango import EventType, GreenMode, DeviceProxy
from tango.server import Device
from tango.server import command, attribute
from tango.test_utils import DeviceTestContext

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

def test_subscribe_event(event_device):
    results = []

    def callback(evt):
        results.append(evt.attr_value.value)

    event_device.subscribe_event(
        "attr", EventType.CHANGE_EVENT, callback, wait=True)
    event_device.command_inout("send_event", wait=True)
    # Wait for tango event
    retries = 10
    for _ in range(retries):
        event_device.read_attribute("state", wait=True)
        if len(results) > 1:
            break
        time.sleep(0.05)
    # Test the event values
    assert results == [0., 1.]
