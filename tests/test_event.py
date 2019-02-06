
# Imports

import time
import socket
from functools import partial

import pytest
import zmq
from six import StringIO

from tango import EventType, GreenMode, DeviceProxy, AttrQuality
from tango.server import Device
from tango.server import command, attribute
from tango.test_utils import DeviceTestContext
from tango.utils import EventCallback

from tango.gevent import DeviceProxy as gevent_DeviceProxy
from tango.futures import DeviceProxy as futures_DeviceProxy
from tango.asyncio import DeviceProxy as asyncio_DeviceProxy


MAX_RETRIES = 60
TIME_PER_RETRY = 0.2


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
    context = DeviceTestContext(EventDevice, port=port, process=True, debug=5)
    with context:
        yield device_proxy_map[green_mode](context.get_device_access())


@pytest.fixture(scope="module")
def event_context(request):
    # Hack: a port have to be specified explicitely for events to work
    port = get_open_port()
    context = DeviceTestContext(EventDevice, port=port, process=True, debug=5)
    with context:
        yield context


# Tests

def test_get_hostnames():
    # Try a few options for debugging
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 0))
        ip = s.getsockname()[0]
        print("ip via UDP connect 8.8.8.8", ip)
        print("gethostbyaddr", socket.gethostbyaddr(ip))

        s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s2.connect(('10.240.0.254', 0))
        ip2 = s.getsockname()[0]
        print("ip via UDP connect 10.240.0.254", ip2)
        print("gethostbyaddr", socket.gethostbyaddr(ip2))

        s3 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s3.connect(('172.17.0.254', 0))
        ip3 = s.getsockname()[0]
        print("ip via UDP connect 172.17.0.254", ip3)
        print("gethostbyaddr", socket.gethostbyaddr(ip3))

        hostname4 = socket.gethostname()
        print("gethostname", hostname4)
        ip4 = socket.gethostbyname(hostname4)
        print("ip via gethostbyname", ip4)
    except Exception as e:
        print("Error with extra lookups", e)
    assert False


def test_subscribe_change_event(event_context):
    results = []

    def callback(evt):
        if evt.attr_value:
            results.append(evt.attr_value.value)
        else:
            print('bad event: %s' % evt)

    admin_device = DeviceProxy(event_context.get_server_access())
    event_device = DeviceProxy(event_context.get_device_access())
    # Subscribe
    eid = event_device.subscribe_event(
        "attr", EventType.CHANGE_EVENT, callback, wait=True)
    assert eid == 1

    zmq_info = admin_device.zmqeventsubscriptionchange(['info'])
    # like [[925], ['Heartbeat: tcp://172.17.0.2:35807', 'Event: tcp://172.17.0.2:44033']]
    print('ZMQ info: %s' % zmq_info)
    heartbeat_addr = zmq_info[1][0].split(' ')[-1]
    event_addr = zmq_info[1][1].split(' ')[-1]

    ctx = zmq.Context()
    heartbeat_sock = ctx.socket(zmq.SUB)
    heartbeat_sock.connect(heartbeat_addr)
    heartbeat_sock.setsockopt(zmq.SUBSCRIBE, '')
    event_sock = ctx.socket(zmq.SUB)
    event_sock.connect(event_addr)
    event_sock.setsockopt(zmq.SUBSCRIBE, '')

    # Wait for tango event
    for count in range(MAX_RETRIES):
        try:
            msg = heartbeat_sock.recv(flags=zmq.NOBLOCK)
            print('  heartbeat msg Rx: %r' % msg)
        except Exception:
            pass
        try:
            msg = event_sock.recv(flags=zmq.NOBLOCK)
            print('  event msg Rx: %r' % msg)
        except Exception:
            pass
        if count == 0:
            # Trigger an event
            event_device.command_inout("send_event", wait=True)
        event_device.read_attribute("state", wait=True)
        # if len(results) > 1:
        #    break
        time.sleep(TIME_PER_RETRY)
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
    for _ in range(MAX_RETRIES):
        event_device.read_attribute("state", wait=True)
        if len(results) > 1:
            break
        time.sleep(TIME_PER_RETRY)
    event_device.command_inout("delete_dyn_attr", 'bla', wait=True)
    # Wait for tango event
    for _ in range(MAX_RETRIES):
        event_device.read_attribute("state", wait=True)
        if len(results) > 2:
            break
        time.sleep(TIME_PER_RETRY)
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
    for _ in range(MAX_RETRIES):
        event_device.read_attribute("state", wait=True)
        if len(ec.get_events()) > 1:
            break
        time.sleep(TIME_PER_RETRY)
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
