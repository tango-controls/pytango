# -*- coding: utf-8 -*-

from six import add_metaclass

from tango import DevState
from tango.server import Device, DeviceMeta

from context import TangoTestContext


def test_empty_device():

    @add_metaclass(DeviceMeta)
    class TestDevice(Device):
        pass

    with TangoTestContext(TestDevice) as proxy:
        assert proxy.state() == DevState.UNKNOWN
        assert proxy.status() == 'The device is in UNKNOWN state.'


def test_set_state():

    @add_metaclass(DeviceMeta)
    class TestDevice(Device):
        def init_device(self):
            self.set_state(DevState.ON)

    with TangoTestContext(TestDevice) as proxy:
        assert proxy.state() == DevState.ON
        assert proxy.status() == 'The device is in ON state.'


def test_set_status():

    status = '\n'.join((
        "This is a multiline status",
        "with special characters such as",
        "Café à la crème"))

    @add_metaclass(DeviceMeta)
    class TestDevice(Device):
        def init_device(self):
            self.set_state(DevState.ON)
            self.set_status(status)

    with TangoTestContext(TestDevice) as proxy:
        assert proxy.state() == DevState.ON
        assert proxy.status() == status
