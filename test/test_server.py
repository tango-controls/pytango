# -*- coding: utf-8 -*-

import pytest
from six import add_metaclass

from tango import DevState
from tango.server import Device, DeviceMeta

from context import TangoTestContext


@pytest.fixture(params=DevState.names.values())
def state(request):
    return request.param


def test_empty_device():

    @add_metaclass(DeviceMeta)
    class TestDevice(Device):
        pass

    with TangoTestContext(TestDevice) as proxy:
        assert proxy.state() == DevState.UNKNOWN
        assert proxy.status() == 'The device is in UNKNOWN state.'


def test_set_state(state):
    status = 'The device is in {0!s} state.'.format(state)

    @add_metaclass(DeviceMeta)
    class TestDevice(Device):
        def init_device(self):
            self.set_state(state)

    with TangoTestContext(TestDevice) as proxy:
        assert proxy.state() == state
        assert proxy.status() == status


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
