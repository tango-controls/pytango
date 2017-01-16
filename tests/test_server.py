# -*- coding: utf-8 -*-

import pytest

from tango import DevState, AttrWriteType
from tango.server import Device, DeviceMeta
from tango.server import command, attribute, device_property
from tango.test_utils import DeviceTestContext

# Pytest fixtures
from tango.test_utils import state, typed_values, server_green_mode
state, typed_values, server_green_mode


# Test state/status

def test_empty_device(server_green_mode):

    class TestDevice(Device):
        green_mode = server_green_mode

    with DeviceTestContext(TestDevice) as proxy:
        assert proxy.state() == DevState.UNKNOWN
        assert proxy.status() == 'The device is in UNKNOWN state.'


def test_set_state(state, server_green_mode):
    status = 'The device is in {0!s} state.'.format(state)

    class TestDevice(Device):
        green_mode = server_green_mode

        def init_device(self):
            self.set_state(state)

    with DeviceTestContext(TestDevice) as proxy:
        assert proxy.state() == state
        assert proxy.status() == status


def test_set_status(server_green_mode):

    status = '\n'.join((
        "This is a multiline status",
        "with special characters such as",
        "Café à la crème"))

    class TestDevice(Device):
        green_mode = server_green_mode

        def init_device(self):
            self.set_state(DevState.ON)
            self.set_status(status)

    with DeviceTestContext(TestDevice) as proxy:
        assert proxy.state() == DevState.ON
        assert proxy.status() == status


# Test commands

def test_identity_command(typed_values, server_green_mode):
    dtype, values = typed_values

    class TestDevice(Device):
        green_mode = server_green_mode

        @command(dtype_in=dtype, dtype_out=dtype)
        def identity(self, arg):
            return arg

    with DeviceTestContext(TestDevice) as proxy:
        for value in values:
            expected = pytest.approx(value)
            assert proxy.identity(value) == expected


# Test attributes

def test_read_write_attribute(typed_values, server_green_mode):
    dtype, values = typed_values

    class TestDevice(Device):
        green_mode = server_green_mode

        @attribute(dtype=dtype, access=AttrWriteType.READ_WRITE)
        def attr(self):
            return self.attr_value

        @attr.write
        def attr(self, value):
            self.attr_value = value

    with DeviceTestContext(TestDevice) as proxy:
        for value in values:
            proxy.attr = value
            expected = pytest.approx(value)
            assert proxy.attr == expected


# Test properties

def test_device_property(typed_values, server_green_mode):
    dtype, values = typed_values
    default = values[0]
    value = values[1]

    class TestDevice(Device):
        green_mode = server_green_mode

        prop = device_property(dtype=dtype, default_value=default)

        @command(dtype_out=dtype)
        def get_prop(self):
            return self.prop

    with DeviceTestContext(TestDevice, process=True) as proxy:
        expected = pytest.approx(default)
        assert proxy.get_prop() == expected

    with DeviceTestContext(TestDevice,
                           properties={'prop': value},
                           process=True) as proxy:
        expected = pytest.approx(value)
        assert proxy.get_prop() == expected
