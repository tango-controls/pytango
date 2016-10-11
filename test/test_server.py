# -*- coding: utf-8 -*-

import pytest
from six import add_metaclass

import tango
from tango import DevState, AttrWriteType, utils
from tango.server import Device, DeviceMeta
from tango.server import command, attribute, device_property

from context import TangoTestContext


# Fixtures

@pytest.fixture(params=DevState.values.values())
def state(request):
    return request.param


@pytest.fixture(params=utils._scalar_types)
def type_value(request):
    dtype = request.param
    # Unsupported types
    if dtype in [tango.DevInt, tango.ConstDevString,
                 tango.DevEncoded, tango.DevUChar]:
        pytest.xfail('Should we support those types?')
    # Supported types
    if dtype in utils._scalar_str_types:
        return dtype, ['hey hey', 'my my']
    if dtype in utils._scalar_bool_types:
        return dtype, [False, True]
    if dtype in utils._scalar_int_types:
        return dtype, [1, 2]
    if dtype in utils._scalar_float_types:
        return dtype, [2.71, 3.14]


# Test state/status

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


# Test commands

def test_identity_command(type_value):
    dtype, values = type_value

    @add_metaclass(DeviceMeta)
    class TestDevice(Device):

        @command(dtype_in=dtype, dtype_out=dtype)
        def identity(self, arg):
            return arg

    with TangoTestContext(TestDevice) as proxy:
        for value in values:
            expected = pytest.approx(value)
            assert proxy.identity(value) == expected


# Test attributes

def test_read_write_attribute(type_value):
    dtype, values = type_value

    @add_metaclass(DeviceMeta)
    class TestDevice(Device):

        @attribute(dtype=dtype, access=AttrWriteType.READ_WRITE)
        def attr(self):
            return self.attr_value

        @attr.write
        def attr(self, value):
            self.attr_value = value

    with TangoTestContext(TestDevice) as proxy:
        for value in values:
            proxy.attr = value
            expected = pytest.approx(value)
            assert proxy.attr == expected


# Test properties

@pytest.fixture
def device_with_property(type_value):
    dtype, values = type_value
    default = values[0]
    other = values[1]

    @add_metaclass(DeviceMeta)
    class TestDevice(Device):

        prop = device_property(dtype=dtype, default_value=default)

        @command(dtype_out=dtype)
        def get_prop(self):
            return self.prop

    return TestDevice, default, other


def test_default_property(device_with_property):
    TestDevice, default, _ = device_with_property
    with TangoTestContext(TestDevice) as proxy:
        expected = pytest.approx(default)
        assert proxy.get_prop() == expected


def test_device_property(device_with_property):
    TestDevice, _, value = device_with_property
    properties = {'prop': value}
    with TangoTestContext(TestDevice, properties=properties) as proxy:
        expected = pytest.approx(value)
        assert proxy.get_prop() == expected
