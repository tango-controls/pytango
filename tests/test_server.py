# -*- coding: utf-8 -*-

import sys
import textwrap
import pytest

from tango import DevState, AttrWriteType, GreenMode
from tango.server import Device
from tango.server import command, attribute, device_property
from tango.test_utils import DeviceTestContext

# Asyncio imports
try:
    import asyncio
except ImportError:
    import trollius as asyncio

# Pytest fixtures
from tango.test_utils import state, typed_values, server_green_mode
state, typed_values, server_green_mode

# Constants
PY3 = sys.version_info >= (3,)
YIELD_FROM = "yield from" if PY3 else "yield asyncio.From"
RETURN = "return" if PY3 else "raise asyncio.Return"


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


# Test inheritance

def test_inheritance(server_green_mode):

    class A(Device):
        green_mode = server_green_mode

        prop1 = device_property(dtype=str, default_value="hello1")
        prop2 = device_property(dtype=str, default_value="hello2")

        @command(dtype_out=str)
        def get_prop1(self):
            return self.prop1

        @command(dtype_out=str)
        def get_prop2(self):
            return self.prop2

        @attribute(access=AttrWriteType.READ_WRITE)
        def attr(self):
            return self.attr_value

        @attr.write
        def attr(self, value):
            self.attr_value = value

        def dev_status(self):
            return ")`'-.,_"

    class B(A):

        prop2 = device_property(dtype=str, default_value="goodbye2")

        @attribute
        def attr2(self):
            return 3.14

        def dev_status(self):
            return 3 * A.dev_status(self)

        if server_green_mode == GreenMode.Asyncio:
            code = textwrap.dedent("""\
                @asyncio.coroutine
                def dev_status(self):
                    coro = super(type(self), self).dev_status()
                    result = {YIELD_FROM}(coro)
                    {RETURN}(3*result)
            """).format(**globals())
            exec(code)

    with DeviceTestContext(B) as proxy:
        assert proxy.get_prop1() == "hello1"
        assert proxy.get_prop2() == "goodbye2"
        proxy.attr = 1.23
        assert proxy.attr == 1.23
        assert proxy.attr2 == 3.14
        assert proxy.status() == ")`'-.,_)`'-.,_)`'-.,_"
