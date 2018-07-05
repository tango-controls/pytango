# -*- coding: utf-8 -*-

import sys
import textwrap
import pytest
import enum

from tango import DevState, AttrWriteType, GreenMode, DevFailed
from tango.server import Device
from tango.server import command, attribute, device_property
from tango.test_utils import DeviceTestContext, assert_close, \
    GoodEnum, BadEnumNonZero, BadEnumSkipValues, BadEnumDuplicates
from tango.utils import get_enum_labels, EnumTypeError


# Asyncio imports
try:
    import asyncio
except ImportError:
    import trollius as asyncio  # noqa: F401

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

    if dtype == (bool,):
        pytest.xfail('Not supported for some reasons')

    class TestDevice(Device):
        green_mode = server_green_mode

        @command(dtype_in=dtype, dtype_out=dtype)
        def identity(self, arg):
            return arg

    with DeviceTestContext(TestDevice) as proxy:
        for value in values:
            assert_close(proxy.identity(value), value)


def test_polled_command(server_green_mode):

    dct = {'Polling1': 100,
           'Polling2': 100000,
           'Polling3': 500}

    class TestDevice(Device):
        green_mode = server_green_mode

        @command(polling_period=dct["Polling1"])
        def Polling1(self):
            pass

        @command(polling_period=dct["Polling2"])
        def Polling2(self):
            pass

        @command(polling_period=dct["Polling3"])
        def Polling3(self):
            pass

    with DeviceTestContext(TestDevice) as proxy:
        ans = proxy.polling_status()

    for info in ans:
        lines = info.split('\n')
        comm = lines[0].split('= ')[1]
        period = int(lines[1].split('= ')[1])
        assert dct[comm] == period


# Test attributes

def test_read_write_attribute(typed_values, server_green_mode):
    dtype, values = typed_values

    class TestDevice(Device):
        green_mode = server_green_mode

        @attribute(dtype=dtype, max_dim_x=10,
                   access=AttrWriteType.READ_WRITE)
        def attr(self):
            return self.attr_value

        @attr.write
        def attr(self, value):
            self.attr_value = value

    with DeviceTestContext(TestDevice) as proxy:
        for value in values:
            proxy.attr = value
            assert_close(proxy.attr, value)


def test_read_write_attribute_enum(server_green_mode):
    values = (member.value for member in GoodEnum)
    enum_labels = get_enum_labels(GoodEnum)

    class TestDevice(Device):
        green_mode = server_green_mode

        def __init__(self, *args, **kwargs):
            super(TestDevice, self).__init__(*args, **kwargs)
            self.attr_from_enum_value = 0
            self.attr_from_labels_value = 0

        @attribute(dtype=GoodEnum, access=AttrWriteType.READ_WRITE)
        def attr_from_enum(self):
            return self.attr_from_enum_value

        @attr_from_enum.write
        def attr_from_enum(self, value):
            self.attr_from_enum_value = value

        @attribute(dtype='DevEnum', enum_labels=enum_labels,
                   access=AttrWriteType.READ_WRITE)
        def attr_from_labels(self):
            return self.attr_from_labels_value

        @attr_from_labels.write
        def attr_from_labels(self, value):
            self.attr_from_labels_value = value

    with DeviceTestContext(TestDevice) as proxy:
        for value, label in zip(values, enum_labels):
            proxy.attr_from_enum = value
            read_attr = proxy.attr_from_enum
            assert read_attr == value
            assert isinstance(read_attr, enum.IntEnum)
            assert read_attr.value == value
            assert read_attr.name == label
            proxy.attr_from_labels = value
            read_attr = proxy.attr_from_labels
            assert read_attr == value
            assert isinstance(read_attr, enum.IntEnum)
            assert read_attr.value == value
            assert read_attr.name == label

    with pytest.raises(TypeError) as context:
        class BadTestDevice(Device):
            green_mode = server_green_mode

            def __init__(self, *args, **kwargs):
                super(BadTestDevice, self).__init__(*args, **kwargs)
                self.attr_value = 0

            # enum_labels may not be specified if dtype is an enum.Enum
            @attribute(dtype=GoodEnum, enum_labels=enum_labels)
            def bad_attr(self):
                return self.attr_value

        BadTestDevice()  # dummy instance for Codacy
    assert 'enum_labels' in str(context.value)


# Test properties

def test_device_property_no_default(typed_values, server_green_mode):
    dtype, values = typed_values
    patched_dtype = dtype if dtype != (bool,) else (int,)
    default = values[0]
    value = values[1]

    class TestDevice(Device):
        green_mode = server_green_mode

        prop = device_property(dtype=dtype)

        @command(dtype_out=patched_dtype)
        def get_prop(self):
            return default if self.prop is None else self.prop

    with DeviceTestContext(TestDevice, process=True) as proxy:
        assert_close(proxy.get_prop(), default)

    with DeviceTestContext(TestDevice,
                           properties={'prop': value},
                           process=True) as proxy:
        assert_close(proxy.get_prop(), value)


def test_device_property_with_default_value(typed_values, server_green_mode):
    dtype, values = typed_values
    patched_dtype = dtype if dtype != (bool,) else (int,)
    default = values[0]
    value = values[1]

    class TestDevice(Device):
        green_mode = server_green_mode

        prop = device_property(dtype=dtype, default_value=default)

        @command(dtype_out=patched_dtype)
        def get_prop(self):
            print(self.prop)
            return self.prop

    with DeviceTestContext(TestDevice, process=True) as proxy:
        assert_close(proxy.get_prop(), default)

    with DeviceTestContext(TestDevice,
                           properties={'prop': value},
                           process=True) as proxy:
        assert_close(proxy.get_prop(), value)


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


def test_polled_attribute(server_green_mode):

    dct = {'PolledAttribute1': 100,
           'PolledAttribute2': 100000,
           'PolledAttribute3': 500}

    class TestDevice(Device):
        green_mode = server_green_mode

        @attribute(polling_period=dct["PolledAttribute1"])
        def PolledAttribute1(self):
            return 42.0

        @attribute(polling_period=dct["PolledAttribute2"])
        def PolledAttribute2(self):
            return 43.0

        @attribute(polling_period=dct["PolledAttribute3"])
        def PolledAttribute3(self):
            return 44.0

    with DeviceTestContext(TestDevice) as proxy:
        ans = proxy.polling_status()
        for x in ans:
            lines = x.split('\n')
            attr = lines[0].split('= ')[1]
            poll_period = int(lines[1].split('= ')[1])
            assert dct[attr] == poll_period


def test_mandatory_device_property(typed_values, server_green_mode):
    dtype, values = typed_values
    patched_dtype = dtype if dtype != (bool,) else (int,)
    default, value = values[:2]

    class TestDevice(Device):
        green_mode = server_green_mode

        prop = device_property(dtype=dtype, mandatory=True)

        @command(dtype_out=patched_dtype)
        def get_prop(self):
            return self.prop

    with DeviceTestContext(TestDevice,
                           properties={'prop': value},
                           process=True) as proxy:
        assert_close(proxy.get_prop(), value)

    with pytest.raises(DevFailed) as context:
        with DeviceTestContext(TestDevice, process=True) as proxy:
            pass
    assert 'Device property prop is mandatory' in str(context.value)


# fixtures

@pytest.fixture(params=[GoodEnum])
def good_enum(request):
    return request.param


@pytest.fixture(params=[BadEnumNonZero, BadEnumSkipValues, BadEnumDuplicates])
def bad_enum(request):
    return request.param


# test utilities for servers

def test_get_enum_labels_success(good_enum):
    expected_labels = ['START', 'MIDDLE', 'END']
    assert get_enum_labels(good_enum) == expected_labels


def test_get_enum_labels_fail(bad_enum):
    with pytest.raises(EnumTypeError):
        get_enum_labels(bad_enum)
