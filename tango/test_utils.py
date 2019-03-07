"""Test utilities"""

import sys
import six
import enum
try:
    import collections.abc as collections_abc  # python 3.3+
except ImportError:
    import collections as collections_abc

# Local imports
from . import DevState, GreenMode
from .server import Device
from .test_context import DeviceTestContext

# Conditional imports
try:
    import pytest
except ImportError:
    pytest = None

try:
    import numpy.testing
except ImportError:
    numpy = None

__all__ = ('DeviceTestContext', 'SimpleDevice')

PY3 = sys.version_info >= (3,)

# char \x00 cannot be sent in a DevString. All other 1-255 chars can
ints = tuple(range(1, 256))
bytes_devstring = bytes(ints) if PY3 else ''.join(map(chr, ints))
str_devstring = bytes_devstring.decode('latin-1')

# Test devices

class SimpleDevice(Device):
    def init_device(self):
        self.set_state(DevState.ON)


# Test enums

class GoodEnum(enum.IntEnum):
    START = 0
    MIDDLE = 1
    END = 2


class BadEnumNonZero(enum.IntEnum):
    START = 1
    MIDDLE = 2
    END = 3


class BadEnumSkipValues(enum.IntEnum):
    START = 0
    MIDDLE = 2
    END = 4


class BadEnumDuplicates(enum.IntEnum):
    START = 0
    MIDDLE = 1
    END = 1


# Helpers

TYPED_VALUES = {
    int: (1, 2),
    float: (2.71, 3.14),
    str: ('hey hey', 'my my', bytes_devstring, str_devstring),
    bool: (False, True),
    (int,): ([1, 2, 3], [9, 8, 7]),
    (float,): ([0.1, 0.2, 0.3], [0.9, 0.8, 0.7]),
    (str,): (['ab', 'cd', 'ef'], ['gh', 'ij', 'kl'], [bytes_devstring], [str_devstring]),
    (bool,): ([False, False, True], [True, False, False])}


def repr_type(x):
    if not isinstance(x, tuple):
        return x.__name__
    return '({},)'.format(x[0].__name__)


# Numpy helpers

if numpy and pytest:

    def assert_close(a, b):
        if isinstance(a, six.string_types):
            assert a == b
            return
        if isinstance(a, collections_abc.Sequence) and len(a) and isinstance(a[0], six.string_types):
            assert list(a) == list(b)
            return
        try:
            assert a == pytest.approx(b)
        except ValueError:
            numpy.testing.assert_allclose(a, b)

# Pytest fixtures

if pytest:

    def create_result(dtype, value):
        if dtype == str:
            if PY3:
                if isinstance(value, bytes):
                    return value.decode('latin-1')
            else:
                if isinstance(value, unicode):
                    return value.encode('latin-1')
        elif dtype == (str,):
            return [create_result(str, v) for v in value]
        return value

    @pytest.fixture(params=DevState.values.values())
    def state(request):
        return request.param

    @pytest.fixture(
        params=list(TYPED_VALUES.items()),
        ids=lambda x: repr_type(x[0]))
    def typed_values(request):
        dtype, values = request.param
        expected = lambda v: create_result(dtype, v)
        return dtype, values, expected

    @pytest.fixture(params=GreenMode.values.values())
    def green_mode(request):
        return request.param

    @pytest.fixture(params=[
        GreenMode.Synchronous,
        GreenMode.Asyncio,
        GreenMode.Gevent])
    def server_green_mode(request):
        return request.param
