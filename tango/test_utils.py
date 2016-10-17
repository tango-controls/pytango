"""Test utilities"""

__all__ = ['DeviceTestContext', 'SimpleDevice']

# Imports
from six import add_metaclass

# Local imports
from . import utils
from . import DevState, CmdArgType, GreenMode
from .server import Device, DeviceMeta
from .test_context import DeviceTestContext

# Conditional imports
try:
    import pytest
except ImportError:
    pytest = None


# Test devices

@add_metaclass(DeviceMeta)
class SimpleDevice(Device):
    def init_device(self):
        self.set_state(DevState.ON)


# Pytest fixtures

if pytest:

    @pytest.fixture(params=DevState.values.values())
    def state(request):
        return request.param

    @pytest.fixture(params=utils._scalar_types)
    def typed_values(request):
        dtype = request.param
        # Unsupported types
        if dtype in [CmdArgType.DevInt, CmdArgType.ConstDevString,
                     CmdArgType.DevEncoded, CmdArgType.DevUChar]:
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

    @pytest.fixture(params=GreenMode.values.values())
    def green_mode(request):
        return request.param

    @pytest.fixture(params=[
        GreenMode.Synchronous,
        GreenMode.Asyncio,
        GreenMode.Gevent])
    def server_green_mode(request):
        return request.param
