

# Imports
from concurrent.futures import Future

import pytest
from numpy.testing import assert_array_equal

import tango
from tango.server import Device, command
from tango.test_utils import DeviceTestContext


def test_async_command_polled(typed_values):
    dtype, values = typed_values

    if dtype == (bool,):
        pytest.xfail('Not supported for some reasons')

    class TestDevice(Device):

        @command(dtype_in=dtype, dtype_out=dtype)
        def identity(self, arg):
            return arg

    with DeviceTestContext(TestDevice) as proxy:
        for value in values:
            eid = proxy.command_inout_asynch('identity', value)
            result = proxy.command_inout_reply(eid, timeout=500)
            assert_array_equal(result, value)


def test_async_command_with_polled_callback(typed_values):
    dtype, values = typed_values

    if dtype == (bool,):
        pytest.xfail('Not supported for some reasons')

    class TestDevice(Device):

        @command(dtype_in=dtype, dtype_out=dtype)
        def identity(self, arg):
            return arg

    api_util = tango.ApiUtil.instance()
    api_util.set_asynch_cb_sub_model(tango.cb_sub_model.PULL_CALLBACK)

    with DeviceTestContext(TestDevice) as proxy:
        for value in values:
            future = Future()
            proxy.command_inout_asynch('identity', value, future.set_result)
            api_util.get_asynch_replies(500)
            result = future.result()
            assert_array_equal(result.argout, value)


def test_async_command_with_pushed_callback(typed_values):
    dtype, values = typed_values

    if dtype == (bool,):
        pytest.xfail('Not supported for some reasons')

    class TestDevice(Device):

        @command(dtype_in=dtype, dtype_out=dtype)
        def identity(self, arg):
            return arg

    api_util = tango.ApiUtil.instance()
    api_util.set_asynch_cb_sub_model(tango.cb_sub_model.PUSH_CALLBACK)

    with DeviceTestContext(TestDevice) as proxy:
        for value in values:
            future = Future()
            proxy.command_inout_asynch('identity', value, future.set_result)
            result = future.result(timeout=0.5)
            assert_array_equal(result.argout, value)
