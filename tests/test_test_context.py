# -*- coding: utf-8 -*-

import pytest

import tango

from tango.server import Device
from tango.server import command, attribute, device_property
from tango.test_utils import (
    DeviceTestContext,
    MultiDeviceTestContext,
    SimpleDevice,
    ClassicAPISimpleDeviceImpl,
    ClassicAPISimpleDeviceClass
)


class Device1(Device):
    @attribute
    def attr1(self):
        return 100


class Device2(Device):
    @attribute
    def attr2(self):
        return 200


def test_single_device(server_green_mode):
    class TestDevice(Device1):
        green_mode = server_green_mode

    with DeviceTestContext(TestDevice) as proxy:
        assert proxy.attr1 == 100


def test_single_device_old_api():
    with DeviceTestContext(ClassicAPISimpleDeviceImpl, ClassicAPISimpleDeviceClass) as proxy:
        assert proxy.attr1 == 100


@pytest.mark.parametrize(
    "spec, device_cls, device",
    [
        (SimpleDevice, None, SimpleDevice),
        ("tango.test_utils.SimpleDevice", None, SimpleDevice),
        (
            ("tango.test_utils.ClassicAPISimpleDeviceClass", "tango.test_utils.ClassicAPISimpleDeviceImpl"),
            ClassicAPISimpleDeviceClass,
            ClassicAPISimpleDeviceImpl,
        ),
        (
            ("tango.test_utils.ClassicAPISimpleDeviceClass", ClassicAPISimpleDeviceImpl),
            ClassicAPISimpleDeviceClass,
            ClassicAPISimpleDeviceImpl
        ),
        (
            (ClassicAPISimpleDeviceClass, "tango.test_utils.ClassicAPISimpleDeviceImpl"),
            ClassicAPISimpleDeviceClass,
            ClassicAPISimpleDeviceImpl,
        ),
        (
            (ClassicAPISimpleDeviceClass, ClassicAPISimpleDeviceImpl),
            ClassicAPISimpleDeviceClass,
            ClassicAPISimpleDeviceImpl,
        ),
    ]
)
def test_multi_devices_info(spec, device_cls, device):
    devices_info = ({"class": spec, "devices": [{"name": "test/device1/1"}]},)

    dev_class = device if isinstance(device, str) else device.__name__

    with MultiDeviceTestContext(devices_info) as context:
        proxy1 = context.get_device("test/device1/1")
        assert proxy1.info().dev_class == dev_class


def test_multi_with_single_device(server_green_mode):
    class TestDevice(Device1):
        green_mode = server_green_mode

    devices_info = ({"class": TestDevice, "devices": [{"name": "test/device1/1"}]},)

    with MultiDeviceTestContext(devices_info) as context:
        proxy1 = context.get_device("test/device1/1")
        assert proxy1.attr1 == 100


def test_multi_with_single_device_old_api():
    devices_info = (
        {
            "class": (ClassicAPISimpleDeviceClass, ClassicAPISimpleDeviceImpl),
            "devices": [{"name": "test/device1/1"}],
        },
    )

    with MultiDeviceTestContext(devices_info) as context:
        proxy1 = context.get_device("test/device1/1")
        assert proxy1.attr1 == 100


def test_multi_with_two_devices(server_green_mode):
    class TestDevice1(Device1):
        green_mode = server_green_mode

    class TestDevice2(Device2):
        green_mode = server_green_mode

    devices_info = (
        {"class": TestDevice1, "devices": [{"name": "test/device1/1"}]},
        {"class": TestDevice2, "devices": [{"name": "test/device2/1"}]},
    )

    with MultiDeviceTestContext(devices_info) as context:
        proxy1 = context.get_device("test/device1/1")
        proxy2 = context.get_device("test/device2/1")
        assert proxy1.attr1 == 100
        assert proxy2.attr2 == 200


def test_multi_device_access():
    devices_info = (
        {"class": Device1, "devices": [{"name": "test/device1/1"}]},
        {"class": Device2, "devices": [{"name": "test/device2/2"}]},
    )

    with MultiDeviceTestContext(devices_info) as context:
        device_access1 = context.get_device_access("test/device1/1")
        device_access2 = context.get_device_access("test/device2/2")
        server_access = context.get_server_access()
        assert "test/device1/1" in device_access1
        assert "test/device2/2" in device_access2
        assert context.server_name in server_access
        proxy1 = tango.DeviceProxy(device_access1)
        proxy2 = tango.DeviceProxy(device_access2)
        proxy_server = tango.DeviceProxy(server_access)
        assert proxy1.attr1 == 100
        assert proxy2.attr2 == 200
        assert proxy_server.State() == tango.DevState.ON


def test_multi_device_proxy_cached():
    devices_info = ({"class": Device1, "devices": [{"name": "test/device1/1"}]},)

    with MultiDeviceTestContext(devices_info) as context:
        device1_first = context.get_device("test/device1/1")
        device1_second = context.get_device("test/device1/1")
        assert device1_first is device1_second


def test_multi_with_two_devices_with_properties(server_green_mode):
    class TestDevice1(Device):
        green_mode = server_green_mode

        prop1 = device_property(dtype=str)

        @command(dtype_out=str)
        def get_prop1(self):
            return self.prop1

    class TestDevice2(Device):
        green_mode = server_green_mode

        prop2 = device_property(dtype=int)

        @command(dtype_out=int)
        def get_prop2(self):
            return self.prop2

    devices_info = (
        {
            "class": TestDevice1,
            "devices": [{"name": "test/device1/1", "properties": {"prop1": "abcd"}}],
        },
        {
            "class": TestDevice2,
            "devices": [{"name": "test/device2/2", "properties": {"prop2": 5555}}],
        },
    )

    with MultiDeviceTestContext(devices_info) as context:
        proxy1 = context.get_device("test/device1/1")
        proxy2 = context.get_device("test/device2/2")
        assert proxy1.get_prop1() == "abcd"
        assert proxy2.get_prop2() == 5555


@pytest.fixture(
    # Per test we have the input config tuple, and then the expected exception type
    params=[
        # empty config
        [tuple(), IndexError],
        # missing/invalid keys
        [({"not-class": Device1, "devices": [{"name": "test/device1/1"}]},), KeyError],
        [({"class": Device1, "not-devices": [{"name": "test/device1/1"}]},), KeyError],
        [({"class": Device1, "devices": [{"not-name": "test/device1/1"}]},), KeyError],
        # duplicate class
        [
            (
                {"class": Device1, "devices": [{"name": "test/device1/1"}]},
                {"class": Device1, "devices": [{"name": "test/device1/2"}]},
            ),
            ValueError,
        ],
        # mixing old "classic" API and new high level API
        [
            (
                {"class": Device1, "devices": [{"name": "test/device1/1"}]},
                {
                    "class": (ClassicAPISimpleDeviceClass, ClassicAPISimpleDeviceImpl),
                    "devices": [{"name": "test/device1/2"}],
                },
            ),
            ValueError,
        ],
    ]
)
def bad_multi_device_config(request):
    return request.param


def test_multi_bad_config_fails(bad_multi_device_config):
    bad_config, expected_error = bad_multi_device_config
    with pytest.raises(expected_error):
        with MultiDeviceTestContext(bad_config):
            pass
