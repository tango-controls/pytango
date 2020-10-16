# -*- coding: utf-8 -*-

import pytest

import tango

from tango.server import AttrWriteType, Device
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
    "class_field, device",
    [
        (SimpleDevice, SimpleDevice),
        ("tango.test_utils.SimpleDevice", SimpleDevice),
        (
            ("tango.test_utils.ClassicAPISimpleDeviceClass", "tango.test_utils.ClassicAPISimpleDeviceImpl"),
            ClassicAPISimpleDeviceImpl
        ),
        (
            ("tango.test_utils.ClassicAPISimpleDeviceClass", ClassicAPISimpleDeviceImpl),
            ClassicAPISimpleDeviceImpl
        ),
        (
            (ClassicAPISimpleDeviceClass, "tango.test_utils.ClassicAPISimpleDeviceImpl"),
            ClassicAPISimpleDeviceImpl
        ),
        (
            (ClassicAPISimpleDeviceClass, ClassicAPISimpleDeviceImpl),
            ClassicAPISimpleDeviceImpl
        ),
    ]
)
def test_multi_devices_info(class_field, device):
    devices_info = ({"class": class_field, "devices": [{"name": "test/device1/1"}]},)

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


@pytest.fixture()
def memorized_attribute_test_device_factory():
    """
    Returns a test device factory that provides a test device with an
    attribute that is memorized or not, according to its boolean
    argument
    """
    def _factory(is_attribute_memorized):
        class _Device(Device):
            def init_device(self):
                self._attr_value = 0

            attr = attribute(
                access=AttrWriteType.READ_WRITE,
                memorized=is_attribute_memorized,
                hw_memorized=is_attribute_memorized
            )

            def read_attr(self):
                return self._attr_value

            def write_attr(self, value):
                self._attr_value = value

        return _Device
    return _factory


@pytest.mark.parametrize(
    "is_attribute_memorized, memorized_value, expected_value",
    [
        (False, None, 0),
        (False, "1", 0),
        (True, None, 0),
        (True, "1", 1),
    ]
)
def test_multi_with_memorized_attribute_values(
    memorized_attribute_test_device_factory,
    is_attribute_memorized,
    memorized_value,
    expected_value
):
    TestDevice = memorized_attribute_test_device_factory(is_attribute_memorized)

    device_info = {"name": "test/device1/1"}
    if memorized_value is not None:
        device_info["memorized"] = {"attr": memorized_value}

    devices_info = (
        {
            "class": TestDevice,
            "devices": [device_info]
        },
    )

    with MultiDeviceTestContext(devices_info) as context:
        proxy = context.get_device("test/device1/1")
        assert proxy.attr == expected_value


@pytest.mark.parametrize(
    "is_attribute_memorized, memorized_value, expected_value",
    [
        (False, None, 0),
        (False, 1, 0),
        (True, None, 0),
        (True, 1, 1),
    ]
)
def test_single_with_memorized_attribute_values(
    memorized_attribute_test_device_factory,
    is_attribute_memorized,
    memorized_value,
    expected_value
):
    TestDevice = memorized_attribute_test_device_factory(is_attribute_memorized)

    kwargs = {
        "memorized": {"attr": memorized_value}
    } if memorized_value is not None else {}

    with DeviceTestContext(TestDevice, **kwargs) as proxy:
        assert proxy.attr == expected_value
