"""
A module defining pytest fixtures for testing ska.mccs.

Requires pytest and pytest-mock
"""
from collections import defaultdict
import pytest
import socket
import tango
from tango.test_context import MultiDeviceTestContext, get_host_ip
from ska.mccs import MccsMaster, MccsSubarray

devices_info = [
    {
        "class": MccsMaster,
        "devices": (
            {
                "name": "low/elt/master",
                "properties": {
                    "MccsSubarrays": ["low/elt/subarray_1"],
                }
            },
        )
    },
    {
        "class": MccsSubarray,
        "devices": [
            {
                "name": "low/elt/subarray_1",
                "properties": {
                }
            },
        ]
    },
]

@pytest.fixture(scope="function")
def tango_context(mocker):
    """
    Creates and returns a TANGO MultiDeviceTestContext object, with
    tango.DeviceProxy patched to work around a name-resolving issue.
    """
    def _get_open_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        return port

    HOST = get_host_ip()
    PORT = _get_open_port()

    _DeviceProxy = tango.DeviceProxy
    mocker.patch(
        'tango.DeviceProxy',
        wraps=lambda fqdn, *args, **kwargs: _DeviceProxy(
            "tango://{0}:{1}/{2}#dbase=no".format(HOST, PORT, fqdn),
            *args,
            **kwargs
        )
    )

    with MultiDeviceTestContext(devices_info, host=HOST, port=PORT) as context:
        yield context
