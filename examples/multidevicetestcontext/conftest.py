"""
A module defining pytest fixtures for testing with MultiDeviceTestContext
Requires pytest, and at least PyTango 9.3.3
(see commit history for the approach to use with PyTango 9.3.2)
"""

from collections import defaultdict
import pytest
from tango.test_context import MultiDeviceTestContext


@pytest.fixture(scope="module")
def devices_info(request):
    yield getattr(request.module, "devices_info")

    
@pytest.fixture(scope="function")
def tango_context(devices_info):
    """
    Creates and returns a TANGO MultiDeviceTestContext object.
    """
    with MultiDeviceTestContext(devices_info, process=True) as context:
        yield context
