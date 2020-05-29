import pytest

from tango import DevFailed, DevSource, DevState

from ska.base.control_model import AdminMode


class TestMccsIntegration:
    """
    Integration test cases for the Mccs device classes
    """

    def test_master_can_enable_subarray(self, tango_context):
        """
        Test that a MccsMaster device can enable an MccsSubarray device.

        Uses the `tango_context` pytest fixture.
        """
        master = tango_context.get_device("low/elt/master")
        subarray = tango_context.get_device("low/elt/subarray_1")

        # check subarray is disabled
        assert subarray_1.adminMode == AdminMode.OFFLINE
        assert subarray_1.State() == DevState.DISABLE

        # enable subarray
        master.EnableSubarray(1)

        # check subarray 1 is enabled.
        assert subarray_1.adminMode == AdminMode.ONLINE
        assert subarray_1.State() == DevState.OFF

        # try to enable subarray 1 again -- this should fail
        with pytest.raises(DevFailed):
            master.EnableSubarray(1)

        # check failure has no side-effect
        assert subarray_1.adminMode == AdminMode.ONLINE
        assert subarray_1.State() == DevState.OFF
