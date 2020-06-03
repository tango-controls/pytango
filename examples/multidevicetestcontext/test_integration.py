import pytest
import tango
from tango.server import Device, attribute, command, device_property


class Master(Device):
    WorkerFQDNs = device_property(dtype="DevVarStringArray")

    @command(dtype_in="DevLong")
    def Turn_Worker_On(self, worker_id):
        worker_fqdn = self.WorkerFQDNs[worker_id - 1]
        worker_device = tango.DeviceProxy(worker_fqdn)
        worker_device.is_on = True

    @command(dtype_in="DevLong")
    def Turn_Worker_Off(self, worker_id):
        worker_fqdn = self.WorkerFQDNs[worker_id - 1]
        worker_device = tango.DeviceProxy(worker_fqdn)
        worker_device.is_on = False


class Worker(Device):
    def init_device(self):
        super(Worker, self).init_device()
        self._is_on = False

    is_on = attribute(
        dtype=tango.DevBoolean,
        access=tango.AttrWriteType.READ_WRITE,
    )

    def read_is_on(self):
        return self._is_on

    def write_is_on(self, value):
        self._is_on = value


devices_info = [
    {
        "class": Master,
        "devices": (
            {
                "name": "device/master/1",
                "properties": {
                    "WorkerFQDNs": [
                        "device/worker/1",
                        "device/worker/2"
                    ],
                }
            },
        )
    },
    {
        "class": Worker,
        "devices": [
            {
                "name": "device/worker/1",
                "properties": {
                }
            },
            {
                "name": "device/worker/2",
                "properties": {
                }
            },
        ]
    },
]


class TestMasterWorkerIntegration:
    def test_master_turn_worker_on(self, tango_context):
        master = tango_context.get_device("device/master/1")
        worker_1 = tango_context.get_device("device/worker/1")
        worker_2 = tango_context.get_device("device/worker/2")

        # check initial state: both workers are off
        assert worker_1.is_on == False
        assert worker_2.is_on == False

        # tell master to enable worker_1
        master.turn_worker_on(1)

        # check worker_1 is now on, and worker_2 is still off
        assert worker_1.is_on == True
        assert worker_2.is_on == False

    def test_master_turn_worker_off(self, tango_context):
        master = tango_context.get_device("device/master/1")
        worker_1 = tango_context.get_device("device/worker/1")
        worker_2 = tango_context.get_device("device/worker/2")

        # tell master to enable both workers
        master.turn_worker_on(1)
        master.turn_worker_on(2)

        # check initial state: both workers are on
        assert worker_1.is_on == True
        assert worker_2.is_on == True

        # tell master to disable worker_1
        master.turn_worker_off(1)

        # check worker_1 is now off, and worker_2 is still on
        assert worker_1.is_on == False
        assert worker_2.is_on == True
