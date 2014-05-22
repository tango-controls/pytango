import time
from PyTango.server import run
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute, command   


class Clock(Device):
    __metaclass__ = DeviceMeta

    @attribute
    def time(self):
	"""The time attribute"""
        return time.time()

    @command(dtype_in=str, dtype_out=str)
    def strftime(self, format):
        return time.strftime(format)


if __name__ == "__main__":
    run([Clock])
