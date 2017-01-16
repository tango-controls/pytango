import time
from PyTango.server import Device, attribute, command


class Clock(Device):

    @attribute
    def time(self):
        return time.time()

    @command(dtype_in=str, dtype_out=str)
    def strftime(self, format):
        return time.strftime(format)


if __name__ == "__main__":
    Clock.run_server()
