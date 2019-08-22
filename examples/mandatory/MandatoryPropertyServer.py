from __future__ import print_function

from tango import DevState
from tango.server import Device


class MandatoryPropertyServer(Device):


    def init_device(self):
        Device.init_device(self)
        self.set_state(DevState.ON)


if __name__ == '__main__':
    MandatoryPropertyServer.run_server()
