from __future__ import print_function

from tango import DevState
from tango.server import Device, device_property


class MandatoryPropertyServer(Device):

    Hostname = device_property(
        dtype='str', mandatory=True,
        doc='The controller host address')

    Port = device_property(
        dtype='int', default_value=3456,
        doc='The controller port number')

    def init_device(self):
        Device.init_device(self)
        print('Port: ', self.Port)
        print('Host: ', self.Hostname)
        self.set_state(DevState.ON)


if __name__ == '__main__':
    MandatoryPropertyServer.run_server()
