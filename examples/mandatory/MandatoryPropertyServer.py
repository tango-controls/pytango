import tango
from tango.server import Device, device_property
from tango.server import attribute

__all__ = ["MandatoryPropertyServer", "main"]


class MandatoryPropertyServer(Device):
    """
    """

    # -----------------
    # Device Properties
    # -----------------

    HostName = device_property(
        dtype='str', doc='The controller host address', mandatory=True
    )
    Port = device_property(
        dtype='int', doc='The port number', default_value=3456
    )

    def init_device(self):
        Device.init_device(self)
        print 'port: ', self.Port
        print 'host: ', self.HostName
        self.set_state(tango.DevState.ON)

# ----------
# Run server
# ----------

if __name__ == '__main__':
    MandatoryPropertyServer.run_server()
