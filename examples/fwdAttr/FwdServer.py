import tango
from tango.server import Device
from tango.server import attribute

__all__ = ("FwdServer", "main")


class FwdServer(Device):
    """
    Start this server: python FwdServer.py myFwdServer
    Start the server containing the root attribute.
    Then using jive select the Attribute properties from myFwdServer
    Select you forwarded attribute and add the value to __root_att
    e.g. __root_att   ->  x/y/z/root_attribute_name
    Now restart the FwdServer
    """

    def init_device(self):
        Device.init_device(self)
        self._current = 0.0
        self.set_state(tango.DevState.ON)

    voltage = attribute(name="voltage", label='Voltage', forwarded=True)

    @attribute(label='Current', dtype='float')
    def current(self):
        return self._current

# ----------
# Run server
# ----------


if __name__ == '__main__':
    FwdServer.run_server()
