import numpy
import logging
from numpy import void

import gevent

import tango
from tango.server import run
from tango.server import Device
from tango.server import attribute, command
from tango.server import class_property, device_property
from tango import AttrQuality, AttrWriteType, DispLevel, DevState

__all__ = ["FwdServer", "main"]

class FwdServer(Device):
    """
    Start this server: python FwdServer.py myFwdServer
    Start the server containing the root attribute.
    Then using jive select the Attribute properties from myFwdServer
    Select you forwarded attribute and add the value to __root_att
    e.g. __root_att   ->  x/y/z/root_attribute_name
    Now restart the FwdServer
    """

    def __init__(self, *args, **kwargs):
        Device.__init__(self, *args, **kwargs)

    def init_device(self):
        Device.init_device(self)
        self._current = 0.0
        self.set_state(tango.DevState.ON)

    def always_executed_hook(self):
        pass

    def delete_device(self):
        pass

    voltage = attribute(name="voltage", label='Voltage', forwarded=True)

    @attribute(label='Current', dtype='float')
    def current(self):
        return self._current

# ----------
# Run server
# ----------

def main():
    from tango import GreenMode
    from tango.server import run
    run([FwdServer,], green_mode=GreenMode.Gevent)

if __name__ == '__main__':
    main()
