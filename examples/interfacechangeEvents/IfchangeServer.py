
# PyTango imports
import PyTango
from PyTango import DebugIt
from PyTango.server import run
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute, command
from PyTango.server import class_property, device_property
from PyTango import AttrQuality, AttrWriteType, DispLevel, DevState

import numpy
import gevent
import logging
from dbus.server import Server

class IfchangeServer(Device):

    __metaclass__ = DeviceMeta

    # ---------------
    # General methods
    # ---------------

    def __init__(self, *args, **kwargs):
        Device.__init__(self, *args, **kwargs)

    def init_device(self):
        Device.init_device(self)
        logging.basicConfig(level=logging.DEBUG)
        self.set_state(PyTango.DevState.ON)

    @attribute(label='Sequence Counter', dtype='int',
        description="Sequence counter")
    def seq_counter(self):
        return 456

    @attribute(label='Voltage', dtype='float',
        description="voltage")
    def volts(self):
        return 3.142

    def read_current():
        return self._current

    def write_current(self, curr):
        self._current = curr
    
    @command(dtype_in=str, doc_in='name of dynamic attribute to add')
    def add_dyn_attr(self, name):
        attr = attribute(name=name, dtype='float', 
                          fget=self.read_current, fset=self.write_current)
        self.add_attribute(attr)
        
    @command(dtype_in=str, doc_in='name of dynamic attribute to delete')
    def delete_dyn_attr(self, name):
        self._remove_attribute(name)

# ----------
# Run server
# ----------

def main():
    from PyTango import GreenMode
    from PyTango.server import run

    run([IfchangeServer,], green_mode=GreenMode.Gevent)

if __name__ == '__main__':
    main()
