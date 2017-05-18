import numpy
import gevent
import logging
from numpy import void

# PyTango imports
import PyTango
from PyTango import DebugIt
from PyTango.server import run
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute, command
from PyTango.server import class_property, device_property
from PyTango import AttrQuality, AttrWriteType, DispLevel, DevState

class IfchangeServer(Device):

    # ---------------
    # General methods
    # ---------------

    def __init__(self, *args, **kwargs):
        Device.__init__(self, *args, **kwargs)

    def init_device(self):
        Device.init_device(self)
        logging.basicConfig(level=logging.DEBUG)
        self._current = 0.0
        self.set_state(PyTango.DevState.ON)

    @attribute(label='Sequence Counter', dtype='int',
        description="Sequence counter")
    def seq_counter(self):
        return 456

    @attribute(label='Voltage', dtype='float',
        description="voltage")
    def volts(self):
        return 3.142

    def read_current(self):
        print "read current method ", self._current
        return self._current

    def write_current(self, curr):
        self._current = curr
        print "set current to ", self._current
   
    def start(self, argin):
        print "start method"
        return 3142

    @command(dtype_in=str, doc_in='name of dynamic attribute to add')
    def add_dyn_attr(self, name):
        attr = attribute(name=name, dtype='float', 
                          fget=self.read_current, fset=self.write_current)
        self.add_attribute(attr)

    @command(dtype_in=str, doc_in='name of dynamic attribute to delete')
    def delete_dyn_attr(self, name):
        self._remove_attribute(name)

    @command
    def add_dyn_cmd(self):
        device_level = True
        cmd = command(f=self.start, dtype_in=str, doc_in='description of input', dtype_out=int, doc_out='description of output',
                      display_level=DispLevel.EXPERT, polling_period=5.1)
        self.add_command(cmd, device_level)

    @command(dtype_in=str, doc_in='name of dynamic command to delete')
    def delete_dyn_cmd(self, name):
        self._remove_command(name, False, True)
        pass

# ----------
# Run server
# ----------

def main():
    from PyTango import GreenMode
    from PyTango.server import run

    run([IfchangeServer,], green_mode=GreenMode.Gevent)

if __name__ == '__main__':
    main()
