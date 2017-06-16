import numpy
import gevent
import logging

import tango
from tango import DevState
from tango.server import run
from tango.server import Device
from tango.server import attribute, command


class IfchangeServer(Device):

    def init_device(self):
        Device.init_device(self)
        logging.basicConfig(level=logging.DEBUG)
        self.set_state(tango.DevState.ON)

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

if __name__ == '__main__':
    IfchangeServer.run_server()
