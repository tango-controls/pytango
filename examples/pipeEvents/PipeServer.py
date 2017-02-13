# PyTango imports

import PyTango
from PyTango import DebugIt
from PyTango.server import run
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute, command, pipe
from PyTango.server import class_property, device_property
from PyTango import AttrQuality, AttrWriteType, DispLevel, DevState

class PipeServer(Device):
    __metaclass__ = DeviceMeta

    def init_device(self):
        Device.init_device(self)
        self.rootBlobName = 'theBlob'
        self.__blob = self.rootBlobName, dict(x=0, y=10, width=100, height=200)

    @pipe(label="Test pipe", fisallowed="is_TestPipe_allowed")
    def TestPipe(self):
        return self.__blob

#     @TestPipe.write
#     def TestPipe(self, blob):
# #        self.__blob = blob
#         print blob
 
    def is_TestPipe_allowed(self, pipeReqType):
        """ pipeReqType is either READ_REQ or WRITE_REQ."""
        if pipeReqType == PyTango.AttReqType.READ_REQ:
            return self.get_state() not in [DevState.FAULT, DevState.OFF]
        else:             
            return self.get_state() not in [DevState.FAULT, DevState.OFF, DevState.MOVING]

    @command
    def cmd_push_pipe_event(self):
        print self.__blob
        self.push_pipe_event("TestPipe",self.__blob);

#
# Run server
#

def main():
    
    from PyTango import GreenMode
    from PyTango.server import run

    run([PipeServer,], green_mode=GreenMode.Gevent)

if __name__ == '__main__':
        main()

