import tango
from tango.server import run
from tango.server import Device
from tango.server import attribute, command, pipe
from tango.server import class_property, device_property
from tango import AttrQuality, AttrWriteType, DispLevel, DevState


class PipeServer(Device):

    def init_device(self):
        Device.init_device(self)
        self.rootBlobName = 'theBlob'
        self.__blob = self.rootBlobName, dict(x=0.3, y=10.22, 
                                              width=105.1, height=206.6)

    @pipe(label="Test pipe", fisallowed="is_TestPipe_allowed")
    def TestPipe(self):
        return self.__blob

    @TestPipe.write
    def TestPipe(self, blob):
        self.__blob = blob
        print blob

    def is_TestPipe_allowed(self, pipeReqType):
        """ pipeReqType is either READ_REQ or WRITE_REQ."""
        if pipeReqType == tango.AttReqType.READ_REQ:
            return self.get_state() not in [DevState.FAULT,
                                            DevState.OFF]
        else:
             return self.get_state() not in [DevState.FAULT,
                                             DevState.OFF, 
                                             DevState.MOVING]

#----------
# Run server
# ----------

def main():
    from tango import GreenMode
    from tango.server import run
    run([PipeServer,], green_mode=GreenMode.Gevent)

if __name__ == '__main__':
    main()
