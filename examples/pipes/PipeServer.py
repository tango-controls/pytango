import tango
from tango.server import Device
from tango.server import pipe
from tango import DevState


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

# ----------
# Run server
# ----------


if __name__ == '__main__':
    PipeServer.run_server()
