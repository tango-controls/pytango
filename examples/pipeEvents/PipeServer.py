# PyTango imports

import PyTango
from PyTango import DebugIt
from PyTango.server import run
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute, command, pipe
from PyTango.server import class_property, device_property
from PyTango import AttrQuality, AttrWriteType, DispLevel, DevState

import time
import gevent
from gevent import socket, event, lock

class PipeServer(Device):
    __metaclass__ = DeviceMeta

    def init_device(self):
        Device.init_device(self)
        self.rootBlobName = 'theBlob'
        self.__blob = self.rootBlobName, dict(x=5, y=7, width=101, height=230)
        self._send_task = None
        
    @pipe(label="Test pipe", fisallowed="is_TestPipe_allowed")
    def TestPipe(self):
        return self.__blob

    def is_TestPipe_allowed(self, pipeReqType):
        """ pipeReqType is either READ_REQ or WRITE_REQ."""
        if pipeReqType == PyTango.AttReqType.READ_REQ:
            return self.get_state() not in [DevState.FAULT, DevState.OFF]
        else:             
            return self.get_state() not in [DevState.FAULT, DevState.OFF, DevState.MOVING]

    @command(dtype_in=('int'), doc_in="Pipe event test case 0 - 4")
    def cmd_push_pipe_event(self, argin):
        if argin == 0:
            float_list = [3.33,3.34,3.35,3.36]
            inner_int_list = [11,12,13,14,15]
            inner_inner_data = [("InnerInnerFirstDE",111),("InnerInnerSecondDE",float_list),("InnerInnerThirdDE",inner_int_list)]
            inner_inner_blob = ("InnerInner",dict(inner_inner_data))
            inner_data = [("InnerFirstDE","Grenoble"),("InnerSecondDE",inner_inner_blob),("InnerThirdDE",True)]
            inner_blob = ("Inner", dict(inner_data))
            int_list = [3,4,5,6]
            pipe_data=[("1DE",inner_blob),("2DE",int_list)]
            blob = ("PipeEvent0", dict(pipe_data))
            self.push_pipe_event("TestPipe", blob);
        elif argin == 1:
            pipeData=[("Another_1DE",2),("Another_2DE","Barcelona"),("Another_3DE",45.67)]
            blob = "PipeEventCase1", dict(pipeData)
            self.push_pipe_event("TestPipe", blob);
        elif argin == 2:
            float_list = [3.142, 6.284, 12.568]
            string_list = ["ESRF", "Alba", "MAXIV"]
            pipeData=[("Qwerty_1DE","Barcelona"),("Azerty_2DE", float_list),("Xserty", string_list)]
            blob = "PipeEventCase2", dict(pipeData)
            self.push_pipe_event("TestPipe", blob)
        elif argin == 3:
            print "not coded yet"
        elif argin == 4:
            alist = [ k for k in range(30)]
            pipeData=[("Lunes", "Girona"),("Martes", alist)]
            blob = "PipeEventCase4", dict(pipeData)
            self.push_pipe_event("TestPipe", blob);
        else:
            print "Invalid test case: Use 0-4"
 
#
# Run server
#

def main():
    
    from PyTango import GreenMode
    from PyTango.server import run

    run([PipeServer,], green_mode=GreenMode.Gevent)

if __name__ == '__main__':
        main()

