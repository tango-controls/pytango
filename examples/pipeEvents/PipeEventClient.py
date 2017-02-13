import PyTango
import numpy
import time
from PyTango import DevState

class EventManager():
     
    def __init__(self, dp):
        self._deviceProxy = dp
        if dp is not None:
            self._event_id = dp.subscribe_event("TestPipe", PyTango.EventType.PIPE_EVENT, self)
 
    def unsubscribe(self):
        self._deviceProxy.unsubscribe_event(self._event_id)
 
    def push_event(self, ev):
        print "Event -----push_event-----------------"
        print ev.reception_date
        print ev.pipe_name
        print ev.event
        print ev.err
        print ev.device
        print ev.pipe_value
        blob = ev.pipe_value
        print blob.data_elt_nb
        print blob.name
        print blob.root_blob_name
        print blob.data_elt_names
        for name in blob.data_elt_names:
            print name
        

def main():
    dev=PyTango.DeviceProxy('pipeServer/tango/1')
    print dev
    em = EventManager(dev)
    time.sleep(30.0)

if __name__ == '__main__':
    main()
