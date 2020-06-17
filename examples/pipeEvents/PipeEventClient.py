import time

import tango
from tango import EventType


class EventManager():

    def __init__(self, dp):
        self._deviceProxy = dp
        if dp is not None:
            print "Subscribed to TestPipe"
            self._event_id = dp.subscribe_event("TestPipe",
                                                EventType.PIPE_EVENT,
                                                self)

    def unsubscribe(self):
        self._deviceProxy.unsubscribe_event(self._event_id)

    def push_event(self, ev):
        print "Event -----push_event-----------------"
        print "Timestamp:      ", ev.reception_date
        print "Pipe name:      ", ev.pipe_name
        print "Event type:     ", ev.event
        print "Device server:  ", ev.device
        print "Event error:    ", ev.err
        if ev.err:
            print "Caught pipe exception"
            err = ev.errors[0]
            print "Error desc:     ", err.desc
            print "Error origin:   ", err.origin
            print "Error reason:   ", err.reason
            print "Error severity: ", err.severity
        else:
            print "Nb data elts:   ", ev.pipe_value.data_elt_nb
            print "Event name:     ", ev.pipe_value.name
            print "Root blob name: ", ev.pipe_value.root_blob_name
            print "Elements names: ", ev.pipe_value.data_elt_names
            nb_elt = ev.pipe_value.data_elt_nb
            for i in range(nb_elt):
                print "Elements:       ", ev.pipe_value.data[i]


def main():
    dev = tango.DeviceProxy('pipeServer/tango/1')
    print dev
    EventManager(dev)
    time.sleep(3000.0)


if __name__ == '__main__':
    main()
