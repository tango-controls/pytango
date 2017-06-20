from __future__ import print_function

import time

import tango


class EventManager():

    def __init__(self, dp):
        self._deviceProxy = dp
        if dp is not None:
            print("Subscribed to Interface Change Events")
            self._event_id = dp.subscribe_event(
                tango.EventType.INTERFACE_CHANGE_EVENT, self)

    def unsubscribe(self):
        self._deviceProxy.unsubscribe_event(self._event_id)

    def push_event(self, ev):
        print("Event -----push_event-----------------")
        print("Timestamp:      ", ev.reception_date)
        print("Event type:     ", ev.event)
        print("Device server:  ", ev.device)
        print("Event error:    ", ev.err)
        if ev.err:
            print("Caught pipe exception")
            err = ev.errors[0]
            print("Error desc:     ", err.desc)
            print("Error origin:   ", err.origin)
            print("Error reason:   ", err.reason)
            print("Error severity: ", err.severity)
        else:
            if ev.cmd_list is not None:
                print("Number of commands   ", len(ev.cmd_list))
                for i in range(len(ev.cmd_list)):
                    cmdInfo = ev.cmd_list[i]
                    print("cmd  -----> ", cmdInfo.cmd_name)
            else:
                print("Number of commands    0")

            if ev.att_list is not None:
                print("Number of attributes ", len(ev.att_list))
                for i in range(len(ev.att_list)):
                    attInfo = ev.att_list[i]
                    print("att  -----> ", attInfo.name)
            else:
                print("Number of attributes  0")

            print("Device started ", ev.dev_started)


def main():
    dev = tango.DeviceProxy('ifchangeServer/tango/1')
    EventManager(dev)
    time.sleep(3000.0)


if __name__ == '__main__':
    main()
