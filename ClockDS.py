#!/usr/bin/env python

"""
Clock Device server showing how to write a TANGO server with a Clock device
which has attributes:

  - time: read-only scalar float
  - gmtime: read-only sequence (spectrum) of integers

commands:

  - ctime: in: float parameter; returns a string
  - mktime: in: sequence (spectrum) of 9 integers; returns a float
"""

import time
from tango import GreenMode, DevState
from tango.server import Device, attribute, command, pipe


class Clock(Device):

    def init_device(self):
        print("Clock server init_device -----------------------------------")
        Device.init_device(self)
        self.__count = 0
        self.__rootBlobName = 'theBlob'
        self.__blob = self.__rootBlobName, dict(x=[0.3,0.4], y=[1,2,3,4,5])
        self.__boolean = True
        self.set_state(DevState.ON)

    def always_executed_hook(self):
        self.__count += 1

#    voltage = attribute(name="voltage", label='Voltage', forwarded=True)

    @attribute(dtype=float)
    def time(self):
        return 6.284;

    @attribute(label='Counter', dtype=int)
    def counter(self):
        return self.__count

    @attribute(dtype=(int,), max_dim_x=9)
    def gmtime(self):
        return time.gmtime()

    @command
    def rubbish(self):
        print("this is rubbish {}".format(self.__count))

    @command(dtype_in=float)
    def rubbish2(self, value):
        print("this is rubbish {}".format(value))

    @command(dtype_out=int)
    def rubbish3(self):
        print("this is rubbish3 {}".format(self.__count))
        return self.__count

    @command(dtype_out='double')
    def rubbish4(self):
        print("this is rubbish4 {}".format(self.__count))
        return 3.142

    @command(dtype_out=bool)
    def rubbish5(self):
        self.__boolean = not self.__boolean
        print("this is rubbish5 {}".format(self.__boolean))
        return self.__boolean

    @command(dtype_out='str')
    def rubbish6(self):
        print("this is rubbish6 {}".format(self.__count))
        return "this is rubbish6 {}".format(self.__count)

    @command(dtype_out='DevState')
    def rubbish7(self):
        print("this is rubbish7 {}".format(self.__count))
        return DevState.OFF;

    @command(dtype_out='DevEncoded')
    def rubbish8(self):
        print("this is rubbish8 {}".format(self.__count))
        encoded_format = "10i"
        encoded_data = [1,2,3,4,5,6,7,8,9,10]
        tup = (encoded_format, encoded_data);
        return tup

    @command(dtype_out='DevVarShortArray')
    def rubbish9(self):
        print("this is rubbish9 {}".format(self.__count))
        data = [[1,2,3,4,5,6],[7,8,9,10,32765,32767]]
        return data

    @command(dtype_out='DevVarDoubleArray')
    def rubbish10(self):
        print("this is rubbish10 {}".format(self.__count))
        data = [1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.0,100.1,32765.,32767.]
        return data

    @command(dtype_out='DevVarStateArray')
    def rubbish11(self):
        print("this is rubbish11 {}".format(self.__count))
        data = [DevState.OFF, DevState.STANDBY, DevState.ON]
        return data

    @command(dtype_out='DevVarBooleanArray')
    def rubbish12(self):
        print("this is rubbish12 {}".format(self.__count))
        data = [True, True, False, False, True]
        return data

    @command(dtype_in=float, dtype_out=str)
    def ctime(self, seconds):
        """
        Convert a time in seconds since the Epoch to a string in local time.
        This is equivalent to asctime(localtime(seconds)). When the time tuple
        is not present, current time as returned by localtime() is used.
        """
        print(seconds)
        tim_str = time.ctime(seconds)
        print(tim_str)
        return tim_str
  
#     @command(dtype_in=(int,), dtype_out=float)
#     def mktime(self, tupl):
#         return time.mktime(tupl)

    @pipe(label="Test pipe")
    def TestPipe(self):
        return self.__blob

    @TestPipe.write
    def TestPipe(self, blob):
#        self.__blob = blob
        print blob


if __name__ == "__main__":
    Clock.run_server(green_mode=GreenMode.Gevent)
