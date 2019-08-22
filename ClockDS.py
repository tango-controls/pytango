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
from tango import GreenMode
from tango.server import Device, attribute, command


class Clock(Device):

    def init_device(self):
        print("got here in python")
        Device.init_device(self)

    @attribute(dtype=float)
    def time(self):
        return time.time()
 
    gmtime = attribute(dtype=(int,), max_dim_x=9)
 
    def read_gmtime(self):
        return time.gmtime()
 
    @command
    def rubbish(self):
        print("this is rubbish")

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


if __name__ == "__main__":
    Clock.run_server(green_mode=GreenMode.Gevent)
