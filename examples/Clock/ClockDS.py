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
from PyTango.server import Device, attribute, command


class Clock(Device):

    @attribute(dtype=float)
    def time(self):
        return time.time()

    gmtime = attribute(dtype=(int,), max_dim_x=9)

    def read_gmtime(self):
        return time.gmtime()

    @command(dtype_in=float, dtype_out=str)
    def ctime(self, seconds):
        """
        Convert a time in seconds since the Epoch to a string in local time.
        This is equivalent to asctime(localtime(seconds)). When the time tuple
        is not present, current time as returned by localtime() is used.
        """
        return time.ctime(seconds)

    @command(dtype_in=(int,), dtype_out=float)
    def mktime(self, tupl):
        return time.mktime(tupl)


if __name__ == "__main__":
    Clock.run_server()
