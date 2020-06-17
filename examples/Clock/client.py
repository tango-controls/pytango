#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2013-2015 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

"""
Simple client to show how to connect to a Clock device from ClockDS

usage: client clock_dev_name
"""

import sys
import tango

if len(sys.argv) != 2:
    print("must provide one and only one clock device name")
    sys.exit(1)

clock = tango.DeviceProxy(sys.argv[1])
t = clock.time
gmt = clock.gmtime
noon = clock.noon
display = clock.display
print(t)
print(gmt)
print(noon, noon.name, noon.value)
if noon == noon.AM:
    print('Good morning!')
print(clock.ctime(t))
print(clock.mktime(gmt))
print(display, display.name, display.value)
clock.display = display.ANALOG
clock.display = 'DIGITAL'  # you can use a valid string to set the value
print(clock.display, clock.display.name, clock.display.value)
display_type = type(display)  # or even create your own IntEnum type
analog = display_type(0)
clock.display = analog
print(clock.display, clock.display.name, clock.display.value)
clock.display = clock.display.DIGITAL
print(clock.display, clock.display.name, clock.display.value)
