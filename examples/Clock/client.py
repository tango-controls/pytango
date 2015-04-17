#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
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
import PyTango

if len(sys.argv) != 2:
    print "must provide one and only one clock device name"
    sys.exit(1)

clock = PyTango.DeviceProxy(sys.argv[1])
t = clock.time
gmt = clock.gmtime
print(t)
print(gmt)
print(clock.ctime(t))
print(clock.mktime(gmt))




