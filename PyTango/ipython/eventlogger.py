################################################################################
##
## This file is part of PyTango, a python binding for Tango
## 
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
## 
## PyTango is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## PyTango is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################

from __future__ import print_function

import re
import io
import operator

class EventLogger(object):
    
    def __init__(self, capacity=100000, pager=None):
        self._capacity = capacity
        self._pager = pager
        self._records = []
        
    def push_event(self, evt):
        attr_name = evt.attr_name
        dev, sep, attr = attr_name.rpartition('/')
        if dev.startswith("tango://"):
            dev = dev[8:]
        if dev.count(":"):
            # if it has tango host
            host, sep, dev = dev.partition('/')
        else:
            host = "-----"
        evt.host = host
        evt.dev_name = dev
        evt.s_attr_name = attr
        self._records.append(evt)
        over = len(self._records) - self._capacity
        if over > 0:
            self._records = self._records[over:]
    
    def model(self):
        return self
    
    def getEvents(self):
        return self._records
    
    def show(self, dexpr=None, aexpr=None):
        if dexpr is not None:
            dexpr = re.compile(dexpr, re.IGNORECASE)
        if aexpr is not None:
            aexpr = re.compile(aexpr, re.IGNORECASE)
            
        s = io.BytesIO()
        lengths = 4, 30, 18, 20, 12, 16
        title = 'ID', 'Device', 'Attribute', 'Value', 'Quality', 'Time'
        templ = "{0:{l[0]}} {1:{l[1]}} {2:{l[2]}} {3:{l[3]}} {4:{l[4]}} {5:{l[5]}}"
        print(templ.format(*title, l=lengths), file=s)
        print(*map(operator.mul, lengths, len(lengths)*"-"), file=s)
        
        for i,r in enumerate(self._records):
            if dexpr is not None and not dexpr.match(r.dev_name): continue
            if aexpr is not None and not aexpr.match(r.s_attr_name): continue
            if r.err:
                v = r.errors[0].reason
                q = 'ERROR'
                ts = r.reception_date.strftime("%H:%M:%S.%f")
            else:
                v = str(r.attr_value.value)
                q = str(r.attr_value.quality)
                ts = r.attr_value.time.strftime("%H:%M:%S.%f")
            msg = templ.format(i, r.dev_name, r.s_attr_name, v, q, ts, l=lengths)
            print(msg, file=s)
        s.seek(0)
        if self._pager is None:
            print(s.read())
        else:
            self._pager(s.read())
