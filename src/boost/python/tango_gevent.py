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

__all__ = ["get_global_threadpool", "get_global_executor", "submit", "spawn"] 

__docformat__ = "restructuredtext"

def get_global_threadpool():
    import gevent
    return gevent.get_hub().threadpool

get_global_executor = get_global_threadpool

def spawn(fn, *args, **kwargs):
    return get_global_threadpool().spawn(fn, *args, **kwargs)

submit = spawn
