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

__all__ = ["get_executor", "submit", "spawn",
           "get_synch_executor", "synch_submit",
           "get_gevent_executor", "gevent_submit",
           "get_futures_executor", "futures_submit"
           "result", "submitable", "green"] 

__docformat__ = "restructuredtext"

from functools import wraps

from ._PyTango import GreenMode
from .tango_gevent import get_global_executor as get_gevent_executor
from .tango_gevent import submit as gevent_submit
from .tango_futures import get_global_executor as get_futures_executor
from .tango_futures import submit as futures_submit
from .utils import get_green_mode

class SynchExecutor(object):
    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

__synch_executor = SynchExecutor()

def get_synch_executor():
    return __synch_executor

def synch_submit(fn, *args, **kwargs):
    return get_synch_executor().submit(fn, *args, **kwargs)

__executor_map = {
    GreenMode.Synchronous: get_synch_executor,
    GreenMode.Futures:     get_futures_executor,
    GreenMode.Gevent:      get_gevent_executor,
}

__submit_map = {
    GreenMode.Synchronous: synch_submit,
    GreenMode.Futures:     futures_submit,
    GreenMode.Gevent:      gevent_submit,
}

def get_executor(mode):
    return __executor_map[mode]()

def get_submitter(mode):
    executor = get_executor(mode)
    if mode == GreenMode.Gevent:
        return executor.spawn
    return executor.submit

def submit(mode, fn, *args, **kwargs):
    return get_submitter(mode)(fn, *args, **kwargs)

spawn = submit

def result(value, green_mode, wait=True, timeout=None):
    if wait and not green_mode is GreenMode.Synchronous:
        if green_mode == GreenMode.Futures:
            return value.result(timeout=timeout)
        elif green_mode == GreenMode.Gevent:
            return value.get(timeout=timeout)
    return value

def submitable(obj, green_mode=None):
    """Returns a proper submit callable for the given object.

    If the object has *_executors* and *_green_mode* members it returns a submit
    callable for the executor corresponding to the green_mode.
    Otherwise it returns the global submit callable for the given green_mode

    :returns: green_mode, submit callable"""
    # determine the efective green_mode
    if green_mode is None:
        if hasattr(obj, "get_green_mode"):
            green_mode = obj.get_green_mode()
        else:
            green_mode = get_green_mode()
    
    if green_mode == GreenMode.Synchronous:
        return green_mode, synch_submit
    
    has_executors = hasattr(obj, "_executors")
    s_func = __submit_map[green_mode]
    if green_mode == GreenMode.Futures:
        if has_executors:
            executor = obj._executors.get(GreenMode.Futures)
            if executor:
                s_func = executor.submit
    elif green_mode == GreenMode.Gevent:
        if has_executors:
            executor = obj._executors.get(GreenMode.Gevent)
            if executor:
                s_func = executor.spawn
    else:
        raise TypeError("Undefined green_mode '%s' for %s" % (str(green_mode)), str(obj))
    return green_mode, s_func

def green(fn):
    """make a method green. Can be used as a decorator"""

    @wraps(fn)
    def greener(self, *args, **kwargs):
        # first take out all green parameters
        green_mode = kwargs.pop('green_mode', None)
        wait = kwargs.pop('wait', True)
        timeout = kwargs.pop('timeout', None)

        # get the proper submitable for the given green_mode
        green_mode, submit = submitable(self, green_mode)

        # submit the method
        ret = submit(fn, self, *args, **kwargs)
        
        # return the proper result        
        return result(ret, green_mode, wait=wait, timeout=timeout)
    return greener     

