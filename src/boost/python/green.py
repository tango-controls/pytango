# ------------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

__all__ = ["get_green_mode", "set_green_mode",
           "get_executor", "submit", "spawn",
           "get_synch_executor", "synch_submit", "synch_wait",
           "get_gevent_executor", "gevent_submit", "gevent_wait",
           "get_futures_executor", "futures_submit", "futures_wait",
           "get_asyncio_executor", "asyncio_submit", "asyncio_wait",
           "wait", "submitable", "green"]

__docformat__ = "restructuredtext"

import os
from functools import wraps

from ._PyTango import GreenMode

# Tango gevent
from .tango_gevent import get_global_executor as get_gevent_executor
from .tango_gevent import submit as gevent_submit
from .tango_gevent import wait as gevent_wait

# Tango futures
from .tango_futures import get_global_executor as get_futures_executor
from .tango_futures import submit as futures_submit
from .tango_futures import wait as futures_wait

# Tango asyncio
from .tango_asyncio import get_global_executor as get_asyncio_executor
from .tango_asyncio import submit as asyncio_submit
from .tango_asyncio import wait as asyncio_wait

__default_green_mode = GreenMode.Synchronous
try:
    __current_green_mode = getattr(GreenMode,
                                   os.environ.get("PYTANGO_GREEN_MODE",
                                                  "Synchronous").capitalize())
except:
    __current_green_mode = __default_green_mode


def set_green_mode(green_mode=None):
    """Sets the global default PyTango green mode.

    Advice: Use only in your final application. Don't use this in a python library
    in order not to interfere with the beavior of other libraries and/or
    application where your library is being.

    :param green_mode: the new global default PyTango green mode
    :type green_mode: GreenMode
    """
    global __current_green_mode
    if __current_green_mode == green_mode:
        return
    if green_mode == GreenMode.Gevent:
        # check if we can change to gevent mode
        import PyTango.gevent
    elif green_mode == GreenMode.Futures:
        # check if we can change to futures mode
        import PyTango.futures
    elif green_mode == GreenMode.Asyncio:
        # check if we can change to asyncio mode
        import PyTango.asyncio
    __current_green_mode = green_mode


def get_green_mode():
    """Returns the current global default PyTango green mode.

    :returns: the current global default PyTango green mode
    :rtype: GreenMode
    """
    return __current_green_mode


class SynchExecutor(object):
    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

__synch_executor = SynchExecutor()

def get_synch_executor():
    return __synch_executor

def synch_submit(fn, *args, **kwargs):
    return get_synch_executor().submit(fn, *args, **kwargs)

def synch_wait(res, timeout=None):
    return res

__executor_map = {
    GreenMode.Synchronous: get_synch_executor,
    GreenMode.Futures:     get_futures_executor,
    GreenMode.Gevent:      get_gevent_executor,
    GreenMode.Asyncio:     get_asyncio_executor,
}

__submit_map = {
    GreenMode.Synchronous: synch_submit,
    GreenMode.Futures:     futures_submit,
    GreenMode.Gevent:      gevent_submit,
    GreenMode.Asyncio:     asyncio_submit,
}

__wait_map = {
    GreenMode.Synchronous: synch_wait,
    GreenMode.Futures:     futures_wait,
    GreenMode.Gevent:      gevent_wait,
    GreenMode.Asyncio:     asyncio_wait,
}

def get_executor(mode):
    return __executor_map[mode]()

def get_submitter(mode):
    executor = get_executor(mode)
    if mode == GreenMode.Gevent:
        return executor.spawn
    return executor.submit

def get_waiter(mode):
    return __wait_map[mode]

def submit(mode, fn, *args, **kwargs):
    return get_submitter(mode)(fn, *args, **kwargs)

spawn = submit

def wait(ret, green_mode=None, timeout=None):
    green_mode = green_mode or get_green_mode()
    return get_waiter(green_mode)(ret, timeout=timeout)

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
        return synch_submit, synch_wait

    has_executors = hasattr(obj, "_executors")
    s_func = __submit_map[green_mode]
    w_func = __submit_map[green_mode]
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
    elif green_mode == GreenMode.Asyncio:
        if has_executors:
            executor = obj._executors.get(GreenMode.Asyncio)
            if executor:
                s_func = excutor.submit
                w_func = partial(w_func, loop=executor.loop)
    else:
        raise TypeError("Undefined green_mode '%s' for %s" % (str(green_mode)), str(obj))
    return s_func, w_func

def green(fn):
    """make a method green. Can be used as a decorator"""

    @wraps(fn)
    def greener(self, *args, **kwargs):
        # first take out all green parameters
        green_mode = kwargs.pop('green_mode', None)
        wait = kwargs.pop('wait', True)
        timeout = kwargs.pop('timeout', None)

        # get the proper submitable for the given green_mode
        submit_func, wait_func = submitable(self, green_mode)

        # submit the method
        ret = submit_func(fn, self, *args, **kwargs)

        # return the proper result
        return wait_func(ret, timeout=timeout) if wait else ret
    return greener
