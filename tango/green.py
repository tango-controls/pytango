# -----------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

__all__ = [
    "get_green_mode", "set_green_mode",
    "get_executor", "submit", "spawn", "result", "wait",
    "get_synch_executor", "synch_submit", "synch_wait",
    "get_gevent_executor", "gevent_submit", "gevent_wait",
    "get_futures_executor", "futures_submit", "futures_wait",
    "get_asyncio_executor", "asyncio_submit", "asyncio_wait",
    "get_object_submitter", "get_object_waiter", "get_object_green_mode",
    "green", "green_cb", "get_wait_default_value",
    "get_event_loop", "get_gevent_event_loop"]

__docformat__ = "restructuredtext"


# Imports
import os
from functools import wraps, partial

# Tango imports
from ._tango import GreenMode

# Gevent imports
from .tango_gevent import get_global_executor as get_gevent_executor
from .tango_gevent import submit as gevent_submit
from .tango_gevent import wait as gevent_wait
from .tango_gevent import get_event_loop as get_gevent_event_loop

# Futures imports
from .tango_futures import get_global_executor as get_futures_executor
from .tango_futures import submit as futures_submit
from .tango_futures import wait as futures_wait

# Asyncio imports
from .tango_asyncio import get_global_executor as get_asyncio_executor
from .tango_asyncio import submit as asyncio_submit
from .tango_asyncio import wait as asyncio_wait
from .tango_asyncio import get_event_loop as get_asyncio_event_loop


# Handle current green mode

__default_green_mode = GreenMode.Synchronous
try:
    __current_green_mode = getattr(
        GreenMode,
        os.environ["PYTANGO_GREEN_MODE"].capitalize())
except Exception:
    __current_green_mode = __default_green_mode


def set_green_mode(green_mode=None):
    """Sets the global default PyTango green mode.

    Advice: Use only in your final application. Don't use this in a python
    library in order not to interfere with the beavior of other libraries
    and/or application where your library is being.

    :param green_mode: the new global default PyTango green mode
    :type green_mode: GreenMode
    """
    global __current_green_mode
    if __current_green_mode == green_mode:
        return
    if green_mode == GreenMode.Gevent:
        # check if we can change to gevent mode
        import tango.gevent
    elif green_mode == GreenMode.Futures:
        # check if we can change to futures mode
        import tango.futures
    elif green_mode == GreenMode.Asyncio:
        # check if we can change to asyncio mode
        import tango.asyncio
    __current_green_mode = green_mode


def get_green_mode():
    """Returns the current global default PyTango green mode.

    :returns: the current global default PyTango green mode
    :rtype: GreenMode
    """
    return __current_green_mode


# Synchronous execution

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


# Excutor, submitter and waiter mapping

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


__event_loop_map = {
    GreenMode.Synchronous: (lambda: None),
    GreenMode.Futures:     (lambda: None),
    GreenMode.Gevent:      get_gevent_event_loop,
    GreenMode.Asyncio:     get_asyncio_event_loop,
}


# Getters

def get_executor(mode):
    return __executor_map[mode]()


def get_submitter(mode):
    return __submit_map[mode]


def get_waiter(mode):
    return __wait_map[mode]


def get_wait_default_value(mode):
    if mode is None:
        mode = get_green_mode()
    return mode not in (GreenMode.Asyncio,)


# Generic submitter/spawner and waiter

def submit(mode, fn, *args, **kwargs):
    return get_submitter(mode)(fn, *args, **kwargs)

spawn = submit


def wait_result(ret, green_mode=None, timeout=None):
    if green_mode is None:
        green_mode = get_green_mode()
    return get_waiter(green_mode)(ret, timeout=timeout)

wait = wait_result


def result(ret, green_mode=None, wait=True, timeout=None):
    if not wait:
        return ret
    return wait_result(ret, green_mode=green_mode, timeout=timeout)


def get_event_loop(mode):
    return __event_loop_map[mode]()


# Get object submitter, waiter and green_mode

def get_object_green_mode(obj):
    if hasattr(obj, "get_green_mode"):
        return obj.get_green_mode()
    return get_green_mode()


def get_object_submitter(obj, green_mode=None):
    """Returns the proper submit callable for the given object.

    If the object has *_executors* and *_green_mode* members it returns
    the submit callable for the executor corresponding to the green_mode.
    Otherwise it returns the global submit callable for the given green_mode.

    :returns: submit callable"""
    # Get green mode
    if green_mode is None:
        green_mode = get_object_green_mode(obj)
    # Get executor
    executors = getattr(obj, "_executors", {})
    executor = executors.get(green_mode, None)
    if executor is None:
        executor = get_executor(green_mode)
    # Get submitter
    if green_mode == GreenMode.Gevent:
        return executor.spawn
    return executor.submit


def get_object_waiter(obj, green_mode=None):
    """Returns the proper wait callable for the given object.

    If the object has *_executors* and *_green_mode* members it returns
    the wait callable for the executor corresponding to the green_mode.
    Otherwise it returns the global wait callable for the given green_mode.

    :returns: wait callable"""
    # Get green mode
    if green_mode is None:
        green_mode = get_object_green_mode(obj)
    # Get waiter
    waiter = get_waiter(green_mode)
    # Asyncio corner case
    if green_mode == GreenMode.Asyncio:
        executors = getattr(obj, "_executors", {})
        executor = executors.get(GreenMode.Asyncio)
        if executor is None:
            return waiter
        return partial(waiter, loop=executor.loop)
    # Return waiter
    return waiter


# Green decorators

def green(fn):
    """make a method green. Can be used as a decorator"""

    @wraps(fn)
    def greener(self, *args, **kwargs):
        # first take out all green parameters
        green_mode = kwargs.pop('green_mode', get_object_green_mode(self))
        wait = kwargs.pop('wait', get_wait_default_value(green_mode))
        timeout = kwargs.pop('timeout', None)

        # get the proper submitable for the given green_mode
        submitter = get_object_submitter(self, green_mode)
        waiter = get_object_waiter(self, green_mode)

        # submit the method
        ret = submitter(fn, self, *args, **kwargs)

        # return the proper result
        return waiter(ret, timeout=timeout) if wait else ret

    return greener


def green_cb(fn, green_mode=None):
    """return a green verion of the given callback."""
    event_loop = get_event_loop(green_mode)
    if event_loop is None:
        return fn

    @wraps(fn)
    def greener(*args, **kwargs):
        event_loop.submit(fn, *args, **kwargs)

    return greener
