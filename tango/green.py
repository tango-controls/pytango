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


# Imports
import os
from functools import wraps

# Compatibility imports
try:
    from threading import get_ident
except:
    from threading import _get_ident as get_ident

# Tango imports
from ._tango import GreenMode

__all__ = ('get_green_mode', 'set_green_mode', 'green', 'green_callback',
           'get_executor', 'get_object_executor')

# Handle current green mode

try:
    _CURRENT_GREEN_MODE = getattr(
        GreenMode,
        os.environ["PYTANGO_GREEN_MODE"].capitalize())
except Exception:
    _CURRENT_GREEN_MODE = GreenMode.Synchronous


def set_green_mode(green_mode=None):
    """Sets the global default PyTango green mode.

    Advice: Use only in your final application. Don't use this in a python
    library in order not to interfere with the beavior of other libraries
    and/or application where your library is being.

    :param green_mode: the new global default PyTango green mode
    :type green_mode: GreenMode
    """
    global _CURRENT_GREEN_MODE
    # Make sure the green mode is available
    get_executor(green_mode)
    # Set the green mode
    _CURRENT_GREEN_MODE = green_mode


def get_green_mode():
    """Returns the current global default PyTango green mode.

    :returns: the current global default PyTango green mode
    :rtype: GreenMode
    """
    return _CURRENT_GREEN_MODE


# Abstract executor class

class AbstractExecutor(object):
    asynchronous = NotImplemented
    default_wait = NotImplemented

    def __init__(self):
        self.thread_id = get_ident()

    def in_executor_context(self):
        return self.thread_id == get_ident()

    def delegate(self, fn, *args, **kwargs):
        """Delegate an operation and return an accessor."""
        if not self.asynchronous:
            raise ValueError('Not supported in synchronous mode')
        raise NotImplementedError

    def access(self, accessor, timeout=None):
        """Return a result from an accessor."""
        if not self.asynchronous:
            raise ValueError('Not supported in synchronous mode')
        raise NotImplementedError

    def submit(self, fn, *args, **kwargs):
        """Submit an operation"""
        if not self.asynchronous:
            return fn(*args, **kwargs)
        raise NotImplementedError

    def execute(self, fn, *args, **kwargs):
        """Execute an operation and return the result."""
        if not self.asynchronous:
            return fn(*args, **kwargs)
        raise NotImplementedError

    def run(self, fn, args=(), kwargs={}, wait=None, timeout=None):
        if wait is None:
            wait = self.default_wait
        # Wait and timeout are not supported in synchronous mode
        if not self.asynchronous and (not wait or timeout):
            raise ValueError('Not supported in synchronous mode')
        # Sychronous (no delegation)
        if not self.asynchronous or not self.in_executor_context():
            return fn(*args, **kwargs)
        # Asynchronous delegation
        accessor = self.delegate(fn, *args, **kwargs)
        if not wait:
            return accessor
        return self.access(accessor, timeout=timeout)


class SynchronousExecutor(AbstractExecutor):
    asynchronous = False
    default_wait = True


# Default synchronous executor

def get_synchronous_executor():
    return _SYNCHRONOUS_EXECUTOR


_SYNCHRONOUS_EXECUTOR = SynchronousExecutor()


# Getters

def get_object_green_mode(obj):
    if hasattr(obj, "get_green_mode"):
        return obj.get_green_mode()
    return get_green_mode()


def get_executor(green_mode=None):
    if green_mode is None:
        green_mode = get_green_mode()
    # Valid green modes
    if green_mode == GreenMode.Synchronous:
        return get_synchronous_executor()
    if green_mode == GreenMode.Gevent:
        from . import gevent_executor
        return gevent_executor.get_global_executor()
    if green_mode == GreenMode.Futures:
        from . import futures_executor
        return futures_executor.get_global_executor()
    if green_mode == GreenMode.Asyncio:
        from . import asyncio_executor
        return asyncio_executor.get_global_executor()
    # Invalid green mode
    raise TypeError("Not a valid green mode")


def get_object_executor(obj, green_mode=None):
    """Returns the proper executor for the given object.

    If the object has *_executors* and *_green_mode* members it returns
    the submit callable for the executor corresponding to the green_mode.
    Otherwise it returns the global executor for the given green_mode.

    Note: *None* is a valid object.

    :returns: submit callable"""
    # Get green mode
    if green_mode is None:
        green_mode = get_object_green_mode(obj)
    # Get executor
    executor = None
    if hasattr(obj, '_executors'):
        executor = obj._executors.get(green_mode, None)
    if executor is None:
        executor = get_executor(green_mode)
    # Get submitter
    return executor


# Green modifiers

def green(fn=None, consume_green_mode=True):
    """Make a function green. Can be used as a decorator."""

    def decorator(fn):
        @wraps(fn)
        def greener(obj, *args, **kwargs):
            args = (obj,) + args
            wait = kwargs.pop('wait', None)
            timeout = kwargs.pop('timeout', None)
            access = kwargs.pop if consume_green_mode else kwargs.get
            green_mode = access('green_mode', None)
            executor = get_object_executor(obj, green_mode)
            return executor.run(fn, args, kwargs, wait=wait, timeout=timeout)

        return greener

    if fn is None:
        return decorator
    return decorator(fn)


def green_callback(fn, obj=None, green_mode=None):
    """Return a green verion of the given callback."""
    executor = get_object_executor(obj, green_mode)

    @wraps(fn)
    def greener(*args, **kwargs):
        return executor.submit(fn, *args, **kwargs)

    return greener
