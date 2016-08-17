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

from __future__ import absolute_import

__all__ = ["get_global_executor", "submit", "spawn", "wait", "get_event_loop"]

__global_executor = None


def __get_executor_class():
    # Imports
    import concurrent.futures
    from functools import partial
    try:
        import asyncio
    except ImportError:
        import trollius as asyncio

    # Asyncio executor
    class AsyncioExecutor(concurrent.futures.Executor):
        """Executor to submit task to a subexecutor through an asyncio loop.
        Warning: This class has nothing to do with the AsyncioExecutor class
        implemented for the server.
        """

        def __init__(self, loop=None, subexecutor=None):
            self.subexecutor = subexecutor
            self.loop = loop or asyncio.get_event_loop()

        def submit(self, fn, *args, **kwargs):
            callback = partial(fn, *args, **kwargs)
            return self.loop.run_in_executor(self.subexecutor, callback)

    # Return
    return AsyncioExecutor


def get_global_executor():
    # Get global executor
    global __global_executor
    if __global_executor is not None:
        return __global_executor
    # Import futures executor
    try:
        from .tango_futures import get_global_executor as get_futures_executor
    except ImportError:
        get_futures_executor = lambda: None
    # Set global
    loop = get_event_loop()
    klass = __get_executor_class()
    subexecutor = get_futures_executor()
    __global_executor = klass(loop, subexecutor)
    return __global_executor


def submit(fn, *args, **kwargs):
    return get_global_executor().submit(fn, *args, **kwargs)

spawn = submit


def wait(fut, timeout=None, loop=None):
    # Imports
    try:
        import asyncio
    except ImportError:
        import trollius as asyncio
    # Run loop
    loop = loop or asyncio.get_event_loop()
    coro = asyncio.wait_for(fut, timeout, loop=loop)
    return loop.run_until_complete(coro)


def get_event_loop():
    # Imports
    try:
        import asyncio
    except ImportError:
        import trollius as asyncio
    # Get loop
    global __event_loop
    if __event_loop is not None:
        return __event_loop
    # Create loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def submit(fn, *args, **kwargs):
        callback = lambda: fn(*args, **kwargs)
        return loop.call_soon_threadsafe(callback)

    # Patch loop
    loop.submit = submit
    __event_loop = loop
    return loop


__event_loop = None
