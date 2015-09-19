# -----------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

__all__ = ["get_global_executor", "submit", "spawn", "wait"]

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
    class AsyncioExecutor(concurrent.futures.Futures):
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
    klass = __get_executor_class()
    subexecutor = get_futures_executor()
    __global_executor = klass(subexecutor=subexecutor)
    return __global_executor


def submit(fn, *args, **kwargs):
    return get_global_executor().submit(fn, *args, **kwargs)


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


spawn = submit
