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

# Future imports
from __future__ import absolute_import

# Imports
import functools

# Asyncio imports
try:
    import asyncio
except ImportError:
    import trollius as asyncio
try:
    from asyncio import run_coroutine_threadsafe
except ImportError:
    from .asyncio_tools import run_coroutine_threadsafe

# Tango imports
from .green import AbstractExecutor

__all__ = ("AsyncioExecutor", "get_global_executor", "set_global_executor")


# Global executor

_EXECUTOR = None


def get_global_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = AsyncioExecutor()
    return _EXECUTOR


def set_global_executor(executor):
    global _EXECUTOR
    _EXECUTOR = executor


# Asyncio executor

class AsyncioExecutor(AbstractExecutor):
    """Asyncio tango executor"""

    asynchronous = True
    default_wait = False

    def __init__(self, loop=None, subexecutor=None):
        super(AsyncioExecutor, self).__init__()
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        self.loop = loop
        self.subexecutor = subexecutor

    def delegate(self, fn, *args, **kwargs):
        """Return the given operation as an asyncio future."""
        callback = functools.partial(fn, *args, **kwargs)
        coro = self.loop.run_in_executor(self.subexecutor, callback)
        return asyncio.ensure_future(coro)

    def access(self, accessor, timeout=None):
        """Return a result from an asyncio future."""
        if self.loop.is_running():
            raise RuntimeError("Loop is already running")
        coro = asyncio.wait_for(accessor, timeout, loop=self.loop)
        return self.loop.run_until_complete(coro)

    def submit(self, fn, *args, **kwargs):
        """Submit an operation"""
        corofn = asyncio.coroutine(lambda: fn(*args, **kwargs))
        return run_coroutine_threadsafe(corofn(), self.loop)

    def execute(self, fn, *args, **kwargs):
        """Execute an operation and return the result."""
        if self.in_executor_context():
            corofn = asyncio.coroutine(lambda: fn(*args, **kwargs))
            return corofn()
        future = self.submit(fn, *args, **kwargs)
        return future.result()
