# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

# Future imports
from __future__ import absolute_import

# Concurrent imports
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Tango imports
from .green import AbstractExecutor

__all__ = ("FuturesExecutor", "get_global_executor", "set_global_executor")

# Global executor

_EXECUTOR = None


def get_global_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = FuturesExecutor()
    return _EXECUTOR


def set_global_executor(executor):
    global _EXECUTOR
    _EXECUTOR = executor


# Futures executor

class FuturesExecutor(AbstractExecutor):
    """Futures tango executor"""

    asynchronous = True
    default_wait = True

    def __init__(self, process=False, max_workers=20):
        super(FuturesExecutor, self).__init__()
        cls = ProcessPoolExecutor if process else ThreadPoolExecutor
        self.subexecutor = cls(max_workers=max_workers)

    def delegate(self, fn, *args, **kwargs):
        """Return the given operation as a concurrent future."""
        return self.subexecutor.submit(fn, *args, **kwargs)

    def access(self, accessor, timeout=None):
        """Return a result from a single callable."""
        return accessor.result(timeout=timeout)

    def submit(self, fn, *args, **kwargs):
        """Submit an operation"""
        return fn(*args, **kwargs)

    def execute(self, fn, *args, **kwargs):
        """Execute an operation and return the result."""
        return fn(*args, **kwargs)
