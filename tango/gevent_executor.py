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

# Imports
import sys
import six
import types
import functools

# Gevent imports
import gevent.queue
import gevent.monkey

# Bypass gevent monkey patching
ThreadSafeEvent = gevent.monkey.get_original('threading', 'Event')

# Tango imports
from .green import AbstractExecutor


__all__ = ["get_global_executor", "set_global_executor", "GeventExecutor"]

# Global executor

_EXECUTOR = None


def get_global_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = GeventExecutor()
    return _EXECUTOR


def set_global_executor(executor):
    global _EXECUTOR
    _EXECUTOR = executor


# Patch for gevent threadpool

def get_global_threadpool():
    """Before gevent-1.1.0, patch the spawn method to propagate exception
    raised in the loop to the AsyncResult.
    """
    threadpool = gevent.get_hub().threadpool
    if gevent.version_info < (1, 1) and not hasattr(threadpool, '_spawn'):
        threadpool._spawn = threadpool.spawn
        threadpool.spawn = types.MethodType(
            spawn, threadpool, type(threadpool))
    return threadpool


class ExceptionWrapper:
    def __init__(self, exception, error_string, tb):
        self.exception = exception
        self.error_string = error_string
        self.tb = tb


def wrap_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return ExceptionWrapper(*sys.exc_info())

    return wrapper


def get_with_exception(result, block=True, timeout=None):
    result = result._get(block, timeout)
    if isinstance(result, ExceptionWrapper):
        # Raise the exception using the caller context
        six.reraise(result.exception, result.error_string, result.tb)
    return result


def spawn(threadpool, fn, *args, **kwargs):
    # The gevent threadpool do not raise exception with async results,
    # we have to wrap it
    fn = wrap_errors(fn)
    result = threadpool._spawn(fn, *args, **kwargs)
    result._get = result.get
    result.get = types.MethodType(get_with_exception, result, type(result))
    return result


# Gevent task and event loop

class GeventTask:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.value = None
        self.exception = None
        self.event = ThreadSafeEvent()

    def run(self):
        try:
            self.value = self.func(*self.args, **self.kwargs)
        except:
            self.exception = sys.exc_info()
        finally:
            self.event.set()

    def spawn(self):
        return gevent.spawn(self.run)

    def result(self):
        self.event.wait()
        if self.exception:
            six.reraise(*self.exception)
        return self.value


class GeventLoop:
    def __init__(self):
        self.hub = gevent.get_hub()

    def submit(self, func, *args, **kwargs):
        task = GeventTask(func, *args, **kwargs)
        watcher = self.hub.loop.async()
        watcher.start(task.spawn)
        watcher.send()
        return task


# Gevent executor

class GeventExecutor(AbstractExecutor):
    """Gevent tango executor"""

    asynchronous = True
    default_wait = True

    def __init__(self, loop=None, subexecutor=None):
        super(GeventExecutor, self).__init__()
        if loop is None:
            loop = GeventLoop()
        if subexecutor is None:
            subexecutor = get_global_threadpool()
        self.loop = loop
        self.subexecutor = subexecutor

    def delegate(self, fn, *args, **kwargs):
        """Return the given operation as a gevent future."""
        return self.subexecutor.spawn(fn, *args, **kwargs)

    def access(self, accessor, timeout=None):
        """Return a result from an gevent future."""
        return accessor.get(timeout=timeout)

    def submit(self, fn, *args, **kwargs):
        return self.loop.submit(fn, *args, **kwargs)

    def execute(self, fn, *args, **kwargs):
        """Execute an operation and return the result."""
        if self.in_executor_context():
            return fn(*args, **kwargs)
        task = self.submit(fn, *args, **kwargs)
        return task.result()
