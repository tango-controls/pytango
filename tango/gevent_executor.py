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
import functools

# Gevent imports
import gevent.queue
import gevent.monkey
import gevent.threadpool

# Bypass gevent monkey patching
ThreadSafeEvent = gevent.monkey.get_original('threading', 'Event')

# Tango imports
from .green import AbstractExecutor


__all__ = ("get_global_executor", "set_global_executor", "GeventExecutor")

# Global executor

_EXECUTOR = None
_THREAD_POOL = None


def get_global_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = GeventExecutor()
    return _EXECUTOR


def set_global_executor(executor):
    global _EXECUTOR
    _EXECUTOR = executor


def get_global_threadpool():
    global _THREAD_POOL
    if _THREAD_POOL is None:
        _THREAD_POOL = ThreadPool(maxsize=10**4)
    return _THREAD_POOL


class ExceptionWrapper:
    def __init__(self, exception, error_string, tb):
        self.exception = exception
        self.error_string = error_string
        self.tb = tb


def wrap_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return ExceptionWrapper(*sys.exc_info())

    return wrapper


def unwrap_error(source):
    result = source.get()
    if isinstance(result, ExceptionWrapper):
        # Raise the exception using the caller context
        six.reraise(result.exception, result.error_string, result.tb)
    return result


class ThreadPool(gevent.threadpool.ThreadPool):

    def spawn(self, fn, *args, **kwargs):
        fn = wrap_error(fn)
        fn_result = super(ThreadPool, self).spawn(fn, *args, **kwargs)
        return gevent.spawn(unwrap_error, fn_result)


# Gevent task and event loop

class GeventTask:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.value = None
        self.exception = None
        self.done = ThreadSafeEvent()
        self.started = ThreadSafeEvent()

    def run(self):
        self.started.set()
        try:
            self.value = self.func(*self.args, **self.kwargs)
        except:
            self.exception = sys.exc_info()
        finally:
            self.done.set()

    def spawn(self):
        return gevent.spawn(self.run)

    def result(self):
        self.done.wait()
        if self.exception:
            six.reraise(*self.exception)
        return self.value


# Gevent executor

class GeventExecutor(AbstractExecutor):
    """Gevent tango executor"""

    asynchronous = True
    default_wait = True

    def __init__(self, loop=None, subexecutor=None):
        super(GeventExecutor, self).__init__()
        if loop is None:
            loop = gevent.get_hub().loop
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

    def create_watcher(self):
        try:
            return self.loop.async_()
        except AttributeError:
            return getattr(self.loop, 'async')()

    def submit(self, fn, *args, **kwargs):
        task = GeventTask(fn, *args, **kwargs)
        watcher = self.create_watcher()
        watcher.start(task.spawn)
        watcher.send()
        task.started.wait()
        # The watcher has to be stopped in order to be garbage-collected.
        # This step is crucial since the watcher holds a reference to the
        # `task.spawn` method which itself holds a reference to the task.
        # It's also important to wait for the task to be spawned before
        # stopping the watcher, otherwise the task won't run.
        watcher.stop()
        return task

    def execute(self, fn, *args, **kwargs):
        """Execute an operation and return the result."""
        if self.in_executor_context():
            return fn(*args, **kwargs)
        task = self.submit(fn, *args, **kwargs)
        return task.result()
