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

# Gevent imports
import gevent.queue

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


# Helpers

def get_threadpool():
    thread_pool = gevent.get_hub().threadpool
    # before gevent-1.1.0, patch the spawn method to propagate exception raised
    # in the loop to the AsyncResult.
    if gevent.version_info < (1, 1):
        thread_pool.submit = patched_spawn
    else:
        thread_pool.submit = thread_pool.spawn
    return thread_pool


class ExceptionWrapper:
    def __init__(self, exception, error_string, tb):
        self.exception = exception
        self.error_string = error_string
        self.tb = tb


class wrap_errors(object):
    def __init__(self, func):
        """Make a new function from `func', such that it catches all exceptions
        and return it as a specific object
        """
        self.func = func

    def __call__(self, *args, **kwargs):
        func = self.func
        try:
            return func(*args, **kwargs)
        except:
            return ExceptionWrapper(*sys.exc_info())

    def __str__(self):
        return str(self.func)

    def __repr__(self):
        return repr(self.func)

    def __getattr__(self, item):
        return getattr(self.func, item)


def get_with_exception(g, block=True, timeout=None):
    result = g._get(block, timeout)
    if isinstance(result, ExceptionWrapper):
        # raise the exception using the caller context
        six.reraise(result.exception, result.error_string, result.tb)
    else:
        return result


def patched_spawn(fn, *args, **kwargs):
    # the gevent threadpool do not raise exception with asyncresults,
    # we have to wrap it
    fn = wrap_errors(fn)
    g = get_global_threadpool().spawn(fn, *args, **kwargs)
    g._get = g.get
    g.get = types.MethodType(get_with_exception, g)
    return g

def spawn(fn, *args, **kwargs):
    return get_global_threadpool().submit(fn, *args, **kwargs)




def make_event_loop():

    def loop(queue):
        while True:
            item = queue.get()
            try:
                f, args, kwargs = item
                gevent.spawn(f, *args, **kwargs)
            except Exception as e:
                sys.excepthook(*sys.exc_info())

    def submit(fn, *args, **kwargs):
        l_async = queue.hub.loop.async()
        queue.put((fn, args, kwargs))
        l_async.send()

    queue = gevent.queue.Queue()
    event_loop = gevent.spawn(loop, queue)
    event_loop.submit = submit
    return event_loop


# Gevent executor

class GeventExecutor(AbstractExecutor):
    """Gevent tango executor"""

    asynchronous = True
    default_wait = True

    def __init__(self, loop=None, subexecutor=None):
        if loop is None:
            loop = make_event_loop()
        if subexecutor is None:
            subexecutor = get_threadpool()
        self.loop = loop
        self.subexecutor = subexecutor

    def delegate(self, fn, *args, **kwargs):
        """Return the given operation as an asyncio future."""
        return self.subexecutor.submit(fn, *args, **kwargs)

    def access(self, accessor, timeout=None):
        """Return a result from an asyncio future."""
        return accessor.get(timeout=timeout)

    def submit(self, fn, *args, **kwargs):
        return self.loop.submit(fn, *args, **kwargs)

    def execute(self, fn, *args, **kwargs):
        """Execute an operation and return the result."""
        raise RuntimeError('Not implemented yet')
