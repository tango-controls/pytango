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

from __future__ import absolute_import
import sys
import types

import six

__all__ = ["get_global_threadpool", "get_global_executor",
           "get_event_loop", "submit", "spawn", "wait"]

def get_global_threadpool():
    import gevent
    return gevent.get_hub().threadpool

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
        six.reraise(result.exception, result.error_string,result.tb)
    else:
        return result

def spawn(fn, *args, **kwargs):
    # the gevent threadpool do not raise exception with asyncresults, we have to wrap it
    fn = wrap_errors(fn)
    g = get_global_threadpool().spawn(fn, *args, **kwargs)
    g._get = g.get
    g.get = types.MethodType(get_with_exception, g)
    return g

get_global_executor = get_global_threadpool

submit = spawn

__event_loop = None

def get_event_loop():
    global __event_loop
    if __event_loop is None:
        import gevent
        import gevent.queue

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
        __event_loop = gevent.spawn(loop, queue)
        __event_loop.submit = submit
    return __event_loop

def wait(greenlet, timeout=None):
    return greenlet.get(timeout=timeout)
