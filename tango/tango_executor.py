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

"""Abstract and synchronous executor definition."""

__all__ = ['AbstractExecutor', 'SynchronousExecutor',
           'get_synchronous_executor']


class AbstractExecutor(object):

    asynchronous = NotImplemented
    default_wait = NotImplemented

    def delegate(self, fn, *args, **kwargs):
        """Delegate an operation and return an accessor."""
        if self.asynchronous:
            raise ValueError('Not supported in synchronous mode')
        raise NotImplementedError

    def access(self, accessor, timeout=None):
        """Return a result from an accessor."""
        if self.asynchronous:
            raise ValueError('Not supported in synchronous mode')
        raise NotImplementedError

    def submit(self, fn, *args, **kwargs):
        """Submit an operation"""
        raise NotImplementedError

    def execute(self, fn, *args, **kwargs):
        """Execute an operation and return the result."""
        raise NotImplementedError

    def run(self, fn, args, kwargs, wait=None, timeout=None):
        if wait is None:
            wait = self.default_wait
        # Sychronous (no delegation)
        if not self.asynchronous:
            if not wait or timeout:
                raise ValueError('Not supported in synchronous mode')
            return fn(*args, **kwargs)
        # Asynchronous delegation
        accessor = self.delegate(fn, *args, **kwargs)
        if not wait:
            return accessor
        return self.access(accessor, timeout=timeout)


class SynchronousExecutor(AbstractExecutor):

    asynchronous = False
    default_wait = True

    def execute(self, fn, *args, **kwargs):
        """Execute an operation and return the result."""
        return fn(*args, **kwargs)

    submit = execute


# Default synchronous executor

def get_synchronous_executor():
    return _SYNCHRONOUS_EXECUTOR

_SYNCHRONOUS_EXECUTOR = SynchronousExecutor()
