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

__all__ = ["get_global_executor", "submit", "spawn", "wait"]

__global_executor = None

MAX_WORKERS = 8
MODE = 'thread'


def __get_executor_class():
    import concurrent.futures
    ret = None
    if MODE == 'thread':
        ret = concurrent.futures.ThreadPoolExecutor
    else:
        ret = concurrent.futures.ProcessPoolExecutor
    return ret


def get_global_executor():
    global __global_executor
    if __global_executor is None:
        klass = __get_executor_class()
        if klass is not None:
            __global_executor = klass(max_workers=MAX_WORKERS)
    return __global_executor


def submit(fn, *args, **kwargs):
    return get_global_executor().submit(fn, *args, **kwargs)


def wait(fut, timeout=None):
    return fut.result(timeout=timeout)


spawn = submit
