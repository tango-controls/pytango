################################################################################
##
## This file is part of PyTango, a python binding for Tango
## 
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
## 
## PyTango is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## PyTango is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################

__all__ = ["uses_future", "get_global_executor", "submit", "spawn"] 

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


spawn = submit
