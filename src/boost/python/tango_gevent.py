# ------------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

from __future__ import absolute_import

__all__ = ["get_global_threadpool", "get_global_executor", "submit", "spawn"]


def get_global_threadpool():
    import gevent
    return gevent.get_hub().threadpool


def spawn(fn, *args, **kwargs):
    return get_global_threadpool().spawn(fn, *args, **kwargs)


get_global_executor = get_global_threadpool

submit = spawn
