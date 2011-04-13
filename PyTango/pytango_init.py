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

"""
This is an internal PyTango module.
"""

__all__ = []

__docformat__ = "restructuredtext"

import attribute_proxy
import base_types
import exception
import callback
import api_util
import encoded_attribute
import connection
import db
import device_attribute
import device_class
import device_data
import device_proxy
import device_server
import group
import group_reply
import group_reply_list
import pytango_pprint
import pyutil
import time_val

__INITIALIZED = False
__DOC = True

def __init():
    global __INITIALIZED
    if __INITIALIZED:
        return
    
    global __DOC
    doc = __DOC
    base_types.init(doc=doc)
    exception.init(doc=doc)
    callback.init(doc=doc)
    api_util.init(doc=doc)
    encoded_attribute.init(doc=doc)
    connection.init(doc=doc)
    db.init(doc=doc)
    device_attribute.init(doc=doc)
    device_class.init(doc=doc)
    device_data.init(doc=doc)
    device_proxy.init(doc=doc)
    device_server.init(doc=doc)
    group.init(doc=doc)
    group_reply.init(doc=doc)
    group_reply_list.init(doc=doc)
    pytango_pprint.init(doc=doc)
    pyutil.init(doc=doc)
    time_val.init(doc=doc)
    
    # must come last: depends on device_proxy.init()
    attribute_proxy.init(doc=doc)

    __INITIALIZED = True
    
__init()