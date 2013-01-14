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

__all__ = ['init']

__docformat__ = "restructuredtext"

from .attribute_proxy import attribute_proxy_init
from .base_types import base_types_init
from .exception import exception_init
from .callback import callback_init
from .api_util import api_util_init
from .encoded_attribute import encoded_attribute_init
from .connection import connection_init
from .db import db_init
from .device_attribute import device_attribute_init
from .device_class import device_class_init
from .device_data import device_data_init
from .device_proxy import device_proxy_init
from .device_server import device_server_init
from .group import group_init
from .group_reply import group_reply_init
from .group_reply_list import group_reply_list_init
from .pytango_pprint import pytango_pprint_init
from .pyutil import pyutil_init
from .time_val import time_val_init
from ._PyTango import constants

__INITIALIZED = False
__DOC = True

def init_constants():
    
    tgver = tuple(map(int, constants.TgLibVers.split(".")))
    tgver_str = "0x%02d%02d%02d00" % (tgver[0], tgver[1], tgver[2])
    constants.TANGO_VERSION_HEX = int(tgver_str, 16)
    

def init():
    global __INITIALIZED
    if __INITIALIZED:
        return
    
    global __DOC
    doc = __DOC
    init_constants()
    base_types_init(doc=doc)
    exception_init(doc=doc)
    callback_init(doc=doc)
    api_util_init(doc=doc)
    encoded_attribute_init(doc=doc)
    connection_init(doc=doc)
    db_init(doc=doc)
    device_attribute_init(doc=doc)
    device_class_init(doc=doc)
    device_data_init(doc=doc)
    device_proxy_init(doc=doc)
    device_server_init(doc=doc)
    group_init(doc=doc)
    group_reply_init(doc=doc)
    group_reply_list_init(doc=doc)
    pytango_pprint_init(doc=doc)
    pyutil_init(doc=doc)
    time_val_init(doc=doc)
    
    # must come last: depends on device_proxy.init()
    attribute_proxy_init(doc=doc)

    __INITIALIZED = True
