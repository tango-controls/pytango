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

"""
This is an internal PyTango module.
"""

__all__ = ('init',)

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
from .auto_monitor import auto_monitor_init
from .pipe import pipe_init
from ._tango import constants
from ._tango import _get_tango_lib_release

__INITIALIZED = False
__DOC = True


def init_constants():
    import sys
    import platform

    tg_ver = tuple(map(int, constants.TgLibVers.split(".")))
    tg_ver_str = "0x%02d%02d%02d00" % (tg_ver[0], tg_ver[1], tg_ver[2])
    constants.TANGO_VERSION_HEX = int(tg_ver_str, 16)

    BOOST_VERSION = ".".join(map(str, (constants.BOOST_MAJOR_VERSION,
                                       constants.BOOST_MINOR_VERSION,
                                       constants.BOOST_PATCH_VERSION)))
    constants.BOOST_VERSION = BOOST_VERSION

    class Compile(object):
        PY_VERSION = constants.PY_VERSION
        TANGO_VERSION = constants.TANGO_VERSION
        BOOST_VERSION = constants.BOOST_VERSION
        NUMPY_VERSION = constants.NUMPY_VERSION
        # UNAME = tuple(map(str, json.loads(constants.UNAME)))

    tg_rt_ver_nb = _get_tango_lib_release()
    tg_rt_major_ver = tg_rt_ver_nb // 100
    tg_rt_minor_ver = tg_rt_ver_nb // 10 % 10
    tg_rt_patch_ver = tg_rt_ver_nb % 10
    tg_rt_ver = ".".join(map(str, (tg_rt_major_ver, tg_rt_minor_ver,
                                   tg_rt_patch_ver)))

    class Runtime(object):
        PY_VERSION = ".".join(map(str, sys.version_info[:3]))
        TANGO_VERSION = tg_rt_ver
        BOOST_VERSION = '0.0.0'
        if constants.NUMPY_SUPPORT:
            import numpy
            NUMPY_VERSION = numpy.__version__
        else:
            NUMPY_VERSION = None
        UNAME = platform.uname()

    constants.Compile = Compile
    constants.Runtime = Runtime


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
    auto_monitor_init(doc=doc)
    pipe_init(doc=doc)

    # must come last: depends on device_proxy.init()
    attribute_proxy_init(doc=doc)

    __INITIALIZED = True
