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

"""This module exposes a futures version of :class:`tango.DeviceProxy` and
:class:`tango.AttributeProxy"""

__all__ = ("DeviceProxy", "AttributeProxy", "check_requirements")

from functools import partial

from tango import GreenMode
from tango.device_proxy import get_device_proxy
from tango.attribute_proxy import get_attribute_proxy


def check_requirements():
    try:
        import concurrent.futures  # noqa: F401
    except ImportError:
        import sys
        if sys.version_info[0] < 3:
            raise ImportError(
                "No module named concurrent. You need to "
                "install the futures backport module to have "
                "access to PyTango futures green mode")


check_requirements()

DeviceProxy = partial(get_device_proxy, green_mode=GreenMode.Futures)
DeviceProxy.__doc__ = """
    DeviceProxy(self, dev_name, wait=True, timeout=True) -> DeviceProxy
    DeviceProxy(self, dev_name, need_check_acc, wait=True, timeout=True) -> DeviceProxy

    Creates a *futures* enabled :class:`~tango.DeviceProxy`.

    The DeviceProxy constructor internally makes some network calls which makes
    it *slow*. By using the futures *green mode* you are allowing other
    python code to be executed in a cooperative way.

    .. note::
        The timeout parameter has no relation with the tango device client side
        timeout (gettable by :meth:`~tango.DeviceProxy.get_timeout_millis` and
        settable through :meth:`~tango.DeviceProxy.set_timeout_millis`)

    :param dev_name: the device name or alias
    :type dev_name: str
    :param need_check_acc: in first version of the function it defaults to True.
                           Determines if at creation time of DeviceProxy it
                           should check for channel access (rarely used)
    :type need_check_acc: bool
    :param wait: whether or not to wait for result of creating a DeviceProxy.
    :type wait: bool
    :param timeout: The number of seconds to wait for the result.
                    If None, then there is no limit on the wait time.
                    Ignored when wait is False.
    :type timeout: float
    :returns:
        if wait is True:
            :class:`~tango.DeviceProxy`
        else:
            :class:`concurrent.futures.Future`
    :throws:
        * a *DevFailed* if wait is True and there is an error creating
          the device.
        * a *concurrent.futures.TimeoutError* if wait is False, timeout is not
          None and the time to create the device has expired.

    New in PyTango 8.1.0
"""

AttributeProxy = partial(get_attribute_proxy, green_mode=GreenMode.Futures)
AttributeProxy.__doc__ = """
    AttributeProxy(self, full_attr_name, wait=True, timeout=True) -> AttributeProxy
    AttributeProxy(self, device_proxy, attr_name, wait=True, timeout=True) -> AttributeProxy

    Creates a *futures* enabled :class:`~tango.AttributeProxy`.

    The AttributeProxy constructor internally makes some network calls which
    makes it *slow*. By using the *gevent mode* you are allowing other python
    code to be executed in a cooperative way.

    :param full_attr_name: the full name of the attribute
    :type full_attr_name: str
    :param device_proxy: the :class:`~tango.DeviceProxy`
    :type device_proxy: DeviceProxy
    :param attr_name: attribute name for the given device proxy
    :type attr_name: str
    :param wait: whether or not to wait for result of creating an
                 AttributeProxy.
    :type wait: bool
    :param timeout: The number of seconds to wait for the result.
                    If None, then there is no limit on the wait time.
                    Ignored when wait is False.
    :type timeout: float
    :returns:
        if wait is True:
            :class:`~tango.AttributeProxy`
        else:
            :class:`concurrent.futures.Future`
    :throws:
        * a *DevFailed* if wait is True  and there is an error creating the
          attribute.
        * a *concurrent.futures.TimeoutError* if wait is False, timeout is not
          None and the time to create the attribute has expired.

    New in PyTango 8.1.0
"""

Device = DeviceProxy
Attribute = AttributeProxy

del GreenMode
del get_device_proxy
del get_attribute_proxy
