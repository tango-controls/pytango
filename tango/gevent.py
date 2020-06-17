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

"""This module exposes a gevent version of :class:`tango.DeviceProxy` and
:class:`tango.AttributeProxy"""

from __future__ import absolute_import
from functools import partial

from ._tango import GreenMode
from .device_proxy import get_device_proxy
from .attribute_proxy import get_attribute_proxy

__all__ = ("DeviceProxy", "AttributeProxy", "check_requirements")


def check_requirements():
    try:
        import gevent
    except ImportError:
        raise ImportError("No module named gevent. You need to install "
                          "gevent module to have access to PyTango gevent "
                          "green mode. Consider using the futures green mode "
                          "instead")

    import distutils.version
    gevent_version = ".".join(map(str, gevent.version_info[:3]))
    if distutils.version.StrictVersion(gevent_version) < "1.0":
        raise ImportError("You need gevent >= 1.0. You are using %s. "
                          "Consider using the futures green mode instead"
                          % gevent_version)


check_requirements()

DeviceProxy = partial(get_device_proxy, green_mode=GreenMode.Gevent)
DeviceProxy.__doc__ = """
    DeviceProxy(self, dev_name, wait=True, timeout=True) -> DeviceProxy
    DeviceProxy(self, dev_name, need_check_acc, wait=True, timeout=True) -> DeviceProxy

    Creates a *gevent* enabled :class:`~tango.DeviceProxy`.

    The DeviceProxy constructor internally makes some network calls which makes
    it *slow*. By using the gevent *green mode* you are allowing other python
    code to be executed in a cooperative way.

    .. note::
        The timeout parameter has no relation with the tango device client side
        timeout (gettable by :meth:`~tango.DeviceProxy.get_timeout_millis` and
        settable through :meth:`~tango.DeviceProxy.set_timeout_millis`)

    :param dev_name: the device name or alias
    :type dev_name: str
    :param need_check_acc: in first version of the function it defaults to True.
                           Determines if at creation time of DeviceProxy it should check
                           for channel access (rarely used)
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
            :class:`gevent.event.AsynchResult`
    :throws:
        * a *DevFailed* if wait is True and there is an error creating
          the device.
        * a *gevent.timeout.Timeout* if wait is False, timeout is not None and
          the time to create the device has expired.

    New in PyTango 8.1.0
"""

AttributeProxy = partial(get_attribute_proxy, green_mode=GreenMode.Gevent)
AttributeProxy.__doc__ = """
    AttributeProxy(self, full_attr_name, wait=True, timeout=True) -> AttributeProxy
    AttributeProxy(self, device_proxy, attr_name, wait=True, timeout=True) -> AttributeProxy

    Creates a *gevent* enabled :class:`~tango.AttributeProxy`.

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
            :class:`gevent.event.AsynchResult`
    :throws:
        * a *DevFailed* if wait is True  and there is an error creating the
          attribute.
        * a *gevent.timeout.Timeout* if wait is False, timeout is not None
          and the time to create the attribute has expired.

    New in PyTango 8.1.0
"""

Device = DeviceProxy
Attribute = AttributeProxy

del GreenMode
del get_device_proxy
del get_attribute_proxy
