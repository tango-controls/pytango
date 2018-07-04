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
This is an internal PyTango module. It completes the binding of
:class:`tango.AttributeProxy`.

To access these members use directly :mod:`tango` module and NOT
tango.attribute_proxy.
"""

import collections

from ._tango import StdStringVector, DbData, DbDatum, DeviceProxy
from ._tango import __AttributeProxy as _AttributeProxy
from .utils import seq_2_StdStringVector, seq_2_DbData, DbData_2_dict
from .utils import is_pure_str, is_non_str_seq
from .green import green, get_green_mode
from .device_proxy import __init_device_proxy_internals as init_device_proxy

__all__ = ("AttributeProxy", "attribute_proxy_init", "get_attribute_proxy")


@green(consume_green_mode=False)
def get_attribute_proxy(*args, **kwargs):
    """
    get_attribute_proxy(self, full_attr_name, green_mode=None, wait=True, timeout=True) -> AttributeProxy
    get_attribute_proxy(self, device_proxy, attr_name, green_mode=None, wait=True, timeout=True) -> AttributeProxy

    Returns a new :class:`~tango.AttributeProxy`.
    There is no difference between using this function and the direct
    :class:`~tango.AttributeProxy` constructor if you use the default kwargs.

    The added value of this function becomes evident when you choose a green_mode
    to be *Futures* or *Gevent*. The AttributeProxy constructor internally makes some
    network calls which makes it *slow*. By using one of the *green modes* as
    green_mode you are allowing other python code to be executed in a cooperative way.

    :param full_attr_name: the full name of the attribute
    :type full_attr_name: str
    :param device_proxy: the :class:`~tango.DeviceProxy`
    :type device_proxy: DeviceProxy
    :param attr_name: attribute name for the given device proxy
    :type attr_name: str
    :param green_mode: determines the mode of execution of the device (including
                      the way it is created). Defaults to the current global
                      green_mode (check :func:`~tango.get_green_mode` and
                      :func:`~tango.set_green_mode`)
    :type green_mode: :obj:`~tango.GreenMode`
    :param wait: whether or not to wait for result. If green_mode
                 Ignored when green_mode is Synchronous (always waits).
    :type wait: bool
    :param timeout: The number of seconds to wait for the result.
                    If None, then there is no limit on the wait time.
                    Ignored when green_mode is Synchronous or wait is False.
    :type timeout: float
    :returns:
        if green_mode is Synchronous or wait is True:
            :class:`~tango.AttributeProxy`
        else if green_mode is Futures:
            :class:`concurrent.futures.Future`
        else if green_mode is Gevent:
            :class:`gevent.event.AsynchResult`
    :throws:
        * a *DevFailed* if green_mode is Synchronous or wait is True
          and there is an error creating the attribute.
        * a *concurrent.futures.TimeoutError* if green_mode is Futures,
          wait is False, timeout is not None and the time to create the attribute
          has expired.
        * a *gevent.timeout.Timeout* if green_mode is Gevent, wait is False,
          timeout is not None and the time to create the attribute has expired.

    New in PyTango 8.1.0
    """
    return AttributeProxy(*args, **kwargs)


def __AttributeProxy__get_property(self, propname, value=None):
    """
    get_property(self, propname, value) -> DbData

            Get a (list) property(ies) for an attribute.

            This method accepts the following types as propname parameter:
            1. string [in] - single property data to be fetched
            2. sequence<string> [in] - several property data to be fetched
            3. tango.DbDatum [in] - single property data to be fetched
            4. tango.DbData [in,out] - several property data to be fetched.
            5. sequence<DbDatum> - several property data to be feteched

            Note: for cases 3, 4 and 5 the 'value' parameter if given, is IGNORED.

            If value is given it must be a tango.DbData that will be filled with the
            property values

        Parameters :
            - propname : (str) property(ies) name(s)
            - value : (tango.DbData) (optional, default is None meaning that the
                      method will create internally a tango.DbData and return
                      it filled with the property values

        Return     : (DbData) containing the property(ies) value(s). If a
                     tango.DbData is given as parameter, it returns the same
                     object otherwise a new tango.DbData is returned

        Throws     : NonDbDevice, ConnectionFailed (with database),
                     CommunicationFailed (with database),
                     DevFailed from database device
    """

    if is_pure_str(propname) or isinstance(propname, StdStringVector):
        new_value = value
        if new_value is None:
            new_value = DbData()
        self._get_property(propname, new_value)
        return DbData_2_dict(new_value)
    elif isinstance(propname, DbDatum):
        new_value = DbData()
        new_value.append(propname)
        self._get_property(new_value)
        return DbData_2_dict(new_value)
    elif isinstance(propname, collections.Sequence):
        if isinstance(propname, DbData):
            self._get_property(propname)
            return DbData_2_dict(propname)

        if is_pure_str(propname[0]):
            new_propname = StdStringVector()
            for i in propname:
                new_propname.append(i)
            new_value = value
            if new_value is None:
                new_value = DbData()
            self._get_property(new_propname, new_value)
            return DbData_2_dict(new_value)
        elif isinstance(propname[0], DbDatum):
            new_value = DbData()
            for i in propname:
                new_value.append(i)
            self._get_property(new_value)
            return DbData_2_dict(new_value)


def __AttributeProxy__put_property(self, value):
    """
    put_property(self, value) -> None

            Insert or update a list of properties for this attribute.
            This method accepts the following types as value parameter:
            1. tango.DbDatum - single property data to be inserted
            2. tango.DbData - several property data to be inserted
            3. sequence<DbDatum> - several property data to be inserted
            4. dict<str, DbDatum> - keys are property names and value has data to be inserted
            5. dict<str, seq<str>> - keys are property names and value has data to be inserted
            6. dict<str, obj> - keys are property names and str(obj) is property value

        Parameters :
            - value : can be one of the following:
                1. tango.DbDatum - single property data to be inserted
                2. tango.DbData - several property data to be inserted
                3. sequence<DbDatum> - several property data to be inserted
                4. dict<str, DbDatum> - keys are property names and value has data to be inserted
                5. dict<str, seq<str>> - keys are property names and value has data to be inserted
                6. dict<str, obj> - keys are property names and str(obj) is property value

        Return     : None

        Throws     : ConnectionFailed, CommunicationFailed
                     DevFailed from device (DB_SQLError)
    """
    if isinstance(value, DbData):
        pass
    elif isinstance(value, DbDatum):
        new_value = DbData()
        new_value.append(value)
        value = new_value
    elif is_non_str_seq(value):
        new_value = seq_2_DbData(value)
    elif isinstance(value, collections.Mapping):
        new_value = DbData()
        for k, v in value.items():
            if isinstance(v, DbDatum):
                new_value.append(v)
                continue
            db_datum = DbDatum(k)
            if is_non_str_seq(v):
                seq_2_StdStringVector(v, db_datum.value_string)
            else:
                db_datum.value_string.append(str(v))
            new_value.append(db_datum)
        value = new_value
    else:
        raise TypeError(
            'Value must be a tango.DbDatum, tango.DbData, '
            'a sequence<DbDatum> or a dictionary')
    return self._put_property(value)


def __AttributeProxy__delete_property(self, value):
    """
    delete_property(self, value) -> None

        Delete a the given of properties for this attribute.
        This method accepts the following types as value parameter:

            1. string [in] - single property to be deleted
            2. tango.DbDatum [in] - single property data to be deleted
            3. tango.DbData [in] - several property data to be deleted
            4. sequence<string> [in]- several property data to be deleted
            5. sequence<DbDatum> [in] - several property data to be deleted
            6. dict<str, obj> [in] - keys are property names to be deleted
               (values are ignored)
            7. dict<str, DbDatum> [in] - several DbDatum.name are property names
               to be deleted (keys are ignored)

        Parameters :
            - value : can be one of the following:

                1. string [in] - single property data to be deleted
                2. tango.DbDatum [in] - single property data to be deleted
                3. tango.DbData [in] - several property data to be deleted
                4. sequence<string> [in]- several property data to be deleted
                5. sequence<DbDatum> [in] - several property data to be deleted
                6. dict<str, obj> [in] - keys are property names to be deleted
                   (values are ignored)
                7. dict<str, DbDatum> [in] - several DbDatum.name are property
                   names to be deleted (keys are ignored)

        Return     : None

        Throws     : ConnectionFailed, CommunicationFailed
                    DevFailed from device (DB_SQLError)
    """
    if isinstance(value, DbData) or isinstance(value, StdStringVector) or \
            is_pure_str(value):
        new_value = value
    elif isinstance(value, DbDatum):
        new_value = DbData()
        new_value.append(value)
    elif isinstance(value, collections.Sequence):
        new_value = DbData()
        for e in value:
            if isinstance(e, DbDatum):
                new_value.append(e)
            else:
                new_value.append(DbDatum(str(e)))
    elif isinstance(value, collections.Mapping):
        new_value = DbData()
        for k, v in value.items():
            if isinstance(v, DbDatum):
                new_value.append(v)
            else:
                new_value.append(DbDatum(k))
    else:
        raise TypeError(
            'Value must be a string, tango.DbDatum, '
            'tango.DbData, a sequence or a dictionary')

    return self._delete_property(new_value)


# It is easier to reimplement AttributeProxy in python using DeviceProxy than
# wrapping C++ AttributeProxy. However I still rely in the original
# AttributeProxy for the constructor (parsing strings if necessary) and some
# other things. With the _method_* functions defined later it is really easy.
# One reason to do it this way: get_device_proxy() will always return the
# same tango.DeviceProxy with this implementation. And then we can trust
# it's automatic event unsubscription to handle events.
class AttributeProxy(object):
    """
        AttributeProxy is the high level Tango object which provides the
        client with an easy-to-use interface to TANGO attributes.

        To create an AttributeProxy, a complete attribute name must be set
        in the object constructor.

        Example:
            att = AttributeProxy("tango/tangotest/1/long_scalar")

        Note: PyTango implementation of AttributeProxy is in part a
        python reimplementation of the AttributeProxy found on the C++ API.
    """

    def __init__(self, *args, **kwds):
        green_mode = kwds.pop('green_mode', get_green_mode())
        self.__attr_proxy = _AttributeProxy(*args, **kwds)
        # get_device_proxy() returns a different python object each time
        # we don't want a different object, so we save the current one.
        self.__dev_proxy = dp = self.__attr_proxy.get_device_proxy()
        init_device_proxy(dp)
        dp.set_green_mode(green_mode)

    def get_device_proxy(self):
        """
        get_device_proxy(self) -> DeviceProxy

                A method which returns the device associated to the attribute

            Parameters : None

            Return     : (DeviceProxy)
        """
        return self.__dev_proxy

    def name(self):
        """
        name(self) -> str

                Returns the attribute name

            Parameters : None
            Return     : (str) with the attribute name
        """
        return self.__attr_proxy.name()

    def __str__(self):
        return "AttributeProxy(%s)" % self.name()

    def __repr__(self):
        return "AttributeProxy(%s)" % self.name()


def _method_dev_and_name(dp_fn_name, doc=True):
    def __new_fn(self, *args, **kwds):
        return getattr(self._AttributeProxy__dev_proxy, dp_fn_name)(self.name(), *args, **kwds)

    if doc:
        __new_fn.__doc__ = "This method is a simple way to do:\n" + \
                           "\tself.get_device_proxy()." + dp_fn_name + \
                           "(self.name(), ...)\n\n" + \
                           "For convenience, here is the documentation of DeviceProxy." + \
                           dp_fn_name + "(...):\n" + \
                           str(getattr(DeviceProxy, dp_fn_name).__doc__)
    __new_fn.__name__ = dp_fn_name
    return __new_fn


def _method_device(dp_fn_name, doc=True):
    def __new_fn(self, *args, **kwds):
        return getattr(self._AttributeProxy__dev_proxy, dp_fn_name)(*args, **kwds)

    if doc:
        __new_fn.__doc__ = "This method is a simple way to do:\n" + \
                           "\tself.get_device_proxy()." + dp_fn_name + \
                           "(...)\n\n" + \
                           "For convenience, here is the documentation of DeviceProxy." + \
                           dp_fn_name + "(...):\n" + \
                           str(getattr(DeviceProxy, dp_fn_name).__doc__)
    __new_fn.__name__ = dp_fn_name
    return __new_fn


def _method_attribute(dp_fn_name, doc=True):
    def __new_fn(self, *args, **kwds):
        return getattr(self._AttributeProxy__attr_proxy, dp_fn_name)(*args, **kwds)

    if doc:
        __new_fn.__doc__ = getattr(_AttributeProxy, dp_fn_name).__doc__
    __new_fn.__name__ = dp_fn_name
    return __new_fn


def __init_AttributeProxy(doc=True):
    _AttributeProxy.get_property = __AttributeProxy__get_property
    _AttributeProxy.put_property = __AttributeProxy__put_property
    _AttributeProxy.delete_property = __AttributeProxy__delete_property

    # General methods
    # AttributeProxy.name                manually defined
    AttributeProxy.status = _method_device('status', doc=doc)
    AttributeProxy.state = _method_device('state', doc=doc)
    AttributeProxy.ping = _method_device('ping', doc=doc)
    AttributeProxy.get_transparency_reconnection = _method_device('get_transparency_reconnection', doc=doc)
    AttributeProxy.set_transparency_reconnection = _method_device('set_transparency_reconnection', doc=doc)

    # Property methods
    AttributeProxy.get_property = _method_attribute('get_property', doc=doc)
    AttributeProxy.put_property = _method_attribute('put_property', doc=doc)
    AttributeProxy.delete_property = _method_attribute('delete_property', doc=doc)

    # Attribute methods
    AttributeProxy.get_config = _method_dev_and_name('get_attribute_config', doc=doc)
    AttributeProxy.set_config = _method_device('set_attribute_config', doc=doc)

    AttributeProxy.write = _method_dev_and_name('write_attribute', doc=doc)
    AttributeProxy.read = _method_dev_and_name('read_attribute', doc=doc)
    AttributeProxy.write_read = _method_dev_and_name('write_read_attribute', doc=doc)

    # History methods...
    AttributeProxy.history = _method_dev_and_name('attribute_history', doc=doc)

    # Polling administration methods
    AttributeProxy.poll = _method_dev_and_name('poll_attribute', doc=doc)
    AttributeProxy.get_poll_period = _method_dev_and_name('get_attribute_poll_period', doc=doc)
    AttributeProxy.is_polled = _method_dev_and_name('is_attribute_polled', doc=doc)
    AttributeProxy.stop_poll = _method_dev_and_name('stop_poll_attribute', doc=doc)

    # Asynchronous methods
    AttributeProxy.read_asynch = _method_dev_and_name('read_attribute_asynch', doc=doc)
    AttributeProxy.read_reply = _method_device('read_attribute_reply', doc=doc)
    AttributeProxy.write_asynch = _method_device('write_attribute_asynch', doc=doc)
    AttributeProxy.write_reply = _method_device('write_attribute_reply', doc=doc)

    # Event methods
    AttributeProxy.subscribe_event = _method_dev_and_name('subscribe_event', doc=doc)
    AttributeProxy.unsubscribe_event = _method_device('unsubscribe_event', doc=doc)

    AttributeProxy.get_events = _method_device('get_events', doc=doc)
    AttributeProxy.event_queue_size = _method_device('event_queue_size', doc=doc)
    AttributeProxy.get_last_event_date = _method_device('get_last_event_date', doc=doc)
    AttributeProxy.is_event_queue_empty = _method_device('is_event_queue_empty', doc=doc)


def attribute_proxy_init(doc=True):
    __init_AttributeProxy(doc=doc)
