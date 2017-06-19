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
High Level API for writting Tango clients

This is an experimental module. Not part of the official API.
"""

import weakref
import functools

import six

import tango
from tango import DeviceProxy as Device
from tango import CmdArgType
from tango.codec import loads
from tango.codec import dumps as _dumps

_FMT = "pickle"

dumps = functools.partial(_dumps, _FMT)


def _command(device, cmd_info, *args, **kwargs):
    name = cmd_info.cmd_name
    if cmd_info.in_type == CmdArgType.DevEncoded:
        result = device.command_inout(name, dumps((args, kwargs)))
    else:
        result = device.command_inout(name, *args, **kwargs)
    if cmd_info.out_type == CmdArgType.DevEncoded:
        result = loads(*result)
    return result


class _DeviceHelper(object):
    __CMD_FILTER = set(("init", "state", "status"))
    __ATTR_FILTER = set(("state", "status"))

    __attr_cache = None
    __cmd_cache = None

    def __init__(self, dev_name, *args, **kwargs):
        self.dev_name = dev_name
        self.device = Device(dev_name, *args, **kwargs)
        self.slots = weakref.WeakKeyDictionary()

    def connect(self, signal, slot, event_type=tango.EventType.CHANGE_EVENT):
        i = self.device.subscribe_event(signal, event_type, slot)
        self.slots[slot] = i
        return i

    def disconnect(self, signal, slot):
        i = self.slots.pop(slot)
        self.device.unsubscribe_event(i)

    def get_attr_cache(self, refresh=False):
        cache = self.__attr_cache
        if not cache:
            refresh = True
        if refresh:
            cache = {}
            dev = self.device
            try:
                for attr_info in dev.attribute_list_query_ex():
                    attr_name = attr_info.name
                    if attr_name.lower() in self.__ATTR_FILTER:
                        continue
                    cache[attr_name] = attr_info
            except tango.DevFailed:
                pass
            self.__attr_cache = cache
        return cache

    def get_attr_info(self, name):
        cache = self.get_attr_cache()
        result = cache.get(name)
        if result:
            return result
        else:
            cache = self.get_attr_cache(refresh=True)
            return cache.get(name)

    def get_cmd_cache(self, refresh=False):
        cache = self.__cmd_cache
        if not cache:
            refresh = True
        if refresh:
            cache = {}
            dev = self.device
            try:
                for cmd_info in dev.command_list_query():
                    cmd_name = cmd_info.cmd_name
                    if cmd_name.lower() in self.__CMD_FILTER:
                        continue
                    cmd_func = functools.partial(_command, dev, cmd_info)
                    cmd_func.__name__ = cmd_name
                    cmd_func.__doc__ = cmd_info.in_type_desc
                    cmd_info.func = cmd_func
                    cache[cmd_name] = cmd_info
            except tango.DevFailed:
                pass
            self.__cmd_cache = cache
        return cache

    def get_cmd_info(self, name):
        cache = self.get_cmd_cache()
        result = cache.get(name)
        if result:
            return result
        else:
            cache = self.get_cmd_cache(refresh=True)
            return cache.get(name)

    def is_cmd(self, name):
        return name.lower() in self.get_cmd_cache()

    def members(self):
        result = self.get_attr_cache().keys()
        result.extend(self.get_cmd_cache().keys())
        return result

    def get(self, name):
        dev = self.device
        result = self.get_attr_info(name)
        if result:
            result = dev.read_attribute(name)
            value = result.value
            if result.type == tango.DevEncoded:
                result = loads(*value)
            else:
                result = value
            return result
        result = self.get_cmd_info(name)
        if result is None:
            raise KeyError("Unknown %s" % name)
        return result

    def set(self, name, value):
        result = self.get_attr_info(name)
        if result is None:
            raise KeyError("Unknown attribute %s" % name)
        if result.data_type == tango.DevEncoded:
            self.device.write_attribute(name, dumps(value))
        else:
            self.device.write_attribute(name, value)

    def get_info(self):
        try:
            return self.__info
        except AttributeError:
            pass
        try:
            info = self.device.info()
            self.__dict__["__info"] = info
            return info
        except tango.DevFailed:
            return None

    def __getitem__(self, name):
        if self.get_attr_info(name) is None:
            raise KeyError("Unknown attribute %s" % name)
        return self.device[name]

    def __setitem__(self, name, value):
        if self.get_attr_info(name) is None:
            raise KeyError("Unknown attribute %s" % name)
        self.device[name] = value

    def __str__(self):
        return self.dstr()

    def __repr__(self):
        return str(self)

    def dstr(self):
        info = self.get_info()
        klass = "Device"
        if info:
            klass = info.dev_class
        return "{0}({1})".format(klass, self.dev_name)


class Object(object):
    """Tango object"""

    def __init__(self, dev_name, *args, **kwargs):
        helper = _DeviceHelper(dev_name, *args, **kwargs)
        self.__dict__["_helper"] = helper

    def __getattr__(self, name):
        try:
            r = self._helper.get(name)
        except KeyError as ke:
            six.raise_from(AttributeError('Unknown {0}'.format(name)), ke)
        if isinstance(r, tango.CommandInfo):
            self.__dict__[name] = r.func
            return r.func
        return r

    def __setattr__(self, name, value):
        try:
            return self._helper.set(name, value)
        except KeyError as ke:
            six.raise_from(AttributeError('Unknown {0}'.format(name)), ke)

    def __getitem__(self, name):
        return self._helper[name]

    def __setitem__(self, name, value):
        self._helper[name] = value

    def __str__(self):
        return str(self._helper)

    def __repr__(self):
        return repr(self._helper)

    def __dir__(self):
        return self._helper.members()


def get_object_proxy(obj):
    """Experimental function. Not part of the official API"""
    return obj._helper.device


def get_object_db(obj):
    """Experimental function. Not part of the official API"""
    return get_object_proxy(obj).get_device_db()


def get_object_name(obj):
    """Experimental function. Not part of the official API"""
    return get_object_proxy(obj).get_name()


def get_object_info(obj):
    """Experimental function. Not part of the official API"""
    return get_object_proxy(obj).info()


def get_attributes_config(obj, refresh=False):
    """Experimental function. Not part of the official API"""
    return obj._helper.get_attr_cache(refresh=refresh)


def get_commands_config(obj, refresh=False):
    """Experimental function. Not part of the official API"""
    return obj._helper.get_cmd_cache(refresh=refresh)


def connect(obj, signal, slot, event_type=tango.EventType.CHANGE_EVENT):
    """Experimental function. Not part of the official API"""
    return obj._helper.connect(signal, slot, event_type=event_type)


def disconnect(obj, signal, slot):
    """Experimental function. Not part of the official API"""
    return obj._helper.disconnect(signal, slot)
