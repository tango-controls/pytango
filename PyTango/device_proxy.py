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

from __future__ import with_statement

__all__ = ["device_proxy_init"]

__docformat__ = "restructuredtext"

import threading

from ._PyTango import StdStringVector, DbData, DbDatum, AttributeInfo, \
    AttributeInfoEx, AttributeInfoList, AttributeInfoListEx, DeviceProxy, \
    __CallBackAutoDie, __CallBackPushEvent, EventType, DevFailed, Except, \
    ExtractAs
from .utils import is_pure_str, is_non_str_seq, is_integer, \
    seq_2_StdStringVector, StdStringVector_2_seq, seq_2_DbData, DbData_2_dict
from .utils import document_method as __document_method
import collections
import numbers

class __TangoInfo(object):
    """Helper class for when DeviceProxy.info() is not available"""
    
    def __init__(self):
        self.dev_class = self.dev_type = 'Device'
        self.doc_url = 'http://www.esrf.fr/computing/cs/tango/tango_doc/ds_doc/'
        self.server_host = 'Unknown'
        self.server_id = 'Unknown'
        self.server_version = 1

#-------------------------------------------------------------------------------
# Pythonic API: transform tango commands into methods and tango attributes into
# class members
#-------------------------------------------------------------------------------

def __check_read_attribute(dev_attr):
    if dev_attr.has_failed:
        raise DevFailed(*dev_attr.get_err_stack())
    return dev_attr

def __DeviceProxy__refresh_cmd_cache(self):
    cmd_list = self.command_list_query()
    cmd_cache = {}
    for cmd in cmd_list:
        n = cmd.cmd_name.lower()
        cmd_cache[n] = cmd, None
    self.__dict__['__cmd_cache'] = cmd_cache

def __DeviceProxy__refresh_attr_cache(self):
    attr_cache = [attr_name.lower() for attr_name in self.get_attribute_list()]
    self.__dict__['__attr_cache'] = attr_cache

def __DeviceProxy__getattr(self, name):
    # trait_names is a feature of IPython. Hopefully they will solve
    # ticket http://ipython.scipy.org/ipython/ipython/ticket/229 someday
    # and the ugly trait_names could be removed.
    if name[:2] == "__" or name == 'trait_names':
        raise AttributeError(name)
    
    name_l = name.lower()
    cmd_info = None
    if not hasattr(self, '__cmd_cache'):
        try:
            self.__refresh_cmd_cache()
        except:
            pass
    try:
        cmd_info = self.__cmd_cache[name_l]
    except:
        pass
        
    if cmd_info is not None:
        d, f = cmd_info
        if f is None:
            doc =  "%s(%s) -> %s\n\n" % (d.cmd_name, d.in_type, d.out_type)
            doc += " -  in (%s): %s\n" % (d.in_type, d.in_type_desc)
            doc += " - out (%s): %s\n" % (d.out_type, d.out_type_desc)
            def f(*args,**kwds): return self.command_inout(name, *args, **kwds)
            f.__doc__ = doc
            self.__cmd_cache[name_l] = d, f
        return f
    
    find_attr = True
    if not hasattr(self, '__attr_cache') or name_l not in self.__attr_cache:
        try:
            self.__refresh_attr_cache()
        except:
            find_attr = False
    
    if not find_attr or name_l not in self.__attr_cache:
        raise AttributeError(name)
    
    return self.read_attribute(name).value

def __DeviceProxy__setattr(self, name, value):
    try:
        if not hasattr(self, '__attr_cache') or name.lower() not in self.__attr_cache:
            self.__refresh_attr_cache()
    except:
        self.__dict__[name] = value
        return
        
    if name.lower() in self.__attr_cache:
        self.write_attribute(name, value)
    else:
        self.__dict__[name] = value


def __DeviceProxy__getAttributeNames(self):
    """Return list of magic attributes to extend introspection."""
    try:
        lst = [cmd.cmd_name for cmd in self.command_list_query()]
        lst += self.get_attribute_list()
        lst += list(map(str.lower, lst))
        lst.sort()
        return lst
    except Exception:
        pass
    return []

def __DeviceProxy__getitem(self, key):
    return self.read_attribute(key)

def __DeviceProxy__setitem(self, key, value):
    return self.write_attribute(key, value)

def __DeviceProxy__contains(self, key):
    return key.lower() in map(str.lower, self.get_attribute_list())

def __DeviceProxy__read_attribute(self, value, extract_as=ExtractAs.Numpy):
    return __check_read_attribute(self._read_attribute(value, extract_as))

def __DeviceProxy__read_attributes_asynch(self, attr_names, cb=None, extract_as=ExtractAs.Numpy):
    """
    read_attributes_asynch( self, attr_names) -> int

            Read asynchronously (polling model) the list of specified attributes.

        Parameters :
                - attr_names : (sequence<str>) A list of attributes to read.
                            It should be a StdStringVector or a sequence of str.
        Return     : an asynchronous call identifier which is needed to get
                        attributes value.

        Throws     : ConnectionFailed

        New in PyTango 7.0.0

    read_attributes_asynch( self, attr_names, callback, extract_as=Numpy) -> None

            Read asynchronously (callback model) an attribute list.

        Parameters :
                - attr_names : (sequence<str>) A list of attributes to read. See read_attributes.
                - callback   : (callable) This callback object should be an instance of a
                            user class with an attr_read() method. It can also
                            be any callable object.
                - extract_as : (ExtractAs) Defaults to numpy.
        Return     : None

        Throws     : ConnectionFailed

        New in PyTango 7.0.0
    """
    if cb is None:
        return self.__read_attributes_asynch(attr_names)

    cb2 = __CallBackAutoDie()
    if isinstance(cb, collections.Callable):
        cb2.attr_read = cb
    else:
        cb2.attr_read = cb.attr_read
    return self.__read_attributes_asynch(attr_names, cb2, extract_as)

def __DeviceProxy__read_attribute_asynch(self, attr_name, cb=None):
    """
    read_attribute_asynch( self, attr_name) -> int
    read_attribute_asynch( self, attr_name, callback) -> None

            Shortcut to self.read_attributes_asynch([attr_name], cb)

        New in PyTango 7.0.0
    """
    return self.read_attributes_asynch([attr_name], cb)

def __DeviceProxy__read_attribute_reply(self, *args, **kwds):
    """
    read_attribute_reply( self, id, extract_as) -> int
    read_attribute_reply( self, id, timeout, extract_as) -> None

            Shortcut to self.read_attributes_reply()[0]
            
        New in PyTango 7.0.0
    """
    return __check_read_attribute(self.read_attributes_reply(*args, **kwds)[0])

def __DeviceProxy__write_attributes_asynch(self, attr_values, cb=None):
    """
    write_attributes_asynch( self, values) -> int
    
            Write asynchronously (polling model) the specified attributes.
            
        Parameters :
                - values : (any) See write_attributes.
        Return     : An asynchronous call identifier which is needed to get the
                    server reply

        Throws     : ConnectionFailed
        
        New in PyTango 7.0.0

    write_attributes_asynch( self, values, callback) -> None
    
            Write asynchronously (callback model) a single attribute.
        
        Parameters :
                - values   : (any) See write_attributes.
                - callback : (callable) This callback object should be an instance of a user
                            class with an attr_written() method . It can also be any
                            callable object.
        Return     : None

        Throws     : ConnectionFailed
        
        New in PyTango 7.0.0
    """
    if cb is None:
        return self.__write_attributes_asynch(attr_values)

    cb2 = __CallBackAutoDie()
    if isinstance(cb, collections.Callable):
        cb2.attr_write = cb
    else:
        cb2.attr_write = cb.attr_write
    return self.__write_attributes_asynch(attr_values, cb2)

def __DeviceProxy__write_attribute_asynch(self, attr_name, value, cb=None):
    """
    write_attributes_asynch( self, values) -> int
    write_attributes_asynch( self, values, callback) -> None

            Shortcut to self.write_attributes_asynch([attr_name, value], cb)

        New in PyTango 7.0.0
    """
    return self.write_attributes_asynch([(attr_name, value)], cb)

def __DeviceProxy__write_read_attribute(self, attr_name, value, extract_as=ExtractAs.Numpy):
    return __check_read_attribute(self._write_read_attribute(attr_name, value, extract_as))

def __DeviceProxy__get_property(self, propname, value=None):
    """
    get_property(propname, value=None) -> PyTango.DbData
    
            Get a (list) property(ies) for a device.

            This method accepts the following types as propname parameter:
            1. string [in] - single property data to be fetched
            2. sequence<string> [in] - several property data to be fetched
            3. PyTango.DbDatum [in] - single property data to be fetched
            4. PyTango.DbData [in,out] - several property data to be fetched.
            5. sequence<DbDatum> - several property data to be feteched

            Note: for cases 3, 4 and 5 the 'value' parameter if given, is IGNORED.

            If value is given it must be a PyTango.DbData that will be filled with the
            property values

        Parameters :
            - propname : (any) property(ies) name(s)
            - value : (DbData) (optional, default is None meaning that the
                      method will create internally a PyTango.DbData and return
                      it filled with the property values

        Return     : (DbData) object containing the property(ies) value(s). If a
                     PyTango.DbData is given as parameter, it returns the same
                     object otherwise a new PyTango.DbData is returned

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
            for i in propname: new_propname.append(i)
            new_value = value
            if new_value is None:
                new_value = DbData()
            self._get_property(new_propname, new_value)
            return DbData_2_dict(new_value)
        elif isinstance(propname[0], DbDatum):
            new_value = DbData()
            for i in propname: new_value.append(i)
            self._get_property(new_value)
            return DbData_2_dict(new_value)

def __DeviceProxy__put_property(self, value):
    """
    put_property(self, value) -> None
    
            Insert or update a list of properties for this device.
            This method accepts the following types as value parameter:
            1. PyTango.DbDatum - single property data to be inserted
            2. PyTango.DbData - several property data to be inserted
            3. sequence<DbDatum> - several property data to be inserted
            4. dict<str, DbDatum> - keys are property names and value has data to be inserted
            5. dict<str, seq<str>> - keys are property names and value has data to be inserted
            6. dict<str, obj> - keys are property names and str(obj) is property value

        Parameters :
            - value : can be one of the following:
                1. PyTango.DbDatum - single property data to be inserted
                2. PyTango.DbData - several property data to be inserted
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
        raise TypeError('value must be a PyTango.DbDatum, PyTango.DbData,'\
                        'a sequence<DbDatum> or a dictionary')
    return self._put_property(value)

def __DeviceProxy__delete_property(self, value):
    """
    delete_property(self, value)
    
            Delete a the given of properties for this device.
            This method accepts the following types as value parameter:
            
                1. string [in] - single property to be deleted
                2. PyTango.DbDatum [in] - single property data to be deleted
                3. PyTango.DbData [in] - several property data to be deleted
                4. sequence<string> [in]- several property data to be deleted
                5. sequence<DbDatum> [in] - several property data to be deleted
                6. dict<str, obj> [in] - keys are property names to be deleted (values are ignored)
                7. dict<str, DbDatum> [in] - several DbDatum.name are property names to be deleted (keys are ignored)

        Parameters :
            - value : can be one of the following:
            
                1. string [in] - single property data to be deleted
                2. PyTango.DbDatum [in] - single property data to be deleted
                3. PyTango.DbData [in] - several property data to be deleted
                4. sequence<string> [in]- several property data to be deleted
                5. sequence<DbDatum> [in] - several property data to be deleted
                6. dict<str, obj> [in] - keys are property names to be deleted (values are ignored)
                7. dict<str, DbDatum> [in] - several DbDatum.name are property names to be deleted (keys are ignored)
        
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
        raise TypeError('value must be a string, PyTango.DbDatum, '\
                        'PyTango.DbData, a sequence or a dictionary')

    return self._delete_property(new_value)

def __DeviceProxy__get_property_list(self, filter, array=None):
    """
    get_property_list(self, filter, array=None) -> obj

            Get the list of property names for the device. The parameter
            filter allows the user to filter the returned name list. The
            wildcard character is '*'. Only one wildcard character is
            allowed in the filter parameter.

        Parameters :
                - filter[in] : (str) the filter wildcard
                - array[out] : (sequence obj or None) (optional, default is None)
                            an array to be filled with the property names. If None
                            a new list will be created internally with the values.

        Return     : the given array filled with the property names (or a new list
                    if array is None)

        Throws     : NonDbDevice, WrongNameSyntax,
                    ConnectionFailed (with database),
                    CommunicationFailed (with database)
                    DevFailed from database device

        New in PyTango 7.0.0
    """

    if array is None:
        new_array = StdStringVector()
        self._get_property_list(filter, new_array)
        return new_array

    if isinstance(array, StdStringVector):
        self._get_property_list(filter, array)
        return array
    elif isinstance(array, collections.Sequence):
        new_array = StdStringVector()
        self._get_property_list(filter, new_array)
        StdStringVector_2_seq(new_array, array)
        return array

    raise TypeError('array must be a mutable sequence<string>')

def __DeviceProxy__get_attribute_config(self, value):
    """
    get_attribute_config( self, name) -> AttributeInfoEx

            Return the attribute configuration for a single attribute.

        Parameters :
                - name : (str) attribute name
        Return     : (AttributeInfoEx) Object containing the attribute
                        information

        Throws     : ConnectionFailed, CommunicationFailed,
                        DevFailed from device
        
        Deprecated: use get_attribute_config_ex instead

    get_attribute_config( self, names) -> AttributeInfoList

            Return the attribute configuration for the list of specified attributes. To get all the
            attributes pass a sequence containing the constant PyTango.constants.AllAttr

        Parameters :
                - names : (sequence<str>) attribute names
        Return     : (AttributeInfoList) Object containing the attributes
                        information

        Throws     : ConnectionFailed, CommunicationFailed,
                        DevFailed from device

        Deprecated: use get_attribute_config_ex instead
    """
    if isinstance(value, StdStringVector) or is_pure_str(value):
        return self._get_attribute_config(value)
    elif isinstance(value, collections.Sequence):
        v = seq_2_StdStringVector(value)
        return self._get_attribute_config(v)

    raise TypeError('value must be a string or a sequence<string>')

def __DeviceProxy__get_attribute_config_ex(self, value):
    """
    get_attribute_config_ex( self, name) -> AttributeInfoListEx :

            Return the extended attribute configuration for a single attribute.

        Parameters :
                - name : (str) attribute name
        Return     : (AttributeInfoEx) Object containing the attribute
                        information

        Throws     : ConnectionFailed, CommunicationFailed,
                        DevFailed from device

    get_attribute_config( self, names) -> AttributeInfoListEx :

            Return the extended attribute configuration for the list of
            specified attributes. To get all the attributes pass a sequence
            containing the constant PyTango.constants.AllAttr
            
        Parameters :
                - names : (sequence<str>) attribute names
        Return     : (AttributeInfoList) Object containing the attributes
                        information

        Throws     : ConnectionFailed, CommunicationFailed,
                        DevFailed from device
    """
    if isinstance(value, StdStringVector):
        return self._get_attribute_config_ex(value)
    elif is_pure_str(value):
        v = StdStringVector()
        v.append(value)
        return self._get_attribute_config_ex(v)
    elif isinstance(value, collections.Sequence):
        v = seq_2_StdStringVector(value)
        return self._get_attribute_config_ex(v)

    raise TypeError('value must be a string or a sequence<string>')

def __DeviceProxy__set_attribute_config(self, value):
    """
    set_attribute_config( self, attr_info) -> None

            Change the attribute configuration for the specified attribute

        Parameters :
                - attr_info : (AttributeInfo) attribute information
        Return     : None

        Throws     : ConnectionFailed, CommunicationFailed,
                        DevFailed from device

    set_attribute_config( self, attr_info_ex) -> None

            Change the extended attribute configuration for the specified attribute

        Parameters :
                - attr_info_ex : (AttributeInfoEx) extended attribute information
        Return     : None

        Throws     : ConnectionFailed, CommunicationFailed,
                        DevFailed from device

    set_attribute_config( self, attr_info) -> None

            Change the attributes configuration for the specified attributes

        Parameters :
                - attr_info : (sequence<AttributeInfo>) attributes information
        Return     : None

        Throws     : ConnectionFailed, CommunicationFailed,
                        DevFailed from device

    set_attribute_config( self, attr_info_ex) -> None
    
            Change the extended attributes configuration for the specified attributes
        
        Parameters :
                - attr_info_ex : (sequence<AttributeInfoListEx>) extended
                                    attributes information
        Return     : None

        Throws     : ConnectionFailed, CommunicationFailed,
                        DevFailed from device

    """
    if isinstance(value, AttributeInfoEx):
        v = AttributeInfoListEx()
        v.append(value)
    elif isinstance(value, AttributeInfo):
        v = AttributeInfoList()
        v.append(value)
    elif isinstance(value, AttributeInfoList):
        v = value
    elif isinstance(value, AttributeInfoListEx):
        v = value
    elif isinstance(value, collections.Sequence):
        if not len(value): return
        if isinstance(value[0], AttributeInfoEx):
            v = AttributeInfoListEx()
        elif isinstance(value[0], AttributeInfo):
            v = AttributeInfoList()
        else:
            raise TypeError('value must be a AttributeInfo, AttributeInfoEx, ' \
                            'sequence<AttributeInfo> or sequence<AttributeInfoEx')
        for i in value: v.append(i)
    else:
        raise TypeError('value must be a AttributeInfo, AttributeInfoEx, ' \
                        'sequence<AttributeInfo> or sequence<AttributeInfoEx')

    return self._set_attribute_config(v)

def __DeviceProxy__get_event_map_lock(self):
    """
    Internal helper method"""
    if not hasattr(self, '_subscribed_events_lock'):
        # do it like this instead of self._subscribed_events = dict() to avoid
        # calling __setattr__ which requests list of tango attributes from device
        self.__dict__['_subscribed_events_lock'] = threading.Lock()
    return self._subscribed_events_lock

def __DeviceProxy__get_event_map(self):
    """
    Internal helper method"""
    if not hasattr(self, '_subscribed_events'):
        # do it like this instead of self._subscribed_events = dict() to avoid
        # calling __setattr__ which requests list of tango attributes from device
        self.__dict__['_subscribed_events'] = dict()
    return self._subscribed_events

def __DeviceProxy__subscribe_event ( self, attr_name, event_type, cb_or_queuesize, filters=[], stateless=False, extract_as=ExtractAs.Numpy):
    """
    subscribe_event(self, attr_name, event, callback, filters=[], stateless=False, extract_as=Numpy) -> int

            The client call to subscribe for event reception in the push model.
            The client implements a callback method which is triggered when the
            event is received. Filtering is done based on the reason specified and
            the event type. For example when reading the state and the reason
            specified is "change" the event will be fired only when the state
            changes. Events consist of an attribute name and the event reason.
            A standard set of reasons are implemented by the system, additional
            device specific reasons can be implemented by device servers programmers.

        Parameters :
            - attr_name : (str) The device attribute name which will be sent
                          as an event e.g. "current".
            - event_type: (EventType) Is the event reason and must be on the enumerated values:
                            * EventType.CHANGE_EVENT
                            * EventType.PERIODIC_EVENT
                            * EventType.ARCHIVE_EVENT
                            * EventType.ATTR_CONF_EVENT
                            * EventType.DATA_READY_EVENT
                            * EventType.USER_EVENT
            - callback  : (callable) Is any callable object or an object with a
                          callable "push_event" method.
            - filters   : (sequence<str>) A variable list of name,value pairs
                          which define additional filters for events.
            - stateless : (bool) When the this flag is set to false, an exception will
                          be thrown when the event subscription encounters a problem.
                          With the stateless flag set to true, the event subscription
                          will always succeed, even if the corresponding device server
                          is not running. The keep alive thread will try every 10
                          seconds to subscribe for the specified event. At every
                          subscription retry, a callback is executed which contains
                          the corresponding exception
            - extract_as : (ExtractAs)

        Return     : An event id which has to be specified when unsubscribing
                     from this event.

        Throws     : EventSystemFailed


    subscribe_event(self, attr_name, event, queuesize, filters=[], stateless=False ) -> int

            The client call to subscribe for event reception in the pull model.
            Instead of a callback method the client has to specify the size of the
            event reception buffer.
            The event reception buffer is implemented as a round robin buffer. This
            way the client can set-up different ways to receive events:
            
                * Event reception buffer size = 1 : The client is interested only
                  in the value of the last event received. All other events that
                  have been received since the last reading are discarded.
                * Event reception buffer size > 1 : The client has chosen to keep
                  an event history of a given size. When more events arrive since
                  the last reading, older events will be discarded.
                * Event reception buffer size = ALL_EVENTS : The client buffers all
                  received events. The buffer size is unlimited and only restricted
                  by the available memory for the client.

            All other parameters are similar to the descriptions given in the
            other subscribe_event() version.
    """
    
    if isinstance(cb_or_queuesize, collections.Callable):
        cb = __CallBackPushEvent()
        cb.push_event = cb_or_queuesize
    elif hasattr(cb_or_queuesize, "push_event") and isinstance(cb_or_queuesize.push_event, collections.Callable):
        cb = __CallBackPushEvent()
        cb.push_event = cb_or_queuesize.push_event
    elif is_integer(cb_or_queuesize):
        cb = cb_or_queuesize # queuesize
    else:
        raise TypeError("Parameter cb_or_queuesize should be a number, a" + \
                    " callable object or an object with a 'push_event' method.")

    event_id = self.__subscribe_event(attr_name, event_type, cb, filters, stateless, extract_as)

    with self.__get_event_map_lock():
        se = self.__get_event_map()
        evt_data = se.get(event_id)
        if evt_data is not None:
            desc = "Internal PyTango error:\n" \
                   "%s.subscribe_event(%s, %s) already has key %d assigned to (%s, %s)\n" \
                   "Please report error to PyTango" % \
                   (self, attr_name, event_type, event_id, evt_data[2], evt_data[1])
            Except.throw_exception("Py_InternalError", desc, "DeviceProxy.subscribe_event")
        se[event_id] = (cb, event_type, attr_name)
    return event_id

def __DeviceProxy__unsubscribe_event(self, event_id):
    """
    unsubscribe_event(self, event_id) -> None

            Unsubscribes a client from receiving the event specified by event_id.

        Parameters :
            - event_id   : (int) is the event identifier returned by the
                            DeviceProxy::subscribe_event(). Unlike in
                            TangoC++ we chech that the event_id has been
                            subscribed in this DeviceProxy.

        Return     : None

        Throws     : EventSystemFailed
    """
    with self.__get_event_map_lock():
        se = self.__get_event_map()
        if event_id not in se:
            raise IndexError("This device proxy does not own this subscription " + str(event_id))
        del se[event_id]
    self.__unsubscribe_event(event_id)

def __DeviceProxy__unsubscribe_event_all(self):
    with self.__get_event_map_lock():
        se = self.__get_event_map()
        event_ids = list(se.keys())
        se.clear()
    for event_id in event_ids:
        self.__unsubscribe_event(event_id)

def __DeviceProxy__get_events(self, event_id, callback=None, extract_as=ExtractAs.Numpy):
    """
    get_events( event_id, callback=None, extract_as=Numpy) -> None

        The method extracts all waiting events from the event reception buffer.

        If callback is not None, it is executed for every event. During event
        subscription the client must have chosen the pull model for this event.
        The callback will receive a parameter of type EventData,
        AttrConfEventData or DataReadyEventData depending on the type of the
        event (event_type parameter of subscribe_event).

        If callback is None, the method extracts all waiting events from the
        event reception buffer. The returned event_list is a vector of
        EventData, AttrConfEventData or DataReadyEventData pointers, just
        the same data the callback would have received.

        Parameters :
            - event_id : (int) is the event identifier returned by the
                DeviceProxy.subscribe_event() method.
                
            - callback : (callable) Any callable object or any object with a "push_event"
                         method.

            - extract_as: (ExtractAs)

        Return     : None

        Throws     : EventSystemFailed

        See Also   : subscribe_event

        New in PyTango 7.0.0
    """
    if callback is None:
        queuesize, event_type, attr_name = self.__get_event_map().get(event_id, (None, None, None))
        if event_type is None:
            raise ValueError("Invalid event_id. You are not subscribed to event %s." % str(event_id))
        if event_type in [  EventType.CHANGE_EVENT,
                            EventType.QUALITY_EVENT,
                            EventType.PERIODIC_EVENT,
                            EventType.ARCHIVE_EVENT,
                            EventType.USER_EVENT ]:
            return self.__get_data_events(event_id, extract_as)
        elif event_type in [ EventType.ATTR_CONF_EVENT ]:
            return self.__get_attr_conf_events(event_id, extract_as)
        elif event_type in [ EventType.DATA_READY_EVENT ]:
            return self.__get_data_ready_events(event_id, extract_as)
        else:
            assert (False)
            raise ValueError("Unknown event_type: " + str(event_type))
    elif isinstance(callback, collections.Callable):
        cb = __CallBackPushEvent()
        cb.push_event = callback
        return self.__get_callback_events(event_id, cb, extract_as)
    elif hasattr(callback, 'push_event') and isinstance(callback.push_event, collections.Callable):
        cb = __CallBackPushEvent()
        cb.push_event = callback.push_event
        return self.__get_callback_events(event_id, cb, extract_as)
    else:
        raise TypeError("Parameter 'callback' should be None, a callable object or an object with a 'push_event' method.")

def __DeviceProxy___get_info_(self):
    """Protected method that gets device info once and stores it in cache"""
    if not hasattr(self, '_dev_info'):
        try:
            self.__dict__["_dev_info"] = self.info()
        except:
            return __TangoInfo()
    return self._dev_info

def __DeviceProxy__str(self):
    info = self._get_info_()
    return "%s(%s)" % (info.dev_class, self.dev_name())

def __DeviceProxy__str(self):
    info = self._get_info_()
    return "%s(%s)" % (info.dev_class, self.dev_name())

def __init_DeviceProxy():
    DeviceProxy.__getattr__ = __DeviceProxy__getattr
    DeviceProxy.__setattr__ = __DeviceProxy__setattr
    DeviceProxy.__getitem__ = __DeviceProxy__getitem
    DeviceProxy.__setitem__ = __DeviceProxy__setitem
    DeviceProxy.__contains__ = __DeviceProxy__contains

    DeviceProxy._getAttributeNames = __DeviceProxy__getAttributeNames

    DeviceProxy.__refresh_cmd_cache = __DeviceProxy__refresh_cmd_cache
    DeviceProxy.__refresh_attr_cache = __DeviceProxy__refresh_attr_cache

    DeviceProxy.read_attribute = __DeviceProxy__read_attribute
    DeviceProxy.read_attributes_asynch = __DeviceProxy__read_attributes_asynch
    DeviceProxy.read_attribute_asynch = __DeviceProxy__read_attribute_asynch
    DeviceProxy.read_attribute_reply = __DeviceProxy__read_attribute_reply
    DeviceProxy.write_attributes_asynch = __DeviceProxy__write_attributes_asynch
    DeviceProxy.write_attribute_asynch = __DeviceProxy__write_attribute_asynch
    DeviceProxy.write_attribute_reply = DeviceProxy.write_attributes_reply
    DeviceProxy.write_read_attribute = __DeviceProxy__write_read_attribute

    DeviceProxy.get_property = __DeviceProxy__get_property
    DeviceProxy.put_property = __DeviceProxy__put_property
    DeviceProxy.delete_property = __DeviceProxy__delete_property
    DeviceProxy.get_property_list = __DeviceProxy__get_property_list
    DeviceProxy.get_attribute_config = __DeviceProxy__get_attribute_config
    DeviceProxy.get_attribute_config_ex = __DeviceProxy__get_attribute_config_ex
    DeviceProxy.set_attribute_config = __DeviceProxy__set_attribute_config

    DeviceProxy.__get_event_map = __DeviceProxy__get_event_map
    DeviceProxy.__get_event_map_lock = __DeviceProxy__get_event_map_lock
    DeviceProxy.subscribe_event = __DeviceProxy__subscribe_event
    DeviceProxy.unsubscribe_event = __DeviceProxy__unsubscribe_event
    DeviceProxy.__unsubscribe_event_all = __DeviceProxy__unsubscribe_event_all
    DeviceProxy.get_events = __DeviceProxy__get_events
    DeviceProxy.__str__ = __DeviceProxy__str
    DeviceProxy.__repr__ = __DeviceProxy__str
    
    DeviceProxy._get_info_ = __DeviceProxy___get_info_

    
def __doc_DeviceProxy():
    def document_method(method_name, desc, append=True):
        return __document_method(DeviceProxy, method_name, desc, append)

    DeviceProxy.__doc__ = """
    DeviceProxy is the high level Tango object which provides the client with
    an easy-to-use interface to TANGO devices. DeviceProxy provides interfaces
    to all TANGO Device interfaces.The DeviceProxy manages timeouts, stateless
    connections and reconnection if the device server is restarted. To create
    a DeviceProxy, a Tango Device name must be set in the object constructor.

    Example :
       dev = PyTango.DeviceProxy("sys/tg_test/1")
    """

#-------------------------------------
#   General methods
#-------------------------------------

    document_method("info", """
    info(self) -> DeviceInfo

            A method which returns information on the device

        Parameters : None
        Return     : (DeviceInfo) object
        Example    :
                dev_info = dev.info()
                print(dev_info.dev_class)
                print(dev_info.server_id)
                print(dev_info.server_host)
                print(dev_info.server_version)
                print(dev_info.doc_url)
                print(dev_info.dev_type)

            All DeviceInfo fields are strings except for the server_version
            which is an integer"
    """ )

    document_method("get_device_db", """
    get_device_db(self) -> Database

            Returns the internal database reference

        Parameters : None
        Return     : (Database) object

        New in PyTango 7.0.0
    """ )

    document_method("status", """
    status(self) -> str

            A method which returns the status of the device as a string.

        Parameters : None
        Return     : (str) describing the device status
    """ )

    document_method("state", """
    state(self) -> DevState

            A method which returns the state of the device.

        Parameters : None
        Return     : (DevState) constant
        Example :
                dev_st = dev.state()
                if dev_st == DevState.ON : ...
    """ )

    document_method("adm_name", """
    adm_name(self) -> str

            Return the name of the corresponding administrator device. This is
            useful if you need to send an administration command to the device
            server, e.g restart it

        New in PyTango 3.0.4
    """ )

    document_method("description", """
    description(self) -> str

            Get device description.

        Parameters : None
        Return     : (str) describing the device
    """ )

    document_method("name", """
    name(self) -> str

            Return the device name from the device itself.
    """ )

    document_method("alias", """
    alias(self) -> str

            Return the device alias if one is defined.
            Otherwise, throws exception.
    """ )

    document_method("ping", """
    ping(self) -> int

            A method which sends a ping to the device

        Parameters : None
        Return     : (int) time elapsed in milliseconds
        Throws     : exception if device is not alive
    """ )

    document_method("black_box", """
    black_box(self, n) -> sequence<str>

            Get the last commands executed on the device server

        Parameters :
            - n : n number of commands to get
        Return     : (sequence<str>) sequence of strings containing the date, time,
                     command and from which client computer the command
                     was executed
        Example :
                print(black_box(4))
    """ )

#-------------------------------------
#   Device methods
#-------------------------------------

    document_method("command_query", """
    command_query(self, command) -> CommandInfo

            Query the device for information about a single command.

        Parameters :
                - command : (str) command name
        Return     : (CommandInfo) object
        Throws     : ConnectionFailed, CommunicationFailed, DevFailed from device
        Example :
                com_info = dev.command_query(""DevString"")
                print(com_info.cmd_name)
                print(com_info.cmd_tag)
                print(com_info.in_type)
                print(com_info.out_type)
                print(com_info.in_type_desc)
                print(com_info.out_type_desc)
                print(com_info.disp_level)
                
        See CommandInfo documentation string form more detail
    """ )

    document_method("command_list_query", """
    command_list_query(self) -> sequence<CommandInfo>

            Query the device for information on all commands.

        Parameters : None
        Return     : (CommandInfoList) Sequence of CommandInfo objects
    """ )

    document_method("import_info", """
    import_info(self) -> DbDevImportInfo

            Query the device for import info from the database.

        Parameters : None
        Return     : (DbDevImportInfo)
        Example :
                dev_import = dev.import_info()
                print(dev_import.name)
                print(dev_import.exported)
                print(dev_ior.ior)
                print(dev_version.version)

        All DbDevImportInfo fields are strings except for exported which
        is an integer"
    """ )

#-------------------------------------
#   Property methods
#-------------------------------------
    # get_property -> in code
    # put_property -> in code
    # delete_property -> in code
    # get_property_list -> in code

#-------------------------------------
#   Attribute methods
#-------------------------------------
    document_method("get_attribute_list", """
    get_attribute_list(self) -> sequence<str>

            Return the names of all attributes implemented for this device.

        Parameters : None
        Return     : sequence<str>

        Throws     : ConnectionFailed, CommunicationFailed,
                     DevFailed from device
    """ )

    # get_attribute_config -> in code
    # get_attribute_config_ex -> in code

    document_method("attribute_query", """
    attribute_query(self, attr_name) -> AttributeInfoEx

            Query the device for information about a single attribute.

        Parameters :
                - attr_name :(str) the attribute name
        Return     : (AttributeInfoEx) containing the attribute
                     configuration

        Throws     : ConnectionFailed, CommunicationFailed,
                     DevFailed from device
    """ )

    document_method("attribute_list_query", """
    attribute_list_query(self) -> sequence<AttributeInfo>

            Query the device for info on all attributes. This method returns
            a sequence of PyTango.AttributeInfo.

        Parameters : None
        Return     : (sequence<AttributeInfo>) containing the
                     attributes configuration

        Throws     : ConnectionFailed, CommunicationFailed,
                     DevFailed from device
    """ )

    document_method("attribute_list_query_ex", """
    attribute_list_query_ex(self) -> sequence<AttributeInfoEx>

            Query the device for info on all attributes. This method returns
            a sequence of PyTango.AttributeInfoEx.

        Parameters : None
        Return     : (sequence<AttributeInfoEx>) containing the
                     attributes configuration

        Throws     : ConnectionFailed, CommunicationFailed,
                     DevFailed from device
    """ )

    # set_attribute_config -> in code

    document_method("read_attribute", """
    read_attribute(self, attr_name, extract_as=ExtractAs.Numpy) -> DeviceAttribute

            Read a single attribute.

        Parameters :
            - attr_name  : (str) The name of the attribute to read.
            - extract_as : (ExtractAs) Defaults to numpy.
        Return     : (DeviceAttribute)
        
        Throws     : ConnectionFailed, CommunicationFailed, DevFailed from device
            
    .. versionchanged:: 7.1.4
        For DevEncoded attributes, before it was returning a DeviceAttribute.value
        as a tuple **(format<str>, data<str>)** no matter what was the *extract_as*
        value was. Since 7.1.4, it returns a **(format<str>, data<buffer>)**
        unless *extract_as* is String, in which case it returns 
        **(format<str>, data<str>)**.

    .. versionchanged:: 8.0.0
        For DevEncoded attributes, now returns a DeviceAttribute.value
        as a tuple **(format<str>, data<bytes>)** unless *extract_as* is String,
        in which case it returns **(format<str>, data<str>)**. Carefull, if
        using python >= 3 data<str> is decoded using default python 
        *utf-8* encoding. This means that PyTango assumes tango DS was written
        encapsulating string into *utf-8* which is the default python enconding.
    """ )

    document_method("read_attributes", """
        read_attributes(self, attr_names, extract_as=ExtractAs.Numpy) -> sequence<DeviceAttribute>

                Read the list of specified attributes.

            Parameters :
                    - attr_names : (sequence<str>) A list of attributes to read.
                    - extract_as : (ExtractAs) Defaults to numpy.
            Return     : (sequence<DeviceAttribute>)

            Throws     : ConnectionFailed, CommunicationFailed, DevFailed from device
    """ )

    document_method("write_attribute", """
        write_attribute(self, attr_name, value) -> None
        write_attribute(self, attr_info, value) -> None

                Write a single attribute.

            Parameters :
                    - attr_name : (str) The name of the attribute to write.
                    - attr_info : (AttributeInfo)
                    - value : The value. For non SCALAR attributes it may be any sequence of sequences.

            Throws     : ConnectionFailed, CommunicationFailed, DeviceUnlocked, DevFailed from device
    """ )

    document_method("write_attributes", """
    write_attributes(self, name_val) -> None

            Write the specified attributes.

        Parameters :
                - name_val: A list of pairs (attr_name, value). See write_attribute

        Throws     : ConnectionFailed, CommunicationFailed, DeviceUnlocked,
                     DevFailed or NamedDevFailedList from device
    """ )

    document_method("write_read_attribute", """
    write_read_attribute(self, attr_name, value, extract_as=ExtractAs.Numpy) -> DeviceAttribute

            Write then read a single attribute in a single network call.
            By default (serialisation by device), the execution of this call in
            the server can't be interrupted by other clients.

        Parameters : see write_attribute(attr_name, value)
        Return     : A PyTango.DeviceAttribute object.

        Throws     : ConnectionFailed, CommunicationFailed, DeviceUnlocked,
                     DevFailed from device, WrongData

        New in PyTango 7.0.0
    """ )

#-------------------------------------
#   History methods
#-------------------------------------
    document_method("command_history", """
    command_history(self, cmd_name, depth) -> sequence<DeviceDataHistory>

            Retrieve command history from the command polling buffer. See
            chapter on Advanced Feature for all details regarding polling

        Parameters :
           - cmd_name  : (str) Command name.
           - depth     : (int) The wanted history depth.
        Return     : This method returns a vector of DeviceDataHistory types.

        Throws     : NonSupportedFeature, ConnectionFailed,
                     CommunicationFailed, DevFailed from device
    """ )

    document_method("attribute_history", """
    attribute_history(self, attr_name, depth, extract_as=ExtractAs.Numpy) -> sequence<DeviceAttributeHistory>

            Retrieve attribute history from the attribute polling buffer. See
            chapter on Advanced Feature for all details regarding polling

        Parameters :
           - attr_name  : (str) Attribute name.
           - depth      : (int) The wanted history depth.
           - extract_as : (ExtractAs)

        Return     : This method returns a vector of DeviceAttributeHistory types.

        Throws     : NonSupportedFeature, ConnectionFailed,
                     CommunicationFailed, DevFailed from device
    """ )

#-------------------------------------
#   Polling administration methods
#-------------------------------------

    document_method("polling_status", """
    polling_status(self) -> sequence<str>

            Return the device polling status.

        Parameters : None
        Return     : (sequence<str>) One string for each polled command/attribute.
                     Each string is multi-line string with:
                     
                        - attribute/command name
                        - attribute/command polling period in milliseconds
                        - attribute/command polling ring buffer
                        - time needed for last attribute/command execution in milliseconds
                        - time since data in the ring buffer has not been updated
                        - delta time between the last records in the ring buffer
                        - exception parameters in case of the last execution failed
    """ )

    document_method("poll_command", """
    poll_command(self, cmd_name, period) -> None

            Add a command to the list of polled commands.

        Parameters :
            - cmd_name : (str) command name
            - period   : (int) polling period in milliseconds
        Return     : None
    """ )

    document_method("poll_attribute", """
    poll_attribute(self, attr_name, period) -> None

            Add an attribute to the list of polled attributes.

        Parameters :
            - attr_name : (str) attribute name
            - period    : (int) polling period in milliseconds
        Return     : None
    """ )

    document_method("get_command_poll_period", """
    get_command_poll_period(self, cmd_name) -> int

            Return the command polling period.

        Parameters :
            - cmd_name : (str) command name
        Return     : polling period in milliseconds
    """ )

    document_method("get_attribute_poll_period", """
    get_attribute_poll_period(self, attr_name) -> int

            Return the attribute polling period.

        Parameters :
            - attr_name : (str) attribute name
        Return     : polling period in milliseconds
    """ )

    document_method("is_command_polled", """
    is_command_polled(self, cmd_name) -> bool

            True if the command is polled.

        Parameters :
            - cmd_name : (str) command name
        Return     : boolean value
    """ )

    document_method("is_attribute_polled", """
    is_attribute_polled(self, attr_name) -> bool

            True if the attribute is polled.

        Parameters :
            - attr_name : (str) attribute name
        Return     : boolean value
    """ )

    document_method("stop_poll_command", """
    stop_poll_command(self, cmd_name) -> None

            Remove a command from the list of polled commands.

        Parameters :
            - cmd_name : (str) command name
        Return     : None
    """ )

    document_method("stop_poll_attribute", """
    stop_poll_attribute(self, attr_name) -> None

            Remove an attribute from the list of polled attributes.

        Parameters :
            - attr_name : (str) attribute name
        Return     : None
    """ )

#-------------------------------------
#   Asynchronous methods
#-------------------------------------

    # read_attribute_asynch -> in code
    # read_attributes_asynch -> in code
    # read_attribute_reply -> in code
    document_method("read_attributes_reply", """
    read_attributes_reply(self, id, extract_as=ExtractAs.Numpy) -> DeviceAttribute

            Check if the answer of an asynchronous read_attribute is
            arrived (polling model).

        Parameters :
            - id         : (int) is the asynchronous call identifier.
            - extract_as : (ExtractAs)
        Return     : If the reply is arrived and if it is a valid reply, it is
                     returned to the caller in a list of DeviceAttribute. If the
                     reply is an exception, it is re-thrown by this call. An
                     exception is also thrown in case of the reply is not yet
                     arrived.

        Throws     : AsynCall, AsynReplyNotArrived, ConnectionFailed,
                     CommunicationFailed, DevFailed from device

        New in PyTango 7.0.0


    read_attributes_reply(self, id, timeout, extract_as=ExtractAs.Numpy) -> DeviceAttribute

            Check if the answer of an asynchronous read_attributes is arrived (polling model).

        Parameters :
            - id         : (int) is the asynchronous call identifier.
            - timeout    : (int)
            - extract_as : (ExtractAs)
        Return     : If the reply is arrived and if it is a valid reply, it is
                     returned to the caller in a list of DeviceAttribute. If the
                     reply is an exception, it is re-thrown by this call. If the
                     reply is not yet arrived, the call will wait (blocking the
                     process) for the time specified in timeout. If after
                     timeout milliseconds, the reply is still not there, an
                     exception is thrown. If timeout is set to 0, the call waits
                     until the reply arrived.

        Throws     : AsynCall, AsynReplyNotArrived, ConnectionFailed,
                     CommunicationFailed, DevFailed from device

        New in PyTango 7.0.0
    """ )

    document_method("pending_asynch_call", """
    pending_asynch_call(self) -> int

            Return number of device asynchronous pending requests"

        New in PyTango 7.0.0
    """ )

    # write_attributes_asynch -> in code

    document_method("write_attributes_reply", """
    write_attributes_reply(self, id) -> None

            Check if the answer of an asynchronous write_attributes is arrived
            (polling model). If the reply is arrived and if it is a valid reply,
            the call returned. If the reply is an exception, it is re-thrown by
            this call. An exception is also thrown in case of the reply is not
            yet arrived.

        Parameters :
            - id : (int) the asynchronous call identifier.
        Return     : None

        Throws     : AsynCall, AsynReplyNotArrived, CommunicationFailed, DevFailed from device.

        New in PyTango 7.0.0

    write_attributes_reply(self, id, timeout) -> None

            Check if the answer of an asynchronous write_attributes is arrived
            (polling model). id is the asynchronous call identifier. If the
            reply is arrived and if it is a valid reply, the call returned. If
            the reply is an exception, it is re-thrown by this call. If the
            reply is not yet arrived, the call will wait (blocking the process)
            for the time specified in timeout. If after timeout milliseconds,
            the reply is still not there, an exception is thrown. If timeout is
            set to 0, the call waits until the reply arrived.
            
        Parameters :
            - id      : (int) the asynchronous call identifier.
            - timeout : (int) the timeout
            
        Return     : None

        Throws     : AsynCall, AsynReplyNotArrived, CommunicationFailed, DevFailed from device.

        New in PyTango 7.0.0
    """ )
    
    document_method("write_attribute_reply", """
    write_attribute_reply(self, id) -> None

            Check if the answer of an asynchronous write_attribute is arrived
            (polling model). If the reply is arrived and if it is a valid reply,
            the call returned. If the reply is an exception, it is re-thrown by
            this call. An exception is also thrown in case of the reply is not
            yet arrived.

        Parameters :
            - id : (int) the asynchronous call identifier.
        Return     : None

        Throws     : AsynCall, AsynReplyNotArrived, CommunicationFailed, DevFailed from device.

        New in PyTango 7.0.0

    write_attribute_reply(self, id, timeout) -> None

            Check if the answer of an asynchronous write_attribute is arrived
            (polling model). id is the asynchronous call identifier. If the
            reply is arrived and if it is a valid reply, the call returned. If
            the reply is an exception, it is re-thrown by this call. If the
            reply is not yet arrived, the call will wait (blocking the process)
            for the time specified in timeout. If after timeout milliseconds,
            the reply is still not there, an exception is thrown. If timeout is
            set to 0, the call waits until the reply arrived.
            
        Parameters :
            - id      : (int) the asynchronous call identifier.
            - timeout : (int) the timeout
            
        Return     : None

        Throws     : AsynCall, AsynReplyNotArrived, CommunicationFailed, DevFailed from device.

        New in PyTango 7.0.0
    """ )

#-------------------------------------
#   Logging administration methods
#-------------------------------------

    document_method("add_logging_target", """
    add_logging_target(self, target_type_target_name) -> None

            Adds a new logging target to the device.

            The target_type_target_name input parameter must follow the
            format: target_type::target_name. Supported target types are:
            console, file and device. For a device target, the target_name
            part of the target_type_target_name parameter must contain the
            name of a log consumer device (as defined in A.8). For a file
            target, target_name is the full path to the file to log to. If
            omitted, the device's name is used to build the file name
            (which is something like domain_family_member.log). Finally, the
            target_name part of the target_type_target_name input parameter
            is ignored in case of a console target and can be omitted.

        Parameters :
            - target_type_target_name : (str) logging target
        Return     : None

        Throws     : DevFailed from device

        New in PyTango 7.0.0
    """ )

    document_method("remove_logging_target", """
    remove_logging_target(self, target_type_target_name) -> None

            Removes a logging target from the device's target list.

            The target_type_target_name input parameter must follow the
            format: target_type::target_name. Supported target types are:
            console, file and device. For a device target, the target_name
            part of the target_type_target_name parameter must contain the
            name of a log consumer device (as defined in ). For a file
            target, target_name is the full path to the file to remove.
            If omitted, the default log file is removed. Finally, the
            target_name part of the target_type_target_name input parameter
            is ignored in case of a console target and can be omitted.
            If target_name is set to '*', all targets of the specified
            target_type are removed.

        Parameters :
            - target_type_target_name : (str) logging target
        Return     : None

        New in PyTango 7.0.0
    """ )

    document_method("get_logging_target", """
    get_logging_target(self) -> sequence<str>

            Returns a sequence of string containing the current device's
            logging targets. Each vector element has the following format:
            target_type::target_name. An empty sequence is returned is the
            device has no logging targets.

        Parameters : None
        Return     : a squence<str> with the logging targets

        New in PyTango 7.0.0
    """ )

    document_method("get_logging_level", """
    get_logging_level(self) -> int

            Returns the current device's logging level, where:
                - 0=OFF
                - 1=FATAL
                - 2=ERROR
                - 3=WARNING
                - 4=INFO
                - 5=DEBUG

        Parameters :None
        Return     : (int) representing the current logging level

        New in PyTango 7.0.0
    """ )

    document_method("set_logging_level", """
    set_logging_level(self, (int)level) -> None

            Changes the device's logging level, where:
                - 0=OFF
                - 1=FATAL
                - 2=ERROR
                - 3=WARNING
                - 4=INFO
                - 5=DEBUG

        Parameters :
            - level : (int) logging level
        Return     : None

        New in PyTango 7.0.0
    """ )

#-------------------------------------
#   Event methods
#-------------------------------------

    # subscribe_event -> in code
    # unsubscribe_event -> in code
    # get_events -> in code

    document_method("event_queue_size", """
    event_queue_size(self, event_id) -> int

            Returns the number of stored events in the event reception
            buffer. After every call to DeviceProxy.get_events(), the event
            queue size is 0. During event subscription the client must have
            chosen the 'pull model' for this event. event_id is the event
            identifier returned by the DeviceProxy.subscribe_event() method.

        Parameters :
            - event_id : (int) event identifier
        Return     : an integer with the queue size

        Throws     : EventSystemFailed

        New in PyTango 7.0.0
    """ )

    document_method("get_last_event_date", """
    get_last_event_date(self, event_id) -> TimeVal

            Returns the arrival time of the last event stored in the event
            reception buffer. After every call to DeviceProxy:get_events(),
            the event reception buffer is empty. In this case an exception
            will be returned. During event subscription the client must have
            chosen the 'pull model' for this event. event_id is the event
            identifier returned by the DeviceProxy.subscribe_event() method.

        Parameters :
            - event_id : (int) event identifier
        Return     : (PyTango.TimeVal) representing the arrival time

        Throws     : EventSystemFailed

        New in PyTango 7.0.0
    """ )

    document_method("is_event_queue_empty", """
    is_event_queue_empty(self, event_id) -> bool

            Returns true when the event reception buffer is empty. During
            event subscription the client must have chosen the 'pull model'
            for this event. event_id is the event identifier returned by the
            DeviceProxy.subscribe_event() method.

            Parameters :
                - event_id : (int) event identifier
            Return     : (bool) True if queue is empty or False otherwise

            Throws     : EventSystemFailed

            New in PyTango 7.0.0
    """ )

#-------------------------------------
#   Locking methods
#-------------------------------------
    document_method("lock", """
    lock(self, (int)lock_validity) -> None

            Lock a device. The lock_validity is the time (in seconds) the
            lock is kept valid after the previous lock call. A default value
            of 10 seconds is provided and should be fine in most cases. In
            case it is necessary to change the lock validity, it's not
            possible to ask for a validity less than a minimum value set to
            2 seconds. The library provided an automatic system to
            periodically re lock the device until an unlock call. No code is
            needed to start/stop this automatic re-locking system. The
            locking system is re-entrant. It is then allowed to call this
            method on a device already locked by the same process. The
            locking system has the following features:

              * It is impossible to lock the database device or any device
                server process admin device
              * Destroying a locked DeviceProxy unlocks the device
              * Restarting a locked device keeps the lock
              * It is impossible to restart a device locked by someone else
              * Restarting a server breaks the lock

            A locked device is protected against the following calls when
            executed by another client:

              * command_inout call except for device state and status
                requested via command and for the set of commands defined as
                allowed following the definition of allowed command in the
                Tango control access schema.
              * write_attribute call
              * write_read_attribute call
              * set_attribute_config call

        Parameters :
            - lock_validity : (int) lock validity time in seconds
                                (optional, default value is
                                PyTango.constants.DEFAULT_LOCK_VALIDITY)
        Return     : None

        New in PyTango 7.0.0
    """ )

    document_method("unlock", """
    unlock(self, (bool)force) -> None

            Unlock a device. If used, the method argument provides a back
            door on the locking system. If this argument is set to true,
            the device will be unlocked even if the caller is not the locker.
            This feature is provided for administration purpopse and should
            be used very carefully. If this feature is used, the locker will
            receive a DeviceUnlocked during the next call which is normally
            protected by the locking Tango system.

        Parameters :
            - force : (bool) force unlocking even if we are not the
                      locker (optional, default value is False)
        Return     : None

        New in PyTango 7.0.0
    """ )

    document_method("locking_status", """
    locking_status(self) -> str

            This method returns a plain string describing the device locking
            status. This string can be:

              * 'Device <device name> is not locked' in case the device is
                not locked
              * 'Device <device name> is locked by CPP or Python client with
                PID <pid> from host <host name>' in case the device is
                locked by a CPP client
              * 'Device <device name> is locked by JAVA client class
                <main class> from host <host name>' in case the device is
                locked by a JAVA client

        Parameters : None
        Return     : a string representing the current locking status

        New in PyTango 7.0.0"
    """ )

    document_method("is_locked", """
    is_locked(self) -> bool

            Returns True if the device is locked. Otherwise, returns False.

        Parameters : None
        Return     : (bool) True if the device is locked. Otherwise, False

        New in PyTango 7.0.0
    """ )

    document_method("is_locked_by_me", """
    is_locked_by_me(self) -> bool

            Returns True if the device is locked by the caller. Otherwise,
            returns False (device not locked or locked by someone else)

        Parameters : None
        Return     : (bool) True if the device is locked by us.
                        Otherwise, False

        New in PyTango 7.0.0
    """ )

    document_method("get_locker", """
    get_locker(self, lockinfo) -> bool

            If the device is locked, this method returns True an set some
            locker process informations in the structure passed as argument.
            If the device is not locked, the method returns False.

        Parameters :
            - lockinfo [out] : (PyTango.LockInfo) object that will be filled
                                with lock informantion
        Return     : (bool) True if the device is locked by us.
                     Otherwise, False

        New in PyTango 7.0.0
    """ )

def device_proxy_init(doc=True):
    __init_DeviceProxy()
    if doc:
        __doc_DeviceProxy()
