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

__all__ = ["group_element_init"]

__docformat__ = "restructuredtext"

import operator

from ._PyTango import StdStringVector, GroupElement

from .utils import document_method as __document_method
from .utils import seq_2_StdStringVector
import collections

def __apply_to(fn, key):
    if isinstance(key, slice):
        if key.step:
            return [ fn(x) for x in range(key.start, key.stop, key.step) ]
        else:
            return [ fn(x) for x in range(key.start, key.stop) ]
    else:
        return fn(key)

def __GroupElement__contains(self, pattern):
    return self.contains(pattern)

def __GroupElement__get_one_item(self, key):
    x = self.get_group(key)
    if x is not None:
        return x
    return self.get_device(key)
    
def __GroupElement__getitem(self, key):
    fn = lambda x: __GroupElement__get_one_item(self, x)
    return __apply_to(fn, key)

def __GroupElement__delitem(self, key):
    fn = lambda x: self.remove(x)
    return __apply_to(fn, key)

def __GroupElement__len(self):
    return self.get_size()

def __GroupElement__add(self, patterns_or_group, timeout_ms=-1):
    if isinstance(patterns_or_group, GroupElement):
        return self.__add(patterns_or_group, timeout_ms)
    elif isinstance(patterns_or_group, StdStringVector):
        return self.__add(patterns_or_group, timeout_ms)
    elif isinstance(patterns_or_group, str):
        return self.__add(patterns_or_group, timeout_ms)
    elif isinstance(patterns_or_group, collections.Sequence):
        patterns = seq_2_StdStringVector(patterns_or_group)
        return self.__add(patterns, timeout_ms)
    else:
        raise TypeError('Parameter patterns_or_group: Should be GroupElement, str or a sequence of strings.')
    
def __GroupElement__remove(self, patterns, forward=True):
    if isinstance(patterns, str):
        return self.__remove(patterns, forward)
    elif isinstance(patterns, collections.Sequence):
        std_patterns = seq_2_StdStringVector(patterns)
        return self.__remove(std_patterns, forward)
    else:
        raise TypeError('Parameter patterns: Should be a str or a sequence of str.')

def __GroupElement__comand_inout(self, cmd_name, param=None, forward=True):
    if param is None:
        idx = self.command_inout_asynch(cmd_name, forget=False, forward=forward, reserved=-1)
    else:
        idx = self.command_inout_asynch(cmd_name, param, forget=False, forward=forward, reserved=-1)
    return self.command_inout_reply(idx)

def __GroupElement__read_attribute(self, attr_name, forward=True):
    idx = self.read_attribute_asynch(attr_name, forward, reserved=-1)
    return self.read_attribute_reply(idx)

def __GroupElement__read_attributes(self, attr_names, forward=True):
    idx = self.read_attributes_asynch(attr_names, forward, reserved=-1)
    return self.read_attributes_reply(idx)

def __GroupElement__write_attribute(self, attr_name, value, forward=True):
    idx = self.write_attribute_asynch(attr_name, value, forward, reserved=-1)
    return self.write_attribute_reply(idx)

def __init_GroupElement():
    
    GroupElement.__contains__ = __GroupElement__contains
    GroupElement.__getitem__ = __GroupElement__getitem
    GroupElement.__delitem__ = __GroupElement__delitem
    GroupElement.__len__ = __GroupElement__len

    GroupElement.add = __GroupElement__add
    GroupElement.remove = __GroupElement__remove
    
    GroupElement.command_inout = __GroupElement__comand_inout
    GroupElement.read_attribute = __GroupElement__read_attribute
    GroupElement.read_attributes = __GroupElement__read_attributes
    GroupElement.write_attribute = __GroupElement__write_attribute

def __doc_GroupElement():
    def document_method(method_name, desc, append=True):
        return __document_method(GroupElement, method_name, desc, append)
    
    document_method("add", """
    add(self, patterns, timeout_ms=-1) -> None
        
            Attaches any device which name matches one of the specified patterns.

            This method first asks to the Tango database the list of device names
            matching one the patterns. Devices are then attached to the group in
            the order in which they are returned by the database.

            Any device already present in the hierarchy (i.e. a device belonging to
            the group or to one of its subgroups), is silently ignored but its
            client side timeout is set to timeout_ms milliseconds if timeout_ms
            is different from -1.

        Parameters :
            - patterns   : (str | sequence<str>) can be a simple device name or
                            a device name pattern (e.g. domain_*/ family/member_*),
                            or a sequence of these.
            - timeout_ms : (int) If timeout_ms is different from -1, the client
                            side timeouts of all devices matching the
                            specified patterns are set to timeout_ms
                            milliseconds.
        Return     : None

        Throws     : TypeError, ArgumentError
    """ )

    document_method("remove", """
    remove(self, patterns, forward=True) -> None
        
            Removes any group or device which name matches the specified pattern. 
            
            The pattern parameter can be a group name, a device name or a device
            name pattern (e.g domain_*/family/member_*).
            
            Since we can have groups with the same name in the hierarchy, a group
            name can be fully qualified to specify which group should be removed.
            Considering the following group:

                ::

                    -> gauges 
                    | -> cell-01 
                    |     |-> penning 
                    |     |    |-> ...
                    |     |-> pirani
                    |          |-> ...
                    | -> cell-02
                    |     |-> penning
                    |     |    |-> ...
                    |     |-> pirani
                    |          |-> ...
                    | -> cell-03
                    |     |-> ... 
                    |     
                    | -> ...  
            
            A call to gauges->remove("penning") will remove any group named
            "penning" in the hierarchy while gauges->remove("gauges.cell-02.penning")
            will only remove the specified group.
        
        Parameters :
            - patterns   : (str | sequence<str>) A string with the pattern or a
                           list of patterns.
            - forward    : (bool) If fwd is set to true (the default), the remove
                           request is also forwarded to subgroups. Otherwise,
                           it is only applied to the local set of elements.
                           For instance, the following code remove any
                           stepper motor in the hierarchy:
                           
                               root_group->remove("*/stepper_motor/*");
                
        Return     : None
        
        Throws     : 
    """ )

    document_method("contains", """
    contains(self, pattern, forward=True) -> bool
        
        Parameters :
            - pattern    : (str) The pattern can be a fully qualified or simple
                            group name, a device name or a device name pattern.
            - forward    : (bool) If fwd is set to true (the default), the remove
                            request is also forwarded to subgroups. Otherwise,
                            it is only applied to the local set of elements.
                
        Return     : (bool) Returns true if the hierarchy contains groups and/or
                     devices which name matches the specified pattern. Returns
                     false otherwise.
        
        Throws     : 
    """ )


    document_method("get_device", """
    get_device(self, dev_name) -> DeviceProxy
    get_device(self, idx) -> DeviceProxy

            Returns a reference to the specified device or None if there is no
            device by that name in the group. Or, returns a reference to the
            "idx-th" device in the hierarchy or NULL if the hierarchy contains
            less than "idx" devices.

            This method may throw an exception in case the specified device belongs
            to the group but can't be reached (not registered, down...). See example
            below:

            ::

                try:
                    dp = g.get_device("my/device/01")
                    if dp is None:
                        # my/device/01 does not belong to the group
                        pass
                except DevFailed, f:
                    # my/device/01 belongs to the group but can't be reached
                    pass

            The request is systematically forwarded to subgroups (i.e. if no device
            named device_name could be found in the local set of devices, the
            request is forwarded to subgroups).
            
        Parameters :
            - dev_name    : (str) Device name.
            - idx         : (int) Device number.
                
        Return     : (DeviceProxy) Be aware that this method returns a
                    different DeviceProxy referring to the same device each time.
                    So, do not use it directly for permanent things.

        Example:
                        # WRONG: The DeviceProxy will quickly go out of scope
                        # and disappear (thus, the event will be automatically
                        # unsubscribed)
                        g.get_device("my/device/01").subscribe_events('attr', callback)

                        # GOOD:
                        dp = g.get_device("my/device/01")
                        dp.subscribe_events('attr', callback)
        
        Throws     : DevFailed
    """ )

    document_method("get_group", """
    get_group(self, group_name ) -> Group

            Returns a reference to the specified group or None if there is no group
            by that name. The group_name can be a fully qualified name.

            Considering the following group:

            ::
                    
                -> gauges
                    |-> cell-01
                    |    |-> penning
                    |    |    |-> ... 
                    |    |-> pirani
                    |    |-> ... 
                    |-> cell-02
                    |    |-> penning
                    |    |    |-> ...
                    |    |-> pirani
                    |    |-> ...
                    | -> cell-03
                    |    |-> ...
                    |
                    | -> ...  

            A call to gauges.get_group("penning") returns the first group named
            "penning" in the hierarchy (i.e. gauges.cell-01.penning) while
            gauges.get_group("gauges.cell-02.penning'') returns the specified group.
            
            The request is systematically forwarded to subgroups (i.e. if no group
            named group_name could be found in the local set of elements, the request
            is forwarded to subgroups).
        
        Parameters :
            - group_name : (str)
        
        Return     : (Group)
        
        Throws     :
        
        New in PyTango 7.0.0
    """ )

    
# Tango methods (~ DeviceProxy interface)
    document_method("ping", """
    ping(self, forward=True) -> bool

            Ping all devices in a group.
        
        Parameters :
            - forward    : (bool) If fwd is set to true (the default), the request
                            is also forwarded to subgroups. Otherwise, it is
                            only applied to the local set of devices.
                
        Return     : (bool) This method returns true if all devices in
                     the group are alive, false otherwise.
        
        Throws     : 
    """ )

    document_method("set_timeout_millis", """
    set_timeout_millis(self, timeout_ms) -> bool
        
            Set client side timeout for all devices composing the group in
            milliseconds. Any method which takes longer than this time to execute
            will throw an exception.

        Parameters :
            - timeout_ms : (int)
                
        Return     : None
        
        Throws     : (errors are ignored)

        New in PyTango 7.0.0
    """ )

    document_method("command_inout_asynch", """
    command_inout_asynch(self, cmd_name, forget=False, forward=True, reserved=-1 ) -> int
    command_inout_asynch(self, cmd_name, param=None forget=False, forward=True, reserved=-1 ) -> int
        
            Executes a Tango command on each device in the group asynchronously.
            The method sends the request to all devices and returns immediately.
            Pass the returned request id to Group.command_inout_reply() to obtain
            the results.

        Parameters :
            - cmd_name : (str) Command name
            - param    : (any)
            - forget   : (bool) Fire and forget flag. If set to true, it means that
                            no reply is expected (i.e. the caller does not care
                            about it and will not even try to get it)
            - forward  : (bool) If it is set to true (the default) request is
                            forwarded to subgroups. Otherwise, it is only applied
                            to the local set of devices.
            - reserved : (int) is reserved for internal purpose and should not be
                            used. This parameter may disappear in a near future.
                
        Return     : (int) request id. Pass the returned request id to
                    Group.command_inout_reply() to obtain the results.
        
        Throws     :
    """ )

    document_method("command_inout_reply", """
    command_inout_reply(self, req_id, timeout_ms=0) -> sequence<GroupCmdReply>

            Returns the results of an asynchronous command.

        Parameters :
            - req_id     : (int) Is a request identifier previously returned by one
                            of the command_inout_asynch methods
            - timeout_ms : (int) For each device in the hierarchy, if the command
                            result is not yet available, command_inout_reply
                            wait timeout_ms milliseconds before throwing an
                            exception. This exception will be part of the
                            global reply. If timeout_ms is set to 0,
                            command_inout_reply waits "indefinitely".
                
        Return     : (sequence<GroupCmdReply>)
        
        Throws     : 
    """ )

    document_method("command_inout", """
    command_inout(   self, cmd_name, param=None, forward=True) -> sequence<GroupCmdReply>

            Just a shortcut to do:
                self.command_inout_reply(self.command_inout_asynch(...))

        Parameters:
            - cmd_name : (str)
            - param    : (any)
            - forward  : (bool)

        Return : (sequence<GroupCmdReply>)
    """ )

    document_method("read_attribute_asynch", """
    read_attribute_asynch(self, attr_name, forward=True, reserved=-1 ) -> int

            Reads an attribute on each device in the group asynchronously.
            The method sends the request to all devices and returns immediately.

        Parameters :
            - attr_name : (str) Name of the attribute to read.
            - forward   : (bool) If it is set to true (the default) request is
                            forwarded to subgroups. Otherwise, it is only applied
                            to the local set of devices.
            - reserved  : (int) is reserved for internal purpose and should not be
                            used. This parameter may disappear in a near future.
                
        Return     : (int) request id. Pass the returned request id to
                    Group.read_attribute_reply() to obtain the results.
        
        Throws     :
    """ )

    document_method("read_attributes_asynch", """
    read_attributes_asynch(self, attr_names, forward=True, reserved=-1 ) -> int

            Reads the attributes on each device in the group asynchronously.
            The method sends the request to all devices and returns immediately.

        Parameters :
            - attr_names : (sequence<str>) Name of the attributes to read.
            - forward    : (bool) If it is set to true (the default) request is
                            forwarded to subgroups. Otherwise, it is only applied
                            to the local set of devices.
            - reserved   : (int) is reserved for internal purpose and should not be
                            used. This parameter may disappear in a near future.
                
        Return     : (int) request id. Pass the returned request id to
                    Group.read_attributes_reply() to obtain the results.
        
        Throws     :
    """ )

    document_method("read_attribute_reply", """
    read_attribute_reply(self, req_id, timeout_ms=0 ) -> sequence<GroupAttrReply>

            Returns the results of an asynchronous attribute reading.

        Parameters :
            - req_id     : (int) a request identifier previously returned by read_attribute_asynch.
            - timeout_ms : (int) For each device in the hierarchy, if the attribute
                            value is not yet available, read_attribute_reply
                            wait timeout_ms milliseconds before throwing an
                            exception. This exception will be part of the
                            global reply. If timeout_ms is set to 0,
                            read_attribute_reply waits "indefinitely".
                
        Return     : (sequence<GroupAttrReply>)
        
        Throws     :
    """ )

    document_method("read_attributes_reply", """
    read_attributes_reply(self, req_id, timeout_ms=0 ) -> sequence<GroupAttrReply>

            Returns the results of an asynchronous attribute reading.

        Parameters :
            - req_id     : (int) a request identifier previously returned by read_attribute_asynch.
            - timeout_ms : (int) For each device in the hierarchy, if the attribute
                            value is not yet available, read_attribute_reply
                            wait timeout_ms milliseconds before throwing an
                            exception. This exception will be part of the
                            global reply. If timeout_ms is set to 0,
                            read_attributes_reply waits "indefinitely".
                
        Return     : (sequence<GroupAttrReply>)
        
        Throws     :
    """ )

    document_method("read_attribute", """
    read_attribute(  self, attr_name, forward=True) -> sequence<GroupAttrReply>

            Just a shortcut to do:
                self.read_attribute_reply(self.read_attribute_asynch(...))
    """ )

    document_method("read_attributes", """
    read_attributes( self, attr_names, forward=True) -> sequence<GroupAttrReply>

            Just a shortcut to do:
                self.read_attributes_reply(self.read_attributes_asynch(...))
    """ )

    document_method("write_attribute_asynch", """
    write_attribute_asynch(self, attr_name, value, forward=True, reserved=-1 ) -> int

            Writes an attribute on each device in the group asynchronously.
            The method sends the request to all devices and returns immediately.

        Parameters :
            - attr_name : (str) Name of the attribute to write.
            - value     : (any) Value to write. See DeviceProxy.write_attribute
            - forward   : (bool) If it is set to true (the default) request is
                            forwarded to subgroups. Otherwise, it is only applied
                            to the local set of devices.
            - reserved  : (int) is reserved for internal purpose and should not
                            be used. This parameter may disappear in a near
                            future.
                
        Return     : (int) request id. Pass the returned request id to
                    Group.write_attribute_reply() to obtain the acknowledgements.
        
        Throws     :
    """ )
        
    document_method("write_attribute_reply", """
    write_attribute_reply(self, req_id, timeout_ms=0 ) -> sequence<GroupReply>

            Returns the acknowledgements of an asynchronous attribute writing.

        Parameters :
            - req_id     : (int) a request identifier previously returned by write_attribute_asynch.
            - timeout_ms : (int) For each device in the hierarchy, if the acknowledgment
                            is not yet available, write_attribute_reply
                            wait timeout_ms milliseconds before throwing an
                            exception. This exception will be part of the
                            global reply. If timeout_ms is set to 0,
                            write_attribute_reply waits "indefinitely".
                
        Return     : (sequence<GroupReply>)
        
        Throws     :
    """ )
        
    document_method("write_attribute", """
    write_attribute( self, attr_name, value, forward=True) -> sequence<GroupReply>

            Just a shortcut to do:
                self.write_attribute_reply(self.write_attribute_asynch(...))
    """ )

# Misc methods
    document_method("get_name", """
        Get the name of the group. Eg: Group('name').get_name() == 'name'
    """ )
    document_method("get_fully_qualified_name", """
        Get the complete (dpt-separated) name of the group. This takes into
        consideration the name of the group and its parents.
    """ )
    document_method("enable", "Enables a group or a device element in a group.")
    document_method("disable", "Disables a group or a device element in a group.")
    document_method("is_enabled", "Check if a group is enabled.\nNew in PyTango 7.0.0")
    document_method("name_equals", "New in PyTango 7.0.0")
    document_method("name_matches", "New in PyTango 7.0.0")
    document_method("get_size", """
    get_size(self, forward=True) -> int

        Parameters :
            - forward : (bool) If it is set to true (the default), the request is
                        forwarded to sub-groups.
                
        Return     : (int) The number of the devices in the hierarchy
        
        Throws     :
    """ )

# "Should not be used" methods
    # get_parent(self)
    

def group_element_init(doc=True):
    __init_GroupElement()
    if doc:
        __doc_GroupElement()
