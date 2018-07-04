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

__all__ = ("Group", "group_init")

__docformat__ = "restructuredtext"

import collections
from ._tango import __Group as _RealGroup, StdStringVector
from .utils import seq_2_StdStringVector, is_pure_str
from .utils import document_method as __document_method
from .device_proxy import __init_device_proxy_internals as init_device_proxy


def _apply_to(fn, key):
    if isinstance(key, slice):
        if key.step:
            return [fn(x) for x in range(key.start, key.stop, key.step)]
        else:
            return [fn(x) for x in range(key.start, key.stop)]
    else:
        return fn(key)


def _get_one_item(group, key):
    x = group.get_group(key)
    if x is not None:
        return x
    return group.get_device(key)


# I define Group as a proxy to __Group, where group is the actual
# C++ Tango::Group object. Most functions just call the __group object
# and are defined dynamically in __init_proxy_Group, also copying it's
# documentation strings.
# The proxy is useful for add(group). In this case the parameter 'group'
# becomes useless. With the proxy we make that parameter come to live
# again before returning.
# Another function that needs to be adapted to this is get_group because
# we want to return a Group, not a __Group!
# The get_device method also needs to be adapted in order to properly
# initialize the returned proxy with its python attributes.
class Group:
    """A Tango Group represents a hierarchy of tango devices. The hierarchy
    may have more than one level. The main goal is to group devices with
    same attribute(s)/command(s) to be able to do parallel requests."""

    def __init__(self, name):
        if is_pure_str(name):
            name = _RealGroup(name)
        if not isinstance(name, _RealGroup):
            raise TypeError("Constructor expected receives a str")
        self.__group = name

    def add(self, pattern_subgroup, timeout_ms=-1):
        if isinstance(pattern_subgroup, Group):
            name = pattern_subgroup.__group.get_name()
            self._add(pattern_subgroup.__group, timeout_ms)
            pattern_subgroup.__group = self.get_group(name)
        else:
            self._add(pattern_subgroup, timeout_ms)

    def _add(self, patterns_or_group, timeout_ms=-1):
        if isinstance(patterns_or_group, _RealGroup):
            return self.__group._add(patterns_or_group, timeout_ms)
        elif isinstance(patterns_or_group, StdStringVector):
            return self.__group._add(patterns_or_group, timeout_ms)
        elif isinstance(patterns_or_group, str):
            return self.__group._add(patterns_or_group, timeout_ms)
        elif isinstance(patterns_or_group, collections.Sequence):
            patterns = seq_2_StdStringVector(patterns_or_group)
            return self.__group._add(patterns, timeout_ms)
        else:
            raise TypeError('Parameter patterns_or_group: Should be Group, str or a sequence of strings.')

    def remove(self, patterns, forward=True):
        if isinstance(patterns, str):
            return self.__group._remove(patterns, forward)
        elif isinstance(patterns, collections.Sequence):
            std_patterns = seq_2_StdStringVector(patterns)
            return self.__group._remove(std_patterns, forward)
        else:
            raise TypeError('Parameter patterns: Should be a str or a sequence of str.')

    def get_device(self, name_or_index):
        proxy = self.__group.get_device(name_or_index)
        init_device_proxy(proxy)
        return proxy

    def get_group(self, group_name):
        internal = self.__group.get_group(group_name)
        if internal is None:
            return None
        return Group(internal)

    def __contains__(self, pattern):
        return self.contains(pattern)

    def __getitem__(self, key):
        fn = lambda x: _get_one_item(self, x)
        return _apply_to(fn, key)

    def __delitem__(self, key):
        fn = lambda x: self.remove(x)
        return _apply_to(fn, key)

    def __len__(self):
        return self.get_size()

    def command_inout(self, cmd_name, param=None, forward=True):
        if param is None:
            idx = self.command_inout_asynch(cmd_name, forget=False, forward=forward)
        else:
            idx = self.command_inout_asynch(cmd_name, param, forget=False, forward=forward)
        return self.command_inout_reply(idx)

    def read_attribute(self, attr_name, forward=True):
        idx = self.__group.read_attribute_asynch(attr_name, forward)
        return self.__group.read_attribute_reply(idx)

    def read_attributes(self, attr_names, forward=True):
        idx = self.__group.read_attributes_asynch(attr_names, forward)
        return self.__group.read_attributes_reply(idx)

    def write_attribute(self, attr_name, value, forward=True, multi=False):
        idx = self.__group.write_attribute_asynch(attr_name, value, forward=forward, multi=multi)
        return self.__group.write_attribute_reply(idx)


def __init_proxy_Group():
    proxy_methods = [
        # 'add',  # Needs to be adapted
        'command_inout_asynch',
        'command_inout_reply',
        'contains',
        'disable',
        'enable',
        # 'get_device',  # Needs to be adapted
        'get_device_list',
        'get_fully_qualified_name',
        # 'get_group',   # Needs to be adapted
        'get_name',
        'get_size',
        'is_enabled',
        'name_equals',
        'name_matches',
        'ping',
        'read_attribute_asynch',
        'read_attribute_reply',
        'read_attributes_asynch',
        'read_attributes_reply',
        'remove_all',
        'set_timeout_millis',
        'write_attribute_asynch',
        'write_attribute_reply']

    def proxy_call_define(fname):
        def fn(self, *args, **kwds):
            return getattr(self._Group__group, fname)(*args, **kwds)

        fn.__doc__ = getattr(_RealGroup, fname).__doc__
        setattr(Group, fname, fn)

    for fname in proxy_methods:
        proxy_call_define(fname)

        # Group.add.__func__.__doc__ = _RealGroup.add.__doc__
        # Group.get_group.__func__.__doc__ = _RealGroup.get_group.__doc__
        # Group.__doc__ = _RealGroup.__doc__


def __doc_Group():
    def document_method(method_name, desc, append=True):
        return __document_method(_RealGroup, method_name, desc, append)

    document_method("_add", """
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
    """)

    document_method("_remove", """
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
    """)

    document_method("remove_all", """
    remove_all(self) -> None

        Removes all elements in the _RealGroup. After such a call, the _RealGroup is empty.
    """)

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
    """)

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
    """)

    document_method("get_device_list", """
    get_device_list(self, forward=True) -> sequence<str>

            Considering the following hierarchy:

            ::

                g2.add("my/device/04")
                g2.add("my/device/05")

                g4.add("my/device/08")
                g4.add("my/device/09")

                g3.add("my/device/06")
                g3.add(g4)
                g3.add("my/device/07")

                g1.add("my/device/01")
                g1.add(g2)
                g1.add("my/device/03")
                g1.add(g3)
                g1.add("my/device/02")

            The returned vector content depends on the value of the forward option.
            If set to true, the results will be organized as follows:

            ::

                    dl = g1.get_device_list(True)

                dl[0] contains "my/device/01" which belongs to g1
                dl[1] contains "my/device/04" which belongs to g1.g2
                dl[2] contains "my/device/05" which belongs to g1.g2
                dl[3] contains "my/device/03" which belongs to g1
                dl[4] contains "my/device/06" which belongs to g1.g3
                dl[5] contains "my/device/08" which belongs to g1.g3.g4
                dl[6] contains "my/device/09" which belongs to g1.g3.g4
                dl[7] contains "my/device/07" which belongs to g1.g3
                dl[8] contains "my/device/02" which belongs to g1

            If the forward option is set to false, the results are:

            ::

                    dl = g1.get_device_list(False);

                dl[0] contains "my/device/01" which belongs to g1
                dl[1] contains "my/device/03" which belongs to g1
                dl[2] contains "my/device/02" which belongs to g1


        Parameters :
            - forward : (bool) If it is set to true (the default), the request
                        is forwarded to sub-groups. Otherwise, it is only
                        applied to the local set of devices.

        Return     : (sequence<str>) The list of devices currently in the hierarchy.

        Throws     :
    """)

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
    """)

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
    """)

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
    """)

    document_method("command_inout_asynch", """
    command_inout_asynch(self, cmd_name, forget=False, forward=True, reserved=-1 ) -> int
    command_inout_asynch(self, cmd_name, param, forget=False, forward=True, reserved=-1 ) -> int
    command_inout_asynch(self, cmd_name, param_list, forget=False, forward=True, reserved=-1 ) -> int

            Executes a Tango command on each device in the group asynchronously.
            The method sends the request to all devices and returns immediately.
            Pass the returned request id to Group.command_inout_reply() to obtain
            the results.

        Parameters :
            - cmd_name   : (str) Command name
            - param      : (any) parameter value
            - param_list : (tango.DeviceDataList) sequence of parameters.
                           When given, it's length must match the group size.
            - forget     : (bool) Fire and forget flag. If set to true, it means that
                           no reply is expected (i.e. the caller does not care
                           about it and will not even try to get it)
            - forward    : (bool) If it is set to true (the default) request is
                            forwarded to subgroups. Otherwise, it is only applied
                            to the local set of devices.
            - reserved : (int) is reserved for internal purpose and should not be
                            used. This parameter may disappear in a near future.

        Return     : (int) request id. Pass the returned request id to
                    Group.command_inout_reply() to obtain the results.

        Throws     :
    """)

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
    """)

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
    """)

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
    """)

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
    """)

    document_method("read_attributes_reply", """
    read_attributes_reply(self, req_id, timeout_ms=0 ) -> sequence<GroupAttrReply>

            Returns the results of an asynchronous attribute reading.

        Parameters :
            - req_id     : (int) a request identifier previously returned by read_attribute_asynch.
            - timeout_ms : (int) For each device in the hierarchy, if the attribute
                           value is not yet available, read_attribute_reply
                           ait timeout_ms milliseconds before throwing an
                           exception. This exception will be part of the
                           global reply. If timeout_ms is set to 0,
                           read_attributes_reply waits "indefinitely".

        Return     : (sequence<GroupAttrReply>)

        Throws     :
    """)

    document_method("write_attribute_asynch", """
    write_attribute_asynch(self, attr_name, value, forward=True, multi=False ) -> int

            Writes an attribute on each device in the group asynchronously.
            The method sends the request to all devices and returns immediately.

        Parameters :
            - attr_name : (str) Name of the attribute to write.
            - value     : (any) Value to write. See DeviceProxy.write_attribute
            - forward   : (bool) If it is set to true (the default) request is
                          forwarded to subgroups. Otherwise, it is only applied
                          to the local set of devices.
            - multi     : (bool) If it is set to false (the default), the same
                          value is applied to all devices in the group.
                          Otherwise the value is interpreted as a sequence of
                          values, and each value is applied to the corresponding
                          device in the group. In this case len(value) must be
                          equal to group.get_size()!

        Return     : (int) request id. Pass the returned request id to
                    Group.write_attribute_reply() to obtain the acknowledgements.

        Throws     :
    """)

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
    """)

    document_method("get_name", """
        Get the name of the group. Eg: Group('name').get_name() == 'name'
    """)
    document_method("get_fully_qualified_name", """
        Get the complete (dpt-separated) name of the group. This takes into
        consideration the name of the group and its parents.
    """)
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
    """)

    def document_group_method(method_name, desc, append=True):
        return __document_method(Group, method_name, desc, append)

    document_group_method("add", _RealGroup._add.__doc__, False)
    document_group_method("add", """
    add(self, subgroup, timeout_ms=-1) -> None

            Attaches a (sub)_RealGroup.

            To remove the subgroup use the remove() method.

        Parameters :
            - subgroup   : (str)
            - timeout_ms : (int) If timeout_ms parameter is different from -1,
                            the client side timeout associated to each device
                            composing the _RealGroup added is set to timeout_ms
                            milliseconds. If timeout_ms is -1, timeouts are
                            not changed.
        Return     : None

        Throws     : TypeError, ArgumentError
    """)

    document_group_method("command_inout", """
    command_inout(self, cmd_name, forward=True) -> sequence<GroupCmdReply>
    command_inout(self, cmd_name, param, forward=True) -> sequence<GroupCmdReply>
    command_inout(self, cmd_name, param_list, forward=True) -> sequence<GroupCmdReply>

            Just a shortcut to do:
                self.command_inout_reply(self.command_inout_asynch(...))

        Parameters:
            - cmd_name   : (str) Command name
            - param      : (any) parameter value
            - param_list : (tango.DeviceDataList) sequence of parameters.
                           When given, it's length must match the group size.
            - forward    : (bool) If it is set to true (the default) request is
                            forwarded to subgroups. Otherwise, it is only applied
                            to the local set of devices.

        Return : (sequence<GroupCmdReply>)
    """)

    document_group_method("read_attribute", """
    read_attribute(self, attr_name, forward=True) -> sequence<GroupAttrReply>

            Just a shortcut to do:
                self.read_attribute_reply(self.read_attribute_asynch(...))
    """)

    document_group_method("read_attributes", """
    read_attributes(self, attr_names, forward=True) -> sequence<GroupAttrReply>

            Just a shortcut to do:
                self.read_attributes_reply(self.read_attributes_asynch(...))
    """)

    document_group_method("write_attribute", """
    write_attribute(self, attr_name, value, forward=True, multi=False) -> sequence<GroupReply>

            Just a shortcut to do:
                self.write_attribute_reply(self.write_attribute_asynch(...))
    """)


def group_init(doc=True):
    if doc:
        __doc_Group()
    __init_proxy_Group()
