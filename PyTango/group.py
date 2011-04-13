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

__all__ = [ "Group" ]

__docformat__ = "restructuredtext"

import operator
import types

from _PyTango import __Group as _RealGroup, GroupElement
from utils import document_method as __document_method

import group_element

# I define Group as a proxy to __Group, where group is the actual
# C++ Tango::Group object. Most functions just call the __group object
# and are defined dynamically in __init_proxy_Group, also copying it's
# documentation strings.
# The proxy is useful for add(group). In this case the parameter 'group'
# becomes useless. With the proxy we make that parameter come to live
# again before returning.
# The other function that needs to be adapted to this is get_group because
# we want to return a Group, not a __Group!
class Group:
    def __init__(self, name):
        if isinstance(name, str):
            name = _RealGroup(name)
        if not isinstance(name, _RealGroup):
            raise TypeError("Constructor expected receives a str")
        self.__group = name

    def add(self, pattern_subgroup, timeout_ms=-1):
        if isinstance(pattern_subgroup, Group):
            name = pattern_subgroup.__group.get_name()
            self.__group.add(pattern_subgroup.__group, timeout_ms)
            pattern_subgroup.__group = self.get_group(name)
        else:
            self.__group.add(pattern_subgroup, timeout_ms)

    def get_group(self, group_name):
        internal = self.__group.get_group(group_name)
        if internal is None:
            return None
        return Group(internal)

def __init_proxy_Group():
    proxy_methods = [
        #'add',
        'command_inout',
        'command_inout_asynch',
        'command_inout_reply',
        'contains',
        'disable',
        'enable',
        'get_device',
        'get_device_list',
        'get_fully_qualified_name',
        #'get_group',
        'get_name',
        'get_size',
        'is_enabled',
        'name_equals',
        'name_matches',
        'ping',
        'read_attribute',
        'read_attribute_asynch',
        'read_attribute_reply',
        'read_attributes',
        'read_attributes_asynch',
        'read_attributes_reply',
        'remove',
        'remove_all',
        'set_timeout_millis',
        'write_attribute',
        'write_attribute_asynch',
        'write_attribute_reply',
        '__contains__',
        '__delitem__',
        '__getitem__',
        '__len__']
        
    def proxy_call_define(fname):
        def fn(self, *args, **kwds):
            return getattr(self._Group__group, fname)(*args, **kwds)
        fn.__doc__ = getattr(_RealGroup, fname).__doc__
        setattr(Group, fname, fn)

    for fname in proxy_methods:
        proxy_call_define(fname)
    
    Group.add.im_func.__doc__ = _RealGroup.add.__doc__
    Group.get_group.im_func.__doc__ = _RealGroup.get_group.__doc__
    Group.__doc__ = _RealGroup.__doc__


def __doc_Group():
    def document_method(method_name, desc, append=True):
        return __document_method(_RealGroup, method_name, desc, append)

    document_method("add", GroupElement.add.__doc__, False)
    document_method("add", """
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
    """ )

    document_method("remove_all", """
    remove_all(self) -> None
    
        Removes all elements in the _RealGroup. After such a call, the _RealGroup is empty.
    """ )

    # I just documented them in group_element.py ...
    #document_method("enable", """""" )
    #document_method("disable", """""" )

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
    """ )

def init(doc=True):
    group_element.init(doc=doc)
    if doc:
        __doc_Group()
    __init_proxy_Group()
