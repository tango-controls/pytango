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

__all__ = ["group_reply_list_init"]

__docformat__ = "restructuredtext"

from ._PyTango import GroupReplyList, GroupCmdReplyList, GroupAttrReplyList


def __GroupReplyList__getitem(self, item):
    # Accessing an item in GroupReplyList and friends makes a C++ copy
    # of the item calling the copy constructor. But the copy constructor
    # of GroupReply is not fair: It extracts the data from the original
    # object!! So, we will only call the original getitem once, the
    # successive calls will just return the cached object.
    # GroupReplyList could be changed to use proxies to the internal
    # c++ object with the apropriate parameter to the
    # boost::vector_indexing_suite, but when the value is extracted
    # it is not anymore in any C++ object but in the python object, so
    # we must still keep it.
    
    # Slices support
    if isinstance(item, slice):
        return [self.__getitem__(x) for x in range(*item.indices(len(self)))]
    
    # We want to get the same cache value for x[-1] as for x[len(x)-1]
    if item < 0:
        item = item + len(self)
    
    # Try to get from cache
    try:
        return self.__listCache[item]
    except AttributeError:
        # The GroupReplyList object is created without the
        # cache attribute, so it is created here
        self.__listCache = dict()
    except KeyError:
        # The decision wheter we are out of bounds or it's just a cache
        # miss will be taken by original_getitem
        pass

    r = self.__GroupReplyList_original_getitem(item)
    self.__listCache[item] = r
    return r

def __GroupReplyList__iter(self):
    # Same problem as getitem. In this case it is easier for me to just
    # reimplement __iter__ in terms of __getitem__
    for x in range(len(self)):
        yield self[x]

def __init_GroupReplyList():
    GroupReplyList.__GroupReplyList_original_getitem = GroupReplyList.__getitem__
    GroupReplyList.__getitem__ = __GroupReplyList__getitem

    GroupCmdReplyList.__GroupReplyList_original_getitem = GroupCmdReplyList.__getitem__
    GroupCmdReplyList.__getitem__ = __GroupReplyList__getitem

    GroupAttrReplyList.__GroupReplyList_original_getitem = GroupAttrReplyList.__getitem__
    GroupAttrReplyList.__getitem__ = __GroupReplyList__getitem

    GroupReplyList.__iter__ = __GroupReplyList__iter
    GroupCmdReplyList.__iter__ = __GroupReplyList__iter
    GroupAttrReplyList.__iter__ = __GroupReplyList__iter

def __doc_GroupReplyList():
    pass

def group_reply_list_init(doc=True):
    __init_GroupReplyList()
    if doc:
        __doc_GroupReplyList()
