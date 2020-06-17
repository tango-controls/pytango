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

__all__ = ("get_class", "get_classes", "get_cpp_class", "get_cpp_classes",
           "get_constructed_class", "get_constructed_classes",
           "class_factory", "delete_class_list",
           "class_list", "cpp_class_list", "constructed_class")

__docformat__ = "restructuredtext"

# list of tuple<DeviceClass class, DeviceImpl class, tango device class name>
class_list = []

# list of tuple<DeviceClass name, tango device class name>
cpp_class_list = []

# list of DeviceClass objects, one for each registered device class
constructed_class = []


def get_classes():
    global class_list
    return class_list


def get_class(name):
    for klass_info in get_classes():
        if klass_info[2] == name:
            return klass_info
    return None


def get_class_by_class(klass):
    for klass_info in get_classes():
        if klass_info[0] == klass:
            return klass_info
    return None


def get_cpp_classes():
    global cpp_class_list
    return cpp_class_list


def get_cpp_class(name):
    for klass_info in get_cpp_classes():
        if klass_info[1] == name:
            return klass_info
    return None


def get_constructed_classes():
    global constructed_class
    return constructed_class


def get_constructed_class(name):
    for klass in get_constructed_classes():
        if klass.get_name() == name:
            return klass
    return None


def get_constructed_class_by_class(klass):
    for k in get_constructed_classes():
        if k.__class__ == klass:
            return k
    return None


#
# A method to delete Tango classes from Python
#

def delete_class_list():
    global constructed_class
    if len(constructed_class) != 0:
        del (constructed_class[:])


#
# A generic class_factory method
#

def class_factory():
    local_class_list = get_classes()
    local_cpp_class_list = get_cpp_classes()

    if ((len(local_class_list) + len(local_cpp_class_list)) == 0):
        print('Oups, no Tango class defined within this device server !!!')
        print('Sorry, but I exit')
        import sys
        sys.exit()

    # Call the delete_class_list function in order to clear the global
    # constructed class Python list. This is necessary only in case of
    # ServerRestart command
    delete_class_list()

    local_constructed_class = get_constructed_classes()
    for class_info in local_class_list:
        device_class_class = class_info[0]
        tango_device_class_name = class_info[2]
        device_class = device_class_class(tango_device_class_name)
        local_constructed_class.append(device_class)
