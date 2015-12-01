/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include <tango.h>

using namespace boost::python;

void export_attribute_info_ex()
{
    class_<Tango::AttributeInfoEx, bases<Tango::AttributeInfo> >
        ("AttributeInfoEx")
        .def(init<const Tango::AttributeInfoEx&>())
        .enable_pickling()
	.def_readwrite("root_attr_name",  &Tango::AttributeInfoEx::root_attr_name)
	.def_readwrite("memorized",  &Tango::AttributeInfoEx::memorized)
	.def_readwrite("enum_labels",  &Tango::AttributeInfoEx::enum_labels)
        .def_readwrite("alarms", &Tango::AttributeInfoEx::alarms)
        .def_readwrite("events", &Tango::AttributeInfoEx::events)
        .def_readwrite("sys_extensions", &Tango::AttributeInfoEx::sys_extensions)
    ;
}
