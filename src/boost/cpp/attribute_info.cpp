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

void export_attribute_info()
{
    class_<Tango::AttributeInfo, bases<Tango::DeviceAttributeConfig> >
        ("AttributeInfo")
        .def(init<const Tango::AttributeInfo&>())
        .enable_pickling()
        .def_readwrite("disp_level", &Tango::AttributeInfo::disp_level)
    ;
}
