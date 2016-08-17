/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include <tango.h>

using namespace boost::python;

void export_attribute_alarm_info()
{
    class_<Tango::AttributeAlarmInfo>("AttributeAlarmInfo")
        .enable_pickling()
        .def_readwrite("min_alarm", &Tango::AttributeAlarmInfo::min_alarm)
        .def_readwrite("max_alarm", &Tango::AttributeAlarmInfo::max_alarm)
        .def_readwrite("min_warning", &Tango::AttributeAlarmInfo::min_warning)
        .def_readwrite("max_warning", &Tango::AttributeAlarmInfo::max_warning)
        .def_readwrite("delta_t", &Tango::AttributeAlarmInfo::delta_t)
        .def_readwrite("delta_val", &Tango::AttributeAlarmInfo::delta_val)
        .def_readwrite("extensions", &Tango::AttributeAlarmInfo::extensions)
    ;
}
