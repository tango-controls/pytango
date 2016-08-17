/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "pytgutils.h"

void export_device_attribute_config()
{
    bopy::class_<Tango::DeviceAttributeConfig>("DeviceAttributeConfig")
        .def(bopy::init<const Tango::DeviceAttributeConfig&>())

        .enable_pickling()
//        .def_pickle(PyDeviceAttributeConfig::PickleSuite())

        .def_readwrite("name", &Tango::DeviceAttributeConfig::name)
        .def_readwrite("writable", &Tango::DeviceAttributeConfig::writable)
        .def_readwrite("data_format", &Tango::DeviceAttributeConfig::data_format)
        .def_readwrite("data_type", &Tango::DeviceAttributeConfig::data_type)
        .def_readwrite("max_dim_x", &Tango::DeviceAttributeConfig::max_dim_x)
        .def_readwrite("max_dim_y", &Tango::DeviceAttributeConfig::max_dim_y)
        .def_readwrite("description", &Tango::DeviceAttributeConfig::description)
        //.def_readwrite("label", &Tango::DeviceAttributeConfig::label)
        .add_property("label", bopy::make_getter(&Tango::DeviceAttributeConfig::label,
                                                 bopy::return_value_policy<bopy::return_by_value>()),
                               bopy::make_setter(&Tango::DeviceAttributeConfig::label,
                                                 bopy::return_value_policy<bopy::return_by_value>()))
        .def_readwrite("unit", &Tango::DeviceAttributeConfig::unit)
        .def_readwrite("standard_unit", &Tango::DeviceAttributeConfig::standard_unit)
        .def_readwrite("display_unit", &Tango::DeviceAttributeConfig::display_unit)
        .def_readwrite("format", &Tango::DeviceAttributeConfig::format)
        .def_readwrite("min_value", &Tango::DeviceAttributeConfig::min_value)
        .def_readwrite("max_value", &Tango::DeviceAttributeConfig::max_value)
        .def_readwrite("min_alarm", &Tango::DeviceAttributeConfig::min_alarm)
        .def_readwrite("max_alarm", &Tango::DeviceAttributeConfig::max_alarm)
        .def_readwrite("writable_attr_name", &Tango::DeviceAttributeConfig::writable_attr_name)
        .def_readwrite("extensions", &Tango::DeviceAttributeConfig::extensions)
    ;
}
