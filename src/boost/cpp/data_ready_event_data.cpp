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

struct PyDataReadyEventData
{
    static inline Tango::DeviceProxy* get_device(Tango::DataReadyEventData &self)
    {
        return self.device;
    }
};

void export_data_ready_event_data()
{
    class_<Tango::DataReadyEventData>("DataReadyEventData",
        init<const Tango::DataReadyEventData &>())

        // The original Tango::EventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::EventData::device directly.
        // See callback.cpp
        .setattr("device",object())
        .def_readonly("attr_name", &Tango::DataReadyEventData::attr_name)
        .def_readonly("event", &Tango::DataReadyEventData::event)
        .def_readonly("attr_data_type", &Tango::DataReadyEventData::attr_data_type)
        .def_readonly("ctr", &Tango::DataReadyEventData::ctr)
        .def_readonly("err", &Tango::DataReadyEventData::err)
        .def_readonly("reception_date", &Tango::DataReadyEventData::reception_date)
        .add_property("errors", make_getter(&Tango::DataReadyEventData::errors, 
            return_value_policy<copy_non_const_reference>()))
        .def("get_date", &Tango::DataReadyEventData::get_date,
            return_internal_reference<>())
    ;
}
