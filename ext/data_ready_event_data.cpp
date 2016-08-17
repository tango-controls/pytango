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

#include "exception.h"

using namespace boost::python;

extern boost::python::object PyTango_DevFailed;

struct PyDataReadyEventData
{
    static inline Tango::DeviceProxy* get_device(Tango::DataReadyEventData &self)
    {
        return self.device;
    }

    static boost::shared_ptr<Tango::DataReadyEventData> makeDataReadyEventData()
    {
        Tango::DataReadyEventData *result = new Tango::DataReadyEventData;
        return boost::shared_ptr<Tango::DataReadyEventData>(result);
    }

    static void set_errors(Tango::DataReadyEventData &event_data, 
	                   boost::python::object &dev_failed)
    {
        Tango::DevFailed df;
        boost::python::object errors = dev_failed.attr("args");
        sequencePyDevError_2_DevErrorList(errors.ptr(), event_data.errors);
    }
};

void export_data_ready_event_data()
{
    class_<Tango::DataReadyEventData>("DataReadyEventData",
        init<const Tango::DataReadyEventData &>())

        .def("__init__", boost::python::make_constructor(PyDataReadyEventData::makeDataReadyEventData))

        // The original Tango::EventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::EventData::device directly.
        // See callback.cpp
        .setattr("device",object())
        .def_readwrite("attr_name", &Tango::DataReadyEventData::attr_name)
        .def_readwrite("event", &Tango::DataReadyEventData::event)
        .def_readwrite("attr_data_type", &Tango::DataReadyEventData::attr_data_type)
        .def_readwrite("ctr", &Tango::DataReadyEventData::ctr)
        .def_readwrite("err", &Tango::DataReadyEventData::err)
        .def_readwrite("reception_date", &Tango::DataReadyEventData::reception_date)
        .add_property("errors", 
		      make_getter(&Tango::DataReadyEventData::errors, 
				  return_value_policy<copy_non_const_reference>()),
		      &PyDataReadyEventData::set_errors)

        .def("get_date", &Tango::DataReadyEventData::get_date,
            return_internal_reference<>())
    ;
}
