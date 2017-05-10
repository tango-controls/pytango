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

namespace PyDevIntrChangeEventData
{
static boost::shared_ptr<Tango::DevIntrChangeEventData> makeDevIntrChangeEventData()
{
    Tango::DevIntrChangeEventData *result = new Tango::DevIntrChangeEventData;
    return boost::shared_ptr<Tango::DevIntrChangeEventData>(result);
}

static void set_errors(Tango::DevIntrChangeEventData &event_data,
                       boost::python::object &dev_failed)
{
    Tango::DevFailed df;
    boost::python::object errors = dev_failed.attr("args");
    sequencePyDevError_2_DevErrorList(errors.ptr(), event_data.errors);
}

}; // end PyDevIntrChangeEventData

void export_devintr_change_event_data()
{
    class_<Tango::DevIntrChangeEventData>("DevIntrChangeEventData",
        init<const Tango::DevIntrChangeEventData &>())

        .def("__init__", boost::python::make_constructor(PyDevIntrChangeEventData::makeDevIntrChangeEventData))
        // The original Tango::DevIntrChangeEventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::DevIntrChangeEventData::device directly.
        // See callback.cpp
        .setattr("device",object())
        .def_readwrite("event", &Tango::DevIntrChangeEventData::event)
        .def_readwrite("device_name", &Tango::DevIntrChangeEventData::device_name)

        .setattr("cmd_list", object())
        .setattr("att_list", object())
        .def_readwrite("dev_started", &Tango::DevIntrChangeEventData::dev_started)

        .def_readwrite("err", &Tango::DevIntrChangeEventData::err)
        .def_readwrite("reception_date", &Tango::DevIntrChangeEventData::reception_date)

		.def_readwrite("err", &Tango::DevIntrChangeEventData::err)
        .add_property("errors",
		      make_getter(&Tango::DevIntrChangeEventData::errors,
				  return_value_policy<copy_non_const_reference>()),
		      &PyDevIntrChangeEventData::set_errors)

        .def("get_date", &Tango::DevIntrChangeEventData::get_date,
            return_internal_reference<>())
    ;

}
