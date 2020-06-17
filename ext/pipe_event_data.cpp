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

namespace PyPipeEventData
{
static boost::shared_ptr<Tango::PipeEventData> makePipeEventData()
{
    Tango::PipeEventData *result = new Tango::PipeEventData;
    return boost::shared_ptr<Tango::PipeEventData>(result);
}

static void set_errors(Tango::PipeEventData &event_data,
                       boost::python::object &dev_failed)
{
    Tango::DevFailed df;
    boost::python::object errors = dev_failed.attr("args");
    sequencePyDevError_2_DevErrorList(errors.ptr(), event_data.errors);
}

}; // end PyPipeEventData

void export_pipe_event_data()
{
    class_<Tango::PipeEventData>("PipeEventData",
        init<const Tango::PipeEventData &>())

        .def("__init__", boost::python::make_constructor(PyPipeEventData::makePipeEventData))
        // The original Tango::PipeEventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::PipeEventData::device directly.
        // See callback.cpp
        .setattr("device",object())
        .def_readwrite("pipe_name", &Tango::PipeEventData::pipe_name)
        .def_readwrite("event", &Tango::PipeEventData::event)

        .setattr("pipe_value",object())

        .def_readwrite("err", &Tango::PipeEventData::err)
        .def_readwrite("reception_date", &Tango::PipeEventData::reception_date)
        .add_property("errors",
		      make_getter(&Tango::PipeEventData::errors,
				  return_value_policy<copy_non_const_reference>()),
		      &PyPipeEventData::set_errors)

        .def("get_date", &Tango::PipeEventData::get_date,
            return_internal_reference<>())
    ;

}
