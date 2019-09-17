/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>
#include <pyutils.h>
#include <exception.h>

namespace py = pybind11;

void export_pipe_event_data(py::module &m) {
    py::class_<Tango::PipeEventData, std::shared_ptr<Tango::PipeEventData>>(m, "PipeEventData")
        .def(py::init<const Tango::PipeEventData &>())
        .def(py::init([](){
            return std::shared_ptr<Tango::PipeEventData>(new Tango::PipeEventData);
        }))

        // The original Tango::PipeEventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::PipeEventData::device directly.
        // See callback.cpp
        //.setattr("device",object())

        .def_readwrite("pipe_name", &Tango::PipeEventData::pipe_name)
        .def_readwrite("event", &Tango::PipeEventData::event)

//        .setattr("pipe_value",object())

        .def_readwrite("err", &Tango::PipeEventData::err)
        .def_readwrite("reception_date", &Tango::PipeEventData::reception_date)
        .def_property("errors", [](){
            return &Tango::PipeEventData::errors;
        },[](Tango::PipeEventData &event_data, py::object& dev_failed) {
            py::object errors = dev_failed.attr("args");
            sequencePyDevError_2_DevErrorList(errors, event_data.errors);
        })

        .def("get_date", &Tango::PipeEventData::get_date,
            py::return_value_policy::reference)
    ;

}
