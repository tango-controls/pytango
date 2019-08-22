/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>
#include <pyutils.h>
#include <exception.h>

namespace py = pybind11;

void export_event_data(py::module &m) {
    py::class_<Tango::EventData, std::shared_ptr<Tango::EventData>>(m, "EventData", py::dynamic_attr())
        .def(py::init<Tango::EventData &>())
        .def(py::init([](){
            Tango::EventData *result = new Tango::EventData;
            result->attr_value = new Tango::DeviceAttribute();
            return std::shared_ptr<Tango::EventData>(result);
        }))

        // The original Tango::EventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::EventData::device directly.
        // See callback.cpp
//        .setattr("device",object())

        .def_readwrite("attr_name", &Tango::EventData::attr_name)
        .def_readwrite("event", &Tango::EventData::event)

        // The original Tango::EventData structure has "get_attr_value" but
        // we can't refer it directly here because we have to extract value
        // and so on.
        // See callback.cpp
//        .setattr("attr_value",object())

        .def_readwrite("err", &Tango::EventData::err)
        .def_readwrite("reception_date", &Tango::EventData::reception_date)
        .def_property("errors", [](){
            return &Tango::EventData::errors;
        },[](Tango::EventData &event_data, py::object &dev_failed) {
            py::object errors = dev_failed.attr("args");
            sequencePyDevError_2_DevErrorList(errors, event_data.errors);
        })

        .def("get_date", &Tango::EventData::get_date,
            py::return_value_policy::reference)
    ;
}
