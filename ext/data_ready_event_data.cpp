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
#include <exception.h>

namespace py = pybind11;

void export_data_ready_event_data(py::module &m) {
    py::class_<Tango::DataReadyEventData, std::shared_ptr<Tango::DataReadyEventData>>(m, "DataReadyEventData")
        .def(py::init<const Tango::DataReadyEventData &>())
        .def(py::init([](){
            Tango::DataReadyEventData *result = new Tango::DataReadyEventData;
            return std::shared_ptr<Tango::DataReadyEventData>(result);
        }))

        // The original Tango::DataReadyEventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::DataReadyEventData::device directly.
        // See callback.cpp
//        .setattr("device",object())

        .def_readwrite("attr_name", &Tango::DataReadyEventData::attr_name)
        .def_readwrite("event", &Tango::DataReadyEventData::event)
        .def_readwrite("attr_data_type", &Tango::DataReadyEventData::attr_data_type)
        .def_readwrite("ctr", &Tango::DataReadyEventData::ctr)
        .def_readwrite("err", &Tango::DataReadyEventData::err)
        .def_readwrite("reception_date", &Tango::DataReadyEventData::reception_date)
        .def_property("errors",
            [](){ //get
                return &Tango::DataReadyEventData::errors;
            },
            [](Tango::DataReadyEventData &event_data, py::object &dev_failed) { // set
                py::object errors = dev_failed.attr("args");
                sequencePyDevError_2_DevErrorList(errors, event_data.errors);
            })

        .def("get_date", &Tango::DataReadyEventData::get_date,
            py::return_value_policy::reference)
    ;
}
