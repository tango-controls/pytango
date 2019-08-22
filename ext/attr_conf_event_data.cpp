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

void export_attr_conf_event_data(py::module &m) {
    py::class_<Tango::AttrConfEventData, std::shared_ptr<Tango::AttrConfEventData>>(m, "AttrConfEventData")
        .def(py::init<const Tango::AttrConfEventData &>())
        .def(py::init([](){
            return std::shared_ptr<Tango::AttrConfEventData>(new Tango::AttrConfEventData);
        }))

        // The original Tango::AttrConfEventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::AttrConfEventData::device directly.
        // See callback.cpp
//        .setattr("device",object())

        .def_readwrite("attr_name", &Tango::AttrConfEventData::attr_name)
        .def_readwrite("event", &Tango::AttrConfEventData::event)

//        .setattr("attr_conf",object())

        .def_readwrite("err", &Tango::AttrConfEventData::err)
        .def_readwrite("reception_date", &Tango::AttrConfEventData::reception_date)
        .def_property("errors", [](){
            return &Tango::DataReadyEventData::errors;
        },[](Tango::AttrConfEventData &event_data, py::object &dev_failed) {
                py::object errors = dev_failed.attr("args");
                sequencePyDevError_2_DevErrorList(errors, event_data.errors);
        })

        .def("get_date", &Tango::AttrConfEventData::get_date,
            py::return_value_policy::reference)
    ;
}

