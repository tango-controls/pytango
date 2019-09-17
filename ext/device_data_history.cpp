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

namespace py = pybind11;

void export_device_data_history(py::module &m) {
    py::class_<Tango::DeviceDataHistory, Tango::DeviceData>(m, "DeviceDataHistory")
        .def(py::init<>())
        .def(py::init<const Tango::DeviceDataHistory &>())
        .def("has_failed", &Tango::DeviceDataHistory::has_failed)
        .def("get_date", &Tango::DeviceDataHistory::get_date,
            py::return_value_policy::reference)
        .def("get_err_stack", &Tango::DeviceDataHistory::get_err_stack,
            py::return_value_policy::copy)
    ;
}
