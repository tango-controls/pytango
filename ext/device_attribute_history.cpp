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

namespace py = pybind11;

void export_device_attribute(py::module &m);

void export_device_attribute_history(py::module &m) {
    py::class_<Tango::DeviceAttributeHistory, Tango::DeviceAttribute>(m, "DeviceAttributeHistory")
        .def(py::init<>())
        .def(py::init<const Tango::DeviceAttributeHistory &>())
        .def("has_failed", &Tango::DeviceAttributeHistory::has_failed)
    ;
}
