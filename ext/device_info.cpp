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

void export_device_info(py::module &m) {
    py::class_<Tango::DeviceInfo>(m, "DeviceInfo")
        .def_readonly("dev_class", &Tango::DeviceInfo::dev_class)
        .def_readonly("server_id", &Tango::DeviceInfo::server_id)
        .def_readonly("server_host", &Tango::DeviceInfo::server_host)
        .def_readonly("server_version", &Tango::DeviceInfo::server_version)
        .def_readonly("doc_url", &Tango::DeviceInfo::doc_url)
        .def_readonly("dev_type", &Tango::DeviceInfo::dev_type)
    ;
}
