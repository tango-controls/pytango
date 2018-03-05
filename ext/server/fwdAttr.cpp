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

void export_user_default_fwdattr_prop(py::module &m) {
    py::class_<Tango::UserDefaultFwdAttrProp>(m, "UserDefaultFwdAttrProp")
        .def("set_label", &Tango::UserDefaultFwdAttrProp::set_label)
    ;
}

void export_fwdattr(py::module &m) {
    py::class_<Tango::FwdAttr>(m, "FwdAttr")
        .def(py::init<const std::string&, const std::string&>())
        .def("set_default_properties", &Tango::FwdAttr::set_default_properties)
    ;
}

