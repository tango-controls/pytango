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
#include <pybind11/stl.h>

namespace py = pybind11;

void export_multi_class_attribute(py::module& m) {
    py::class_<Tango::MultiClassAttribute>(m, "MultiClassAttribute")

        .def("get_attr", [](Tango::MultiClassAttribute& self, std::string&  attr_name) -> Tango::Attr& {
            return self.get_attr(attr_name);
        }, py::return_value_policy::reference)

        .def("remove_attr", [](Tango::MultiClassAttribute& self, const std::string& attr_name, const std::string& cl_name) -> void {
            self.remove_attr(attr_name, cl_name);
        })

        .def("get_attr_list", [](Tango::MultiClassAttribute& self) -> std::vector<Tango::Attr *>& {
            return self.get_attr_list();
        }, py::return_value_policy::reference)
    ;
}
