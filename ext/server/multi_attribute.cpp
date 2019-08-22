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

void export_multi_attribute(py::module& m) {
    py::class_<Tango::MultiAttribute>(m, "MultiAttribute")

        .def("get_attr_by_name", [](Tango::MultiAttribute& self, const char *attr_name) -> Tango::Attribute& {
                return self.get_attr_by_name(attr_name);
            }, py::return_value_policy::reference)

        .def("get_attr_by_ind", [](Tango::MultiAttribute& self, const long index) -> Tango::Attribute& {
                return self.get_attr_by_ind(index);
            }, py::return_value_policy::reference)

        .def("get_w_attr_by_name", [](Tango::MultiAttribute& self, const char *attr_name) -> Tango::WAttribute& {
                return self.get_w_attr_by_name(attr_name);
            }, py::return_value_policy::reference)

        .def("get_w_attr_by_ind", [](Tango::MultiAttribute& self, const long index) -> Tango::WAttribute& {
                return self.get_w_attr_by_ind(index);
            }, py::return_value_policy::reference)

        .def("get_attr_ind_by_name", &Tango::MultiAttribute::get_attr_ind_by_name)

        .def("get_alarm_list", [](Tango::MultiAttribute& self) -> std::vector<long>& {
                return self.get_alarm_list();
            }, py::return_value_policy::reference)

        .def("get_attr_nb", &Tango::MultiAttribute::get_attr_nb)

        .def("check_alarm", (bool (Tango::MultiAttribute::*) ())
            &Tango::MultiAttribute::check_alarm)

        .def("check_alarm", (bool (Tango::MultiAttribute::*) (const long))
            &Tango::MultiAttribute::check_alarm)

        .def("check_alarm", (bool (Tango::MultiAttribute::*) (const char *))
            &Tango::MultiAttribute::check_alarm)

        .def("read_alarm", (void (Tango::MultiAttribute::*) (const std::string& ))
            &Tango::MultiAttribute::read_alarm)

        .def("get_attribute_list", [](Tango::MultiAttribute& self) -> std::vector<Tango::Attribute*>& {
                return self.get_attribute_list();
            }, py::return_value_policy::reference)
    ;
}
