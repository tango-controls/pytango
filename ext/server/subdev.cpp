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

void export_sub_dev_diag(py::module &m) {
    py::class_<Tango::SubDevDiag>(m, "SubDevDiag")
        .def("set_associated_device", [](Tango::SubDevDiag& self, std::string&  dev_name) -> void {
            self.set_associated_device(dev_name);
        })
        .def("get_associated_device", [](Tango::SubDevDiag& self) -> std::string {
            return self.get_associated_device();
        })
        .def("register_sub_device", [](Tango::SubDevDiag& self, std::string& dev_name, std::string& sub_dev_name) -> void {
            return self.register_sub_device(dev_name, sub_dev_name);
        })
        .def("remove_sub_devices", [](Tango::SubDevDiag& self, std::string& dev_name) -> void {
            self.remove_sub_devices(dev_name);
        })
        .def("remove_sub_devices", [](Tango::SubDevDiag& self) -> void {
            self.remove_sub_devices();
        })
        .def("get_sub_devices", [](Tango::SubDevDiag& self) {
            Tango::DevVarStringArray *sub_devs = self.get_sub_devices();
            py::list py_sub_devs;
            for(unsigned i = 0; i < sub_devs->length(); ++i) {
                py_sub_devs.append((*sub_devs)[i].in());
            }
            delete sub_devs;
            return py_sub_devs;
        })
        .def("store_sub_devices", [](Tango::SubDevDiag& self) -> void {
            self.store_sub_devices();
        })
        .def("get_sub_devices_from_cache", [](Tango::SubDevDiag& self) -> void {
            self.get_sub_devices_from_cache();
        })
    ;
}
