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
#include <defs.h>

namespace py = pybind11;

void export_group_reply(py::module &m) {
    py::class_<Tango::GroupReply>(m, "GroupReply")
        .def("has_failed", &Tango::GroupReply::has_failed)
        .def("group_element_enabled", &Tango::GroupReply::group_element_enabled)
        .def("dev_name", &Tango::GroupReply::dev_name, py::return_value_policy::copy)
        .def("obj_name", &Tango::GroupReply::obj_name, py::return_value_policy::copy)
        .def("get_err_stack", &Tango::GroupReply::get_err_stack, py::return_value_policy::copy)
    ;

    py::class_<Tango::GroupCmdReply, Tango::GroupReply>(m, "GroupCmdReply")
        .def("get_data_raw", &Tango::GroupCmdReply::get_data, py::return_value_policy::reference_internal)
    ;

    py::class_<Tango::GroupAttrReply, Tango::GroupReply>(m, "GroupAttrReply")
//        .def("__get_data", [](Tango::GroupAttrReply& self) -> py::object {
//                // Usually we pass a device_proxy to "convert_to_python" in order to
//                // get the data_format of the DeviceAttribute for Tango versions
//                // older than 7.0. However, GroupAttrReply has no device_proxy to use!
//                // So, we are using update_data_format() in:
//                //       GroupElement::read_attribute_reply/read_attributes_reply
////                return PyDeviceAttribute::convert_to_python(
////                return new Tango::DeviceAttribute(self.get_data());
//                return (py::object)nullptr;
//        })
    ;
}
