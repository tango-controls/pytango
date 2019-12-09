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
#include "device_attribute.h"

namespace py = pybind11;

void export_group_reply(py::module &m) {
    py::class_<Tango::GroupReply>(m, "GroupReply")
        .def("has_failed", [](Tango::GroupReply& self) -> bool {
            return self.has_failed(); // Tango C++ signature
        })
        .def("group_element_enabled", [](Tango::GroupReply& self) -> bool {
            return self.group_element_enabled(); // Tango C++ signature
        })
        .def("dev_name", [](Tango::GroupReply& self) -> std::string {
            std::string ret = static_cast<std::string>(self.dev_name()); // Tango C++ signature
            return ret;
        })
        .def("obj_name", [](Tango::GroupReply& self) -> std::string {
            std::string ret = static_cast<std::string>(self.obj_name()); // Tango C++ signature
            return ret;
        })
        .def("get_err_stack", [](Tango::GroupReply& self) -> Tango::DevErrorList {
            return self.get_err_stack(); // Tango C++ signature
        })
    ;

    py::class_<Tango::GroupCmdReply, Tango::GroupReply>(m, "GroupCmdReply")
        .def("get_data_raw", [](Tango::GroupCmdReply& self) -> Tango::DeviceData& {
            return self.get_data(); // Tango C++ signature
        })
    ;

    py::class_<Tango::GroupAttrReply, Tango::GroupReply>(m, "GroupAttrReply")
        .def("__get_data", [](Tango::GroupAttrReply& self) -> py::object {
                // Usually we pass a device_proxy to "convert_to_python" in order to
                // get the data_format of the DeviceAttribute for Tango versions
                // older than 7.0. However, GroupAttrReply has no device_proxy to use!
                // So, we are using update_data_format() in:
                // GroupElement::read_attribute_reply/read_attributes_reply
            return PyDeviceAttribute::convert_to_python(
                new Tango::DeviceAttribute(self.get_data())); // Tango C++ signature
        })
    ;
}
