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

void export_device_attribute_config(py::module &m) {
    py::class_<Tango::DeviceAttributeConfig>(m, "DeviceAttributeConfig")
//        .def(py::init<Tango::DeviceAttributeConfig&>())
        .def_readwrite("name", &Tango::DeviceAttributeConfig::name)
        .def_readwrite("writable", &Tango::DeviceAttributeConfig::writable)
        .def_readwrite("data_format", &Tango::DeviceAttributeConfig::data_format)
        .def_readwrite("data_type", &Tango::DeviceAttributeConfig::data_type)
        .def_readwrite("max_dim_x", &Tango::DeviceAttributeConfig::max_dim_x)
        .def_readwrite("max_dim_y", &Tango::DeviceAttributeConfig::max_dim_y)
        .def_readwrite("description", &Tango::DeviceAttributeConfig::description)
        .def_property("label", [](Tango::DeviceAttributeConfig& self) {
                return self.label;
            }, [](Tango::DeviceAttributeConfig& self, std::string& label) {
                self.label = label;
            })
        .def_readwrite("unit", &Tango::DeviceAttributeConfig::unit)
        .def_readwrite("standard_unit", &Tango::DeviceAttributeConfig::standard_unit)
        .def_readwrite("display_unit", &Tango::DeviceAttributeConfig::display_unit)
        .def_readwrite("format", &Tango::DeviceAttributeConfig::format)
        .def_readwrite("min_value", &Tango::DeviceAttributeConfig::min_value)
        .def_readwrite("max_value", &Tango::DeviceAttributeConfig::max_value)
        .def_readwrite("min_alarm", &Tango::DeviceAttributeConfig::min_alarm)
        .def_readwrite("max_alarm", &Tango::DeviceAttributeConfig::max_alarm)
        .def_readwrite("writable_attr_name", &Tango::DeviceAttributeConfig::writable_attr_name)

        .def_property("extensions", [](Tango::DeviceAttributeConfig& self) -> py::list {
            py::list py_list;
            for(auto& item : self.extensions)
                py_list.append(item);
            return py_list;
        },[](Tango::DeviceAttributeConfig& self, py::list py_list) -> void {
            for(auto& item : py_list)
                self.extensions.push_back(item.cast<std::string>());
        })

        .def(py::pickle(
            [](const Tango::DeviceAttributeConfig &p) { //__getstate__
                return py::make_tuple(p.name,
                        p.writable,
                        p.data_format,
                        p.data_type,
                        p.max_dim_x,
                        p.max_dim_y,
                        p.description,
                        p.label,
                        p.unit,
                        p.standard_unit,
                        p.display_unit,
                        p.format,
                        p.min_value,
                        p.max_value,
                        p.min_alarm,
                        p.max_alarm,
                        p.writable_attr_name,
                        p.extensions);
            },
            [](py::tuple t) { //__setstate__
                if (t.size() != 19)
                    throw std::runtime_error("Invalid state!");
                Tango::DeviceAttributeConfig p = Tango::DeviceAttributeConfig();
                p.name = t[0].cast<std::string>();
                p.writable = t[1].cast<Tango::AttrWriteType>();
                p.data_format = t[2].cast<Tango::AttrDataFormat>();
                p.data_type = t[3].cast<int>();
                p.max_dim_x = t[4].cast<int>();
                p.max_dim_y = t[5].cast<int>();
                p.description = t[6].cast<std::string>();
                p.label = t[7].cast<std::string>();
                p.unit = t[8].cast<std::string>();
                p.standard_unit = t[9].cast<std::string>();
                p.display_unit = t[10].cast<std::string>();
                p.format = t[11].cast<std::string>();
                p.min_value = t[12].cast<std::string>();
                p.max_value = t[13].cast<std::string>();
                p.min_alarm = t[14].cast<std::string>();
                p.max_alarm = t[15].cast<std::string>();
                p.writable_attr_name = t[16].cast<std::string>();
                p.extensions = t[17].cast<std::vector<std::string>>();
                return p;
            }
        ));
    ;
}
