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

void export_attribute_info_ex(py::module &m) {
    py::class_<Tango::AttributeInfoEx, Tango::AttributeInfo>(m, "AttributeInfoEx")
        .def(py::init<const Tango::AttributeInfoEx&>())
        .def_readwrite("root_attr_name", &Tango::AttributeInfoEx::root_attr_name)
        .def_readwrite("memorized", &Tango::AttributeInfoEx::memorized)
        .def_readwrite("enum_labels", &Tango::AttributeInfoEx::enum_labels)

        .def_property("enum_labels", [](Tango::AttributeInfoEx& self) -> py::list {
            py::list py_list;
            for(auto& item : self.enum_labels)
                py_list.append(item);
            return py_list;
        },[](Tango::AttributeInfoEx& self, py::list py_list) -> void {
            for(auto& item : py_list)
                self.enum_labels.push_back(item.cast<std::string>());
        })

        .def_readwrite("alarms", &Tango::AttributeInfoEx::alarms)
        .def_readwrite("events", &Tango::AttributeInfoEx::events)

        .def_property("sys_extensions", [](Tango::AttributeInfoEx& self) -> py::list {
            py::list py_list;
            for(auto& item : self.sys_extensions)
                py_list.append(item);
            return py_list;
        },[](Tango::AttributeInfoEx& self, py::list py_list) -> void {
            for(auto& item : py_list)
                self.sys_extensions.push_back(item.cast<std::string>());
        })

        .def(py::pickle(
            [](const Tango::AttributeInfoEx &p) { //__getstate__
                return py::make_tuple(p.root_attr_name,
                    p.memorized,
                    p.enum_labels,
                    p.alarms,
                    p.events,
                    p.sys_extensions);
            },
            [](py::tuple t) { //__setstate__
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state!");
                Tango::AttributeInfoEx p = Tango::AttributeInfoEx();
                p.root_attr_name = t[0].cast<std::string>();
                p.memorized = t[1].cast<Tango::AttrMemorizedType>();
                p.enum_labels = t[2].cast<std::vector<std::string>>();
                p.alarms = t[3].cast<Tango::AttributeAlarmInfo>();
                p.events = t[4].cast<Tango::AttributeEventInfo>();
                p.sys_extensions = t[5].cast<std::vector<std::string>>();
                return p;
            }
        ));
    ;
}
