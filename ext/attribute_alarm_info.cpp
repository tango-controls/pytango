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


void export_attribute_alarm_info(py::module &m) {
    py::class_<Tango::AttributeAlarmInfo>(m, "AttributeAlarmInfo")
        .def_readwrite("min_alarm", &Tango::AttributeAlarmInfo::min_alarm)
        .def_readwrite("max_alarm", &Tango::AttributeAlarmInfo::max_alarm)
        .def_readwrite("min_warning", &Tango::AttributeAlarmInfo::min_warning)
        .def_readwrite("max_warning", &Tango::AttributeAlarmInfo::max_warning)
        .def_readwrite("delta_t", &Tango::AttributeAlarmInfo::delta_t)
        .def_readwrite("delta_val", &Tango::AttributeAlarmInfo::delta_val)
        .def_property("extensions", [](Tango::AttributeAlarmInfo& self) -> py::list {
            py::list py_list;
            for(auto& item : self.extensions)
                py_list.append(item);
            return py_list;
        },[](Tango::AttributeAlarmInfo& self, py::list py_list) -> void {
            for(auto& item : py_list)
                self.extensions.push_back(item.cast<std::string>());
        })
        .def(py::pickle(
            [](const Tango::AttributeAlarmInfo &p) { //__getstate__
                return py::make_tuple(p.min_alarm,
                        p.max_alarm,
                        p.min_warning,
                        p.max_warning,
                        p.delta_t,
                        p.delta_val,
                        p.extensions);
            },
            [](py::tuple t) { //__setstate__
                if (t.size() != 7)
                    throw std::runtime_error("Invalid state!");
                Tango::AttributeAlarmInfo p = Tango::AttributeAlarmInfo();
                p.min_alarm = t[0].cast<std::string>();
                p.max_alarm = t[1].cast<std::string>();
                p.min_warning = t[2].cast<std::string>();
                p.max_warning = t[3].cast<std::string>();
                p.delta_t = t[4].cast<std::string>();
                p.delta_val = t[5].cast<std::string>();
                p.extensions = t[6].cast<std::vector<std::string> >();
                return p;
            }
        ));
    ;
}
