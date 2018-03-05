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

void export_periodic_event_info(py::module &m) {
    py::class_<Tango::PeriodicEventInfo>(m, "PeriodicEventInfo")
        .def_readwrite("period", &Tango::PeriodicEventInfo::period)

        .def_property("extensions", [](Tango::PeriodicEventInfo& self) -> py::list {
            py::list py_list;
            for(auto& item : self.extensions)
                py_list.append(item);
            return py_list;
        },[](Tango::PeriodicEventInfo& self, py::list py_list) -> void {
            for(auto& item : py_list)
                self.extensions.push_back(item.cast<std::string>());
        })

        .def(py::pickle(
            [](const Tango::PeriodicEventInfo &p) { //__getstate__
                return py::make_tuple(p.period, p.extensions);
            },
            [](py::tuple t) { //__setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                Tango::PeriodicEventInfo p = Tango::PeriodicEventInfo();
                p.period = t[0].cast<std::string>();
                p.extensions = t[1].cast<std::vector<std::string> >();
                return p;
            }
        ));
    ;
}
