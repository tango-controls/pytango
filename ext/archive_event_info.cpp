/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2018 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void export_archive_event_info(py::module &m) {
    py::class_<Tango::ArchiveEventInfo>(m, "ArchiveEventInfo")
        .def_readwrite("archive_rel_change", &Tango::ArchiveEventInfo::archive_rel_change)
        .def_readwrite("archive_abs_change", &Tango::ArchiveEventInfo::archive_abs_change)
        .def_readwrite("archive_period", &Tango::ArchiveEventInfo::archive_period)
        .def_property("extensions", [](Tango::ArchiveEventInfo& self) -> py::list {
            py::list py_list;
            for(auto& item : self.extensions)
                py_list.append(item);
            return py_list;
        },[](Tango::ArchiveEventInfo& self, py::list py_list) -> void {
            for(auto& item : py_list)
                self.extensions.push_back(item.cast<std::string>());
        })
        .def(py::pickle(
            [](const Tango::ArchiveEventInfo &p) { //__getstate__
                return py::make_tuple(p.archive_rel_change,
                    p.archive_abs_change,
                    p.archive_period,
                    p.extensions);
            },
            [](py::tuple t) { //__setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");
                Tango::ArchiveEventInfo p = Tango::ArchiveEventInfo();
                p.archive_rel_change = t[0].cast<std::string>();
                p.archive_abs_change = t[1].cast<std::string>();
                p.archive_period = t[2].cast<std::string>();
                p.extensions = t[3].cast<std::vector<std::string> >();
                return p;
            }
        ));
    ;
}
