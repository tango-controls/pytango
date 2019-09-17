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

void export_pipe_info(py::module &m) {
    py::class_<Tango::PipeInfo>(m, "PipeInfo")
//        .def(py::init<const Tango::PipeInfo&>())
        .def_readwrite("name", &Tango::PipeInfo::name)
        .def_readwrite("description", &Tango::PipeInfo::description)
        .def_readwrite("label", &Tango::PipeInfo::label)
        .def_readwrite("disp_level", &Tango::PipeInfo::disp_level)
        .def_readwrite("writable", &Tango::PipeInfo::writable)
        .def_readwrite("extensions", &Tango::PipeInfo::extensions)

        .def_property("extensions", [](Tango::PipeInfo& self) -> py::list {
            py::list py_list;
            for(auto& item : self.extensions)
                py_list.append(item);
            return py_list;
        },[](Tango::PipeInfo& self, py::list py_list) -> void {
            for(auto& item : py_list)
                self.extensions.push_back(item.cast<std::string>());
        })

        .def(py::pickle(
            [](const Tango::PipeInfo &p) { //__getstate__
                return py::make_tuple(p.name, p.description, p.label,
                        p.disp_level, p.writable, p.extensions);
            },
            [](py::tuple t) { //__setstate__
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state!");
                Tango::PipeInfo p = Tango::PipeInfo();
                p.name = t[0].cast<std::string>();
                p.description = t[1].cast<std::string>();
                p.label = t[2].cast<std::string>();
                p.disp_level = t[3].cast<Tango::DispLevel>();
                p.writable = t[4].cast<Tango::PipeWriteType>();
                p.extensions = t[5].cast<std::vector<std::string>>();
                return p;
            }
        ));
    ;
}
