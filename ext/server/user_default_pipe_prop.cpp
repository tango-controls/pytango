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

void export_user_default_pipe_prop(py::module &m) {
    py::class_<Tango::UserDefaultPipeProp>(m, "UserDefaultPipeProp")
        .def(py::init())
        .def("set_label", [](Tango::UserDefaultPipeProp& self,  std::string& def_label) -> void {
            self.set_label(def_label);
        })
        .def("set_description", [](Tango::UserDefaultPipeProp& self,  std::string& def_desc) -> void {
            self.set_description(def_desc);
        })
    ;
}
