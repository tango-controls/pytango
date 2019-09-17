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

void export_time_val(py::module &m) {
    py::class_<Tango::TimeVal>(m, "TimeVal")
        .def_readwrite("tv_sec", &Tango::TimeVal::tv_sec)
        .def_readwrite("tv_usec", &Tango::TimeVal::tv_usec)
        .def_readwrite("tv_nsec", &Tango::TimeVal::tv_nsec)
    ;
}
