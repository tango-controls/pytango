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

void export_locker_info(py::module &m) {
    py::class_<Tango::LockerInfo>(m, "LockerInfo")
        .def_readonly("ll", &Tango::LockerInfo::ll)
        .def_readonly("locker_host", &Tango::LockerInfo::locker_host)
        .def_readonly("locker_class", &Tango::LockerInfo::locker_class)
        .def_property_readonly("li", [](Tango::LockerInfo &li) {
            return (li.ll == Tango::CPP) ?
                    py::object(py::cast(li.li.LockerPid)) :
                    py::make_tuple(li.li.UUID);
            })
    ;
}
