/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include <tango.h>

struct PyLockerInfo
{
    static inline boost::python::object get_locker_id(Tango::LockerInfo &li)
    {
        return (li.ll == Tango::CPP) ?
            boost::python::object(li.li.LockerPid) :
            boost::python::tuple(li.li.UUID);
    }
};

void export_locker_info()
{
    boost::python::class_<Tango::LockerInfo>("LockerInfo")
        .def_readonly("ll", &Tango::LockerInfo::ll)
        .add_property("li", &PyLockerInfo::get_locker_id)
        .def_readonly("locker_host", &Tango::LockerInfo::locker_host)
        .def_readonly("locker_class", &Tango::LockerInfo::locker_class)
    ;
}
