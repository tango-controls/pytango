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


void export_attribute_event_info(py::module &m) {
    py::class_<Tango::AttributeEventInfo>(m, "AttributeEventInfo")
       .def_readwrite("ch_event", &Tango::AttributeEventInfo::ch_event)
       .def_readwrite("per_event", &Tango::AttributeEventInfo::per_event)
       .def_readwrite("arch_event", &Tango::AttributeEventInfo::arch_event)
       .def(py::pickle(
           [](const Tango::AttributeEventInfo &p) { //__getstate__
               return py::make_tuple(p.ch_event, p.per_event,
                       p.arch_event);
           },
           [](py::tuple t) { //__setstate__
               if (t.size() != 3)
                   throw std::runtime_error("Invalid state!");
               Tango::AttributeEventInfo p = Tango::AttributeEventInfo();
               p.ch_event = t[0].cast<Tango::ChangeEventInfo>();
               p.per_event = t[1].cast<Tango::PeriodicEventInfo>();
               p.arch_event = t[2].cast<Tango::ArchiveEventInfo>();
               return p;
           }
       ));
    ;
}
