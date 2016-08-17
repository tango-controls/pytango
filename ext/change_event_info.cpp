/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include <tango.h>

using namespace boost::python;

void export_change_event_info()
{
    class_<Tango::ChangeEventInfo>("ChangeEventInfo")
        .enable_pickling()
        .def_readwrite("rel_change", &Tango::ChangeEventInfo::rel_change)
        .def_readwrite("abs_change", &Tango::ChangeEventInfo::abs_change)
        .def_readwrite("extensions", &Tango::ChangeEventInfo::extensions)
    ;
}
