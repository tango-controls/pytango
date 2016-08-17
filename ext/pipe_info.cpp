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

void export_pipe_info()
{
    class_<Tango::PipeInfo>
        ("PipeInfo")
        .def(init<const Tango::PipeInfo&>())
        .enable_pickling()
	.def_readwrite("name",  &Tango::PipeInfo::name)
	.def_readwrite("description",  &Tango::PipeInfo::description)
	.def_readwrite("label",  &Tango::PipeInfo::label)
        .def_readwrite("disp_level", &Tango::PipeInfo::disp_level)
        .def_readwrite("writable", &Tango::PipeInfo::writable)
        .def_readwrite("extensions", &Tango::PipeInfo::extensions)
    ;
}
