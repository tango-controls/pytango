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

using namespace boost::python;

void export_dev_command_info()
{
    typedef Tango::CmdArgType Tango::_DevCommandInfo::* MemCmdArgType;

    class_<Tango::DevCommandInfo>("DevCommandInfo")
        .def_readonly("cmd_name", &Tango::DevCommandInfo::cmd_name)
        .def_readonly("cmd_tag", &Tango::DevCommandInfo::cmd_tag)
        .def_readonly("in_type",
                reinterpret_cast<MemCmdArgType>(&Tango::DevCommandInfo::in_type))
        .def_readonly("out_type",
                reinterpret_cast<MemCmdArgType>(&Tango::DevCommandInfo::out_type))
        .def_readonly("in_type_desc", &Tango::DevCommandInfo::in_type_desc)
        .def_readonly("out_type_desc", &Tango::DevCommandInfo::out_type_desc)
    ;
}
