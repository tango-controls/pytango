/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   (copyleft) CELLS / ALBA Synchrotron, Bellaterra, Spain
  
   This is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.
  
   This software is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with this program; if not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

#include <boost/python.hpp>
#include <tango/tango.h>

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
