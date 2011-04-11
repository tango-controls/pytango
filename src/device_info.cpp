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
#include <tango.h>

using namespace boost::python;

void export_device_info()
{
    class_<Tango::DeviceInfo>("DeviceInfo",
        "A structure containing available information for a device with the\n"
        "following members,\n"
        " - dev_class : string\n"
        " - server_id : string\n"
        " - server_host : string\n"
        " - server_version : integer\n"
        " - doc_url : string")
        .def_readonly("dev_class", &Tango::DeviceInfo::dev_class)
        .def_readonly("server_id", &Tango::DeviceInfo::server_id)
        .def_readonly("server_host", &Tango::DeviceInfo::server_host)
        .def_readonly("server_version", &Tango::DeviceInfo::server_version)
        .def_readonly("doc_url", &Tango::DeviceInfo::doc_url)
        .def_readonly("dev_type", &Tango::DeviceInfo::dev_type)
    ;
}
