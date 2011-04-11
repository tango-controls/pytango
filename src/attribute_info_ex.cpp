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

void export_attribute_info_ex()
{
    class_<Tango::AttributeInfoEx, bases<Tango::AttributeInfo> >
        ("AttributeInfoEx")
        .def_readwrite("alarms", &Tango::AttributeInfoEx::alarms)
        .def_readwrite("events", &Tango::AttributeInfoEx::events)
        .def_readwrite("sys_extensions", &Tango::AttributeInfoEx::sys_extensions)
    ;
}
