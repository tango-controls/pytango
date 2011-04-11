/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
   
   PyTango is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   PyTango is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

#include "pytgutils.h"
#include <sstream>

using namespace boost::python;

void export_multi_attribute()
{
    class_<Tango::MultiAttribute, boost::noncopyable>("MultiAttribute", no_init)
        .def("get_attr_by_name", &Tango::MultiAttribute::get_attr_by_name,
            return_value_policy<reference_existing_object>())
        .def("get_attr_by_ind", &Tango::MultiAttribute::get_attr_by_ind,
            return_value_policy<reference_existing_object>())
        .def("get_w_attr_by_name", &Tango::MultiAttribute::get_w_attr_by_name,
            return_value_policy<reference_existing_object>())
        .def("get_w_attr_by_ind", &Tango::MultiAttribute::get_w_attr_by_ind,
            return_value_policy<reference_existing_object>())
        .def("get_attr_ind_by_name", &Tango::MultiAttribute::get_attr_ind_by_name) // New in 7.0.0
        .def("get_alarm_list", &Tango::MultiAttribute::get_alarm_list,
            return_internal_reference<>()) // New in 7.0.0
        .def("get_attr_nb", &Tango::MultiAttribute::get_attr_nb) // New in 7.0.0
        .def("check_alarm",
            (bool (Tango::MultiAttribute::*) ())
            &Tango::MultiAttribute::check_alarm) // New in 7.0.0
        .def("check_alarm",
            (bool (Tango::MultiAttribute::*) (const long))
            &Tango::MultiAttribute::check_alarm) // New in 7.0.0
        .def("check_alarm",
            (bool (Tango::MultiAttribute::*) (const char *))
            &Tango::MultiAttribute::check_alarm) // New in 7.0.0
        .def("read_alarm",
            (void (Tango::MultiAttribute::*) (const std::string &))
            &Tango::MultiAttribute::read_alarm) // New in 7.0.0
    ;
}
