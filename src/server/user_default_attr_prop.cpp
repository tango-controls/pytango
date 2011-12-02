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

#include "precompiled_header.hpp"
#include <tango.h>

using namespace boost::python;

void export_user_default_attr_prop()
{
    class_<Tango::UserDefaultAttrProp, boost::noncopyable>("UserDefaultAttrProp")
        .def("set_label", &Tango::UserDefaultAttrProp::set_label)
        .def("set_description", &Tango::UserDefaultAttrProp::set_description)
        .def("set_format", &Tango::UserDefaultAttrProp::set_format)
        .def("set_unit", &Tango::UserDefaultAttrProp::set_unit)
        .def("set_standard_unit", &Tango::UserDefaultAttrProp::set_standard_unit)
        .def("set_display_unit", &Tango::UserDefaultAttrProp::set_display_unit)
        .def("set_min_value", &Tango::UserDefaultAttrProp::set_min_value)
        .def("set_max_value", &Tango::UserDefaultAttrProp::set_max_value)
        .def("set_min_alarm", &Tango::UserDefaultAttrProp::set_min_alarm)
        .def("set_max_alarm", &Tango::UserDefaultAttrProp::set_max_alarm)
        .def("set_min_warning", &Tango::UserDefaultAttrProp::set_min_warning)
        .def("set_max_warning", &Tango::UserDefaultAttrProp::set_max_warning)
        .def("set_delta_t", &Tango::UserDefaultAttrProp::set_delta_t)
        .def("set_delta_val", &Tango::UserDefaultAttrProp::set_delta_val)
        .def("set_abs_change", &Tango::UserDefaultAttrProp::set_abs_change)
        .def("set_rel_change", &Tango::UserDefaultAttrProp::set_rel_change)
        .def("set_period", &Tango::UserDefaultAttrProp::set_period)
        .def("set_archive_abs_change", &Tango::UserDefaultAttrProp::set_archive_abs_change)
        .def("set_archive_rel_change", &Tango::UserDefaultAttrProp::set_archive_rel_change)
        .def("set_archive_period", &Tango::UserDefaultAttrProp::set_archive_period)
        
        .def_readwrite("label", &Tango::UserDefaultAttrProp::label)
        .def_readwrite("description", &Tango::UserDefaultAttrProp::description)
        .def_readwrite("unit", &Tango::UserDefaultAttrProp::unit)
        .def_readwrite("standard_unit", &Tango::UserDefaultAttrProp::standard_unit)
        .def_readwrite("display_unit", &Tango::UserDefaultAttrProp::display_unit)
        .def_readwrite("format", &Tango::UserDefaultAttrProp::format)
        .def_readwrite("min_value", &Tango::UserDefaultAttrProp::min_value)
        .def_readwrite("max_value", &Tango::UserDefaultAttrProp::max_value)
        .def_readwrite("min_alarm", &Tango::UserDefaultAttrProp::min_alarm)
        .def_readwrite("max_alarm", &Tango::UserDefaultAttrProp::max_alarm)
        .def_readwrite("min_warning", &Tango::UserDefaultAttrProp::min_warning)
        .def_readwrite("max_warning", &Tango::UserDefaultAttrProp::max_warning)
        .def_readwrite("delta_val", &Tango::UserDefaultAttrProp::delta_val)
        .def_readwrite("delta_t", &Tango::UserDefaultAttrProp::delta_t)
        .def_readwrite("abs_change", &Tango::UserDefaultAttrProp::abs_change)
        .def_readwrite("rel_change", &Tango::UserDefaultAttrProp::rel_change)
        .def_readwrite("period", &Tango::UserDefaultAttrProp::period)
        .def_readwrite("archive_abs_change", &Tango::UserDefaultAttrProp::archive_abs_change)
        .def_readwrite("archive_rel_change", &Tango::UserDefaultAttrProp::archive_rel_change)
        .def_readwrite("archive_period", &Tango::UserDefaultAttrProp::archive_period)
    ;

}

