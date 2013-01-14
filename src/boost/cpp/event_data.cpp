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

void export_event_data()
{
    class_<Tango::EventData>("EventData",
        init<const Tango::EventData &>())
    
        // The original Tango::EventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::EventData::device directly.
        // See callback.cpp
        .setattr("device", object())
        
        .def_readonly("attr_name", &Tango::EventData::attr_name)
        .def_readonly("event", &Tango::EventData::event)
        
        // The original Tango::EventData structure has "get_attr_value" but
        // we can't refer it directly here because we have to extract value
        // and so on.
        // See callback.cpp
        .setattr("attr_value", object())
        
        .def_readonly("err", &Tango::EventData::err)
        .def_readonly("reception_date", &Tango::EventData::reception_date)
        .add_property("errors", make_getter(&Tango::EventData::errors, 
            return_value_policy<copy_non_const_reference>()))
        
        .def("get_date", &Tango::EventData::get_date, 
            return_internal_reference<>())
    ;
}
