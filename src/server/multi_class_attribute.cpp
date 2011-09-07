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

void export_multi_class_attribute()
{
    Tango::Attr& (Tango::MultiClassAttribute::*get_attr_)(std::string &) =
        &Tango::MultiClassAttribute::get_attr;
    void (Tango::MultiClassAttribute::*remove_attr_)(std::string &, const std::string &) =
        &Tango::MultiClassAttribute::remove_attr;

    class_<Tango::MultiClassAttribute, boost::noncopyable>("MultiClassAttribute", no_init)
        .def("get_attr",
            (Tango::Attr& (Tango::MultiClassAttribute::*) (const std::string &))
            get_attr_,
            return_value_policy<reference_existing_object>())
        .def("remove_attr",
            (void (Tango::MultiClassAttribute::*) (const std::string &, const std::string &))
            remove_attr_)
        .def("get_attr_list", &Tango::MultiClassAttribute::get_attr_list,
            return_value_policy<reference_existing_object>())
    ;
}
