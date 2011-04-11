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

#ifndef _ATTRIBUTE_H_
#define _ATTRIBUTE_H_

#include <boost/python.hpp>
#include <tango.h>

namespace PyAttribute
{
    void set_value(Tango::Attribute &, boost::python::object &);

    void set_value(Tango::Attribute &, boost::python::str &,
                   boost::python::str &);

    void set_value(Tango::Attribute &, boost::python::object &, long);

    void set_value(Tango::Attribute &, boost::python::object &, long, long);

    void set_value_date_quality(Tango::Attribute &, boost::python::object &,
                                double, Tango::AttrQuality);

    void set_value_date_quality(Tango::Attribute &, boost::python::str &,
                                boost::python::str &, double,
                                Tango::AttrQuality);

    void set_value_date_quality(Tango::Attribute &, boost::python::object &,
                                double, Tango::AttrQuality , long);

    void set_value_date_quality(Tango::Attribute &, boost::python::object &,
                                double, Tango::AttrQuality , long, long);
                                
    boost::python::object get_properties(Tango::Attribute &,
                                         boost::python::object &);

    boost::python::object get_properties_2(Tango::Attribute &,
                                           boost::python::object &);

    boost::python::object get_properties_3(Tango::Attribute &,
                                           boost::python::object &);

    void set_properties(Tango::Attribute &, boost::python::object &,
                        boost::python::object &);
    
    void set_properties_3(Tango::Attribute &, boost::python::object &,
                          boost::python::object &);
};


#endif // _ATTRIBUTE_H_
