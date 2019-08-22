/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#ifndef _ATTRIBUTE_H_
#define _ATTRIBUTE_H_

#include <tango.h>
#include <pybind11/pybind11.h>
#include <pyutils.h>

namespace py = pybind11;

namespace PyAttribute
{
    void set_value(Tango::Attribute &, py::object &);
    void set_value(Tango::Attribute &, const std::string& , const std::string& );
    void set_value(Tango::Attribute &, const std::string& , py::object &);
    void set_value(Tango::Attribute &, py::object &, long);
    void set_value(Tango::Attribute &, py::object &, long, long);

    void set_value_date_quality(Tango::Attribute &, py::object &,
                                double, Tango::AttrQuality);
    void set_value_date_quality(Tango::Attribute &, const std::string& ,
                                const std::string& , double, Tango::AttrQuality);
    void set_value_date_quality(Tango::Attribute &, const std::string& ,
                                py::object &, double, Tango::AttrQuality);
    void set_value_date_quality(Tango::Attribute &, py::object &,
                                double, Tango::AttrQuality , long);
    void set_value_date_quality(Tango::Attribute &, py::object &,
                                double, Tango::AttrQuality , long, long);

    py::object get_properties(Tango::Attribute &, py::object &);
    py::object get_properties_2(Tango::Attribute &, py::object &);
    py::object get_properties_3(Tango::Attribute &, py::object &);
    py::object get_properties_multi_attr_prop(Tango::Attribute &, py::object &);

    void set_properties(Tango::Attribute &, py::object &, py::object &);
    void set_properties_3(Tango::Attribute &, py::object &, py::object &);
    void set_properties_multi_attr_prop(Tango::Attribute &, py::object &);
};

#endif // _ATTRIBUTE_H_
