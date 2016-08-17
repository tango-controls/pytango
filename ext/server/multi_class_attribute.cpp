/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "pytgutils.h"

using namespace boost::python;

void export_multi_class_attribute()
{
    Tango::Attr& (Tango::MultiClassAttribute::*get_attr_)(std::string &) =
        &Tango::MultiClassAttribute::get_attr;

    class_<Tango::MultiClassAttribute, boost::noncopyable>("MultiClassAttribute", no_init)
        .def("get_attr",
            (Tango::Attr& (Tango::MultiClassAttribute::*) (const std::string &))
            get_attr_,
            return_value_policy<reference_existing_object>())
	.def("remove_attr", &Tango::MultiClassAttribute::remove_attr)
        .def("get_attr_list", &Tango::MultiClassAttribute::get_attr_list,
            return_value_policy<reference_existing_object>())
    ;
}
