/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include <tango.h>

#include "exception.h"

using namespace boost::python;

extern boost::python::object PyTango_DevFailed;

namespace PyAttrConfEventData
{
    static boost::shared_ptr<Tango::AttrConfEventData> makeAttrConfEventData()
    {
        Tango::AttrConfEventData *result = new Tango::AttrConfEventData;
        return boost::shared_ptr<Tango::AttrConfEventData>(result);
    }

    static void set_errors(Tango::AttrConfEventData &event_data, 
                           boost::python::object &dev_failed)
    {
        Tango::DevFailed df;
        boost::python::object errors = dev_failed.attr("args");
        sequencePyDevError_2_DevErrorList(errors.ptr(), event_data.errors);
    }
};

void export_attr_conf_event_data()
{
    class_<Tango::AttrConfEventData>("AttrConfEventData",
        init<const Tango::AttrConfEventData &>())

        .def("__init__", boost::python::make_constructor(PyAttrConfEventData::makeAttrConfEventData))

        // The original Tango::AttrConfEventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::AttrConfEventData::device directly.
        // See callback.cpp
        .setattr("device",object())
        .def_readwrite("attr_name", &Tango::AttrConfEventData::attr_name)
        .def_readwrite("event", &Tango::AttrConfEventData::event)

        .setattr("attr_conf",object())

        .def_readwrite("err", &Tango::AttrConfEventData::err)
        .def_readwrite("reception_date", &Tango::AttrConfEventData::reception_date)
        .add_property("errors", 
		      make_getter(&Tango::AttrConfEventData::errors, 
				  return_value_policy<copy_non_const_reference>()),
		      &PyAttrConfEventData::set_errors)

        .def("get_date", &Tango::AttrConfEventData::get_date,
            return_internal_reference<>())
    ;
}
