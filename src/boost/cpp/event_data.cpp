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

namespace PyEventData
{
    static boost::shared_ptr<Tango::EventData> makeEventData()
    {
        Tango::EventData *result = new Tango::EventData;
        result->attr_value = new Tango::DeviceAttribute();
       return boost::shared_ptr<Tango::EventData>(result);
    }

    static void set_errors(Tango::EventData &event_data, boost::python::object &error)
    {
        PyObject* error_ptr = error.ptr();
        if (PyObject_IsInstance(error_ptr, PyTango_DevFailed.ptr()))
        {
            Tango::DevFailed df;
	    boost::python::object error_list = error.attr("args");
	    sequencePyDevError_2_DevErrorList(error_list.ptr(), event_data.errors);
        }
        else
        {
            sequencePyDevError_2_DevErrorList(error_ptr, event_data.errors);
        }
    }
};

void export_event_data()
{
    class_<Tango::EventData>("EventData",
        init<const Tango::EventData &>())
    
        .def("__init__", boost::python::make_constructor(PyEventData::makeEventData))

        // The original Tango::EventData structure has a 'device' field.
        // However, if we returned this directly we would get a different
        // python device each time. So we are doing our weird things to make
        // sure the device returned is the same where the read action was
        // performed. So we don't return Tango::EventData::device directly.
        // See callback.cpp
        .setattr("device", object())
        
        .def_readwrite("attr_name", &Tango::EventData::attr_name)
        .def_readwrite("event", &Tango::EventData::event)
        
        // The original Tango::EventData structure has "get_attr_value" but
        // we can't refer it directly here because we have to extract value
        // and so on.
        // See callback.cpp
        .setattr("attr_value", object())
        
        .def_readwrite("err", &Tango::EventData::err)
        .def_readwrite("reception_date", &Tango::EventData::reception_date)
        .add_property("errors", 
		      make_getter(&Tango::EventData::errors, 
				  return_value_policy<copy_non_const_reference>()),
		      &PyEventData::set_errors)
        
        .def("get_date", &Tango::EventData::get_date, 
            return_internal_reference<>())
    ;
}
