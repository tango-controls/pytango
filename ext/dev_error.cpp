/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include <tango.h>
#include "pyutils.h"

namespace PyDevError
{
    static void from_str_to_char(PyObject* in, CORBA::String_member& out)
    {
        if (PyUnicode_Check(in))
        {
            PyObject *bytes_in = PyUnicode_AsLatin1String(in);
            out = CORBA::string_dup(PyBytes_AsString(bytes_in));
            Py_DECREF(bytes_in);
        }
        else 
        {
            out = CORBA::string_dup(PyBytes_AsString(in));
        }
    }

    static inline PyObject* get_reason(Tango::DevError &self)
    { return from_char_to_python_str(self.reason); }

    static inline void set_reason(Tango::DevError &self, PyObject *str)
    { PyDevError::from_str_to_char(str, self.reason); }

    static inline PyObject* get_desc(Tango::DevError &self)
    { return from_char_to_python_str(self.desc); }

    static inline void set_desc(Tango::DevError &self, PyObject *str)
    { PyDevError::from_str_to_char(str, self.desc); }

    static inline PyObject* get_origin(Tango::DevError &self)
    { return from_char_to_python_str(self.origin); }

    static inline void set_origin(Tango::DevError &self, PyObject *str)
    { PyDevError::from_str_to_char(str, self.origin); }
};

void export_dev_error()
{
    bopy::class_<Tango::DevError>("DevError")
        .enable_pickling()
        .add_property("reason", &PyDevError::get_reason, &PyDevError::set_reason)
        .def_readwrite("severity", &Tango::DevError::severity)
        .add_property("desc", &PyDevError::get_desc, &PyDevError::set_desc)
        .add_property("origin", &PyDevError::get_origin, &PyDevError::set_origin)
    ;
}
