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
#include "pyutils.h"
#include <tango.h>

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
    { return from_char_to_str(self.reason); }

    static inline void set_reason(Tango::DevError &self, PyObject *str)
    { PyDevError::from_str_to_char(str, self.reason); }

    static inline PyObject* get_desc(Tango::DevError &self)
    { return from_char_to_str(self.desc); }

    static inline void set_desc(Tango::DevError &self, PyObject *str)
    { PyDevError::from_str_to_char(str, self.desc); }

    static inline PyObject* get_origin(Tango::DevError &self)
    { return from_char_to_str(self.origin); }

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
