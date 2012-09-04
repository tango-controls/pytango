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
#include "defs.h"
#include "pyutils.h"

using namespace boost::python;

PyObject* from_char_to_str(const std::string& in, 
                           const char* encoding /*=NULL defaults to latin-1 */,
                           const char* errors /*="strict" */)
{
    return from_char_to_str(in.c_str(), in.size(), encoding, errors);
}

PyObject* from_char_to_str(const char* in, Py_ssize_t size /* =-1 */, 
                           const char* encoding /*=NULL defaults to latin-1 */,
                           const char* errors /*="strict" */)
{
if (size < 0)
{
    size = strlen(in);
}
#ifdef PYTANGO_PY3K
    if (!encoding)
    {
        return PyUnicode_DecodeLatin1(in, size, errors);
    }
    else
    {   
        return PyUnicode_Decode(in, size, encoding, errors);
    }
#else
    return PyString_FromStringAndSize(in, size);
#endif
}

void from_str_to_char(PyObject* in, std::string& out)
{
    if (PyUnicode_Check(in))
    {
        PyObject *bytes_in = PyUnicode_AsLatin1String(in);
        out = PyBytes_AsString(bytes_in);
        Py_DECREF(bytes_in);
    }
    else 
    {
        out = std::string(PyBytes_AsString(in), PyBytes_Size(in));
    }
}

bool is_method_defined(object &obj, const std::string &method_name)
{
    return is_method_defined(obj.ptr(), method_name);
}

bool is_method_defined(PyObject *obj, const std::string &method_name)
{
    bool exists, is_method;
    is_method_defined(obj, method_name, exists, is_method);
    return exists && is_method;
}

void is_method_defined(object &obj, const std::string &method_name,
                       bool &exists, bool &is_method)
{
    is_method_defined(obj.ptr(), method_name, exists, is_method);
}

void is_method_defined(PyObject *obj, const std::string &method_name,
                       bool &exists, bool &is_method)
{
    exists = is_method = false;

    PyObject *meth = PyObject_GetAttrString_(obj, method_name.c_str());

    exists = NULL != meth;

    if (!exists)
    {
        PyErr_Clear();
        return;
    }

    is_method = (1 == PyCallable_Check(meth));
    Py_DECREF(meth);
}
