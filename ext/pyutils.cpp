/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "defs.h"
#include "pyutils.h"

using namespace boost::python;

bopy::object from_char_to_boost_str(const std::string& in,
                                    const char* encoding /*=NULL defaults to latin-1 */,
                                    const char* errors /*="strict" */)
{
    return from_char_to_boost_str(in.c_str(), in.size(), encoding, errors);
}

bopy::object from_char_to_boost_str(const char* in, Py_ssize_t size /* =-1 */,
                                    const char* encoding /*=NULL defaults to latin-1 */,
                                    const char* errors /*="strict" */)
{
    return bopy::object(bopy::handle<>(from_char_to_python_str(in, size, encoding, errors)));
}

PyObject* from_char_to_python_str(const std::string& in,
                                  const char* encoding /*=NULL defaults to latin-1 */,
                                  const char* errors /*="strict" */)
{
    return from_char_to_python_str(in.c_str(), in.size(), encoding, errors);
}

PyObject* from_char_to_python_str(const char* in, Py_ssize_t size /* =-1 */,
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

void from_str_to_char(const bopy::object& in, std::string& out)
{
    from_str_to_char(in.ptr(), out);
}

void from_str_to_char(PyObject* in, std::string& out)
{
    if (PyUnicode_Check(in))
    {
        PyObject *bytes_in = PyUnicode_AsLatin1String(in);
        out = std::string(PyBytes_AsString(bytes_in), PyBytes_Size(bytes_in));
        Py_DECREF(bytes_in);
    }
    else
    {
        out = std::string(PyBytes_AsString(in), PyBytes_Size(in));
    }
}

char* from_str_to_char(const bopy::object& in)
{
    return from_str_to_char(in.ptr());
}

// The result is a newly allocated buffer. It is the responsibility
// of the caller to manage the memory returned by this function
char* from_str_to_char(PyObject* in)
{
    char* out = NULL;
    if (PyUnicode_Check(in))
    {
        PyObject *bytes_in = PyUnicode_AsLatin1String(in);
	Py_ssize_t size = PyBytes_Size(bytes_in);
	out = new char[size+1];
	out[size] = '\0';
	out = strncpy(out, PyBytes_AsString(bytes_in), size);
        Py_DECREF(bytes_in);
    }
    else
    {
	Py_ssize_t size = PyBytes_Size(in);
	out = new char[size+1];
	out[size] = '\0';
	out = strncpy(out, PyBytes_AsString(in), size);
    }
    return out;
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

#ifdef PYCAPSULE_OLD

int PyCapsule_SetName(PyObject *capsule, const char *unused)
{
    unused = unused;
    PyErr_SetString(PyExc_NotImplementedError,
        "can't use PyCapsule_SetName with CObjects");
    return 1;
}

void *PyCapsule_Import(const char *name, int no_block)
{
    PyObject *object = NULL;
    void *return_value = NULL;
    char *trace;
    size_t name_length = (strlen(name) + 1) * sizeof(char);
    char *name_dup = (char *)PyMem_MALLOC(name_length);

    if (!name_dup) {
        return NULL;
    }

    memcpy(name_dup, name, name_length);

    trace = name_dup;
    while (trace) {
        char *dot = strchr(trace, '.');
        if (dot) {
            *dot++ = '\0';
        }

        if (object == NULL) {
            if (no_block) {
                object = PyImport_ImportModuleNoBlock(trace);
            } else {
                object = PyImport_ImportModule(trace);
                if (!object) {
                    PyErr_Format(PyExc_ImportError,
                        "PyCapsule_Import could not "
                        "import module \"%s\"", trace);
                }
            }
        } else {
            PyObject *object2 = PyObject_GetAttrString(object, trace);
            Py_DECREF(object);
            object = object2;
        }
        if (!object) {
            goto EXIT;
        }

        trace = dot;
    }

    if (PyCObject_Check(object)) {
        PyCObject *cobject = (PyCObject *)object;
        return_value = cobject->cobject;
    } else {
        PyErr_Format(PyExc_AttributeError,
            "PyCapsule_Import \"%s\" is not valid",
            name);
    }

EXIT:
    Py_XDECREF(object);
    if (name_dup) {
        PyMem_FREE(name_dup);
    }
    return return_value;
}

#endif

bool hasattr(boost::python::object& obj, const std::string& name)
{
    return PyObject_HasAttrString(obj.ptr(), name.c_str());
}
