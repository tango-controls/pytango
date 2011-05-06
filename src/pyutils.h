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

#pragma once

#include <boost/python.hpp>

#define arg_(a) boost::python::arg(a)

#if PY_VERSION_HEX < 0x02050000
typedef int Py_ssize_t;
#endif

// -----------------------------------------------------------------------------
// The following section contains functions that changed signature from <=2.4
// using char* to >=2.5 using const char*. Basically we defined them here using
// const std::string

inline PyObject *PyObject_GetAttrString_(PyObject *o, const std::string &attr_name)
{
#if PY_VERSION_HEX < 0x02050000
    char *attr = const_cast<char *>(attr_name.c_str());
#else
    const char *attr = attr_name.c_str();
#endif
    return PyObject_GetAttrString(o, attr);
}

inline PyObject *PyImport_ImportModule_(const std::string &name)
{
#if PY_VERSION_HEX < 0x02050000
    char *attr = const_cast<char *>(name.c_str());
#else
    const char *attr = name.c_str();
#endif
    return PyImport_ImportModule(attr);
}

// -----------------------------------------------------------------------------
// The following section defines missing symbols in python <3.0 with macros

#if PY_VERSION_HEX < 0x02070000
    #if PY_VERSION_HEX < 0x02060000
        #define PyObject_CheckBuffer(object) (0)

        #define PyObject_GetBuffer(obj, view, flags) (PyErr_SetString(PyExc_NotImplementedError, \
                        "new buffer interface is not available"), -1)
        #define PyBuffer_FillInfo(view, obj, buf, len, readonly, flags) (PyErr_SetString(PyExc_NotImplementedError, \
                    "new buffer interface is not available"), -1)
        #define PyBuffer_Release(obj) (PyErr_SetString(PyExc_NotImplementedError, \
                        "new buffer interface is not available"), -1)
        // Bytes->String
        #define PyBytes_FromStringAndSize PyString_FromStringAndSize
        #define PyBytes_FromString PyString_FromString
        #define PyBytes_AsString PyString_AsString
        #define PyBytes_Size PyString_Size
    #endif

    #define PyMemoryView_FromBuffer(info) (PyErr_SetString(PyExc_NotImplementedError, \
                    "new buffer interface is not available"), (PyObject *)NULL)
    #define PyMemoryView_FromObject(object)     (PyErr_SetString(PyExc_NotImplementedError, \
                                        "new buffer interface is not available"), (PyObject *)NULL)
#endif

#if PY_VERSION_HEX >= 0x03000000
    // for buffers
    #define Py_END_OF_BUFFER ((Py_ssize_t) 0)

    #define PyObject_CheckReadBuffer(object) (0)

    #define PyBuffer_FromMemory(ptr, s) (PyErr_SetString(PyExc_NotImplementedError, \
                            "old buffer interface is not available"), (PyObject *)NULL)
    #define PyBuffer_FromReadWriteMemory(ptr, s) (PyErr_SetString(PyExc_NotImplementedError, \
                            "old buffer interface is not available"), (PyObject *)NULL)
    #define PyBuffer_FromObject(object, offset, size)  (PyErr_SetString(PyExc_NotImplementedError, \
                            "old buffer interface is not available"), (PyObject *)NULL)
    #define PyBuffer_FromReadWriteObject(object, offset, size)  (PyErr_SetString(PyExc_NotImplementedError, \
                            "old buffer interface is not available"), (PyObject *)NULL)

#endif

inline void raise_(PyObject *type, const char *message)
{
    PyErr_SetString(type, message);
    boost::python::throw_error_already_set();
}

/// You should run any I/O intensive operations (like requesting data through
/// the network) in the context of an object like this.
class AutoPythonAllowThreads
{
    PyThreadState *m_save;
    
public:
    
    inline void giveup() { if (m_save) { PyEval_RestoreThread(m_save); m_save = 0; } }
    
    inline AutoPythonAllowThreads() { m_save = PyEval_SaveThread(); } ;
    inline ~AutoPythonAllowThreads() {giveup();} ;
};

/**
 * Determines if the given method name exists and is callable
 * within the python class
 *
 * @param[in] obj object to search for the method
 * @param[in] method_name the name of the method
 *
 * @return returns true is the method exists or false otherwise
 */
bool is_method_defined(boost::python::object &obj, const std::string &method_name);

/**
 * Determines if the given method name exists and is callable
 * within the python class
 *
 * @param[in] obj object to search for the method
 * @param[in] method_name the name of the method
 *
 * @return returns true is the method exists or false otherwise
 */
bool is_method_defined(PyObject *obj, const std::string &method_name);

/**
 * Determines if the given method name exists and is callable
 * within the python class
 *
 * @param[in] obj object to search for the method
 * @param[in] method_name the name of the method
 * @param[out] exists set to true if the symbol exists or false otherwise
 * @param[out] is_method set to true if the symbol exists and is a method
 *             or false otherwise
 */
void is_method_defined(PyObject *obj, const std::string &method_name,
                       bool &exists, bool &is_method);

/**
 * Determines if the given method name exists and is callable
 * within the python class
 *
 * @param[in] obj object to search for the method
 * @param[in] method_name the name of the method
 * @param[out] exists set to true if the symbol exists or false otherwise
 * @param[out] is_method set to true if the symbol exists and is a method
 *             or false otherwise
 */
void is_method_defined(boost::python::object &obj, const std::string &method_name,
                       bool &exists, bool &is_method);

#define PYTANGO_MOD \
    boost::python::object pytango((boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("PyTango")))));
    
#define CALL_METHOD(retType, self, name, ...) \
    boost::python::call_method<retType>(self, name , __VA_ARGS__);
    
