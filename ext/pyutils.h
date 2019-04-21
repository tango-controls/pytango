/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include <boost/python.hpp>
#include <tango.h>

namespace bopy = boost::python;

#define arg_(a) boost::python::arg(a)

#if PY_MAJOR_VERSION >= 3
#define PYTANGO_PY3K
#endif

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

// Bytes interface
#if PY_VERSION_HEX < 0x02060000
    #define PyBytesObject PyStringObject
    #define PyBytes_Type PyString_Type

    #define PyBytes_Check PyString_Check
    #define PyBytes_CheckExact PyString_CheckExact
    #define PyBytes_CHECK_INTERNED PyString_CHECK_INTERNED
    #define PyBytes_AS_STRING PyString_AS_STRING
    #define PyBytes_GET_SIZE PyString_GET_SIZE
    #define Py_TPFLAGS_BYTES_SUBCLASS Py_TPFLAGS_STRING_SUBCLASS

    #define PyBytes_FromStringAndSize PyString_FromStringAndSize
    #define PyBytes_FromString PyString_FromString
    #define PyBytes_FromFormatV PyString_FromFormatV
    #define PyBytes_FromFormat PyString_FromFormat
    #define PyBytes_Size PyString_Size
    #define PyBytes_AsString PyString_AsString
    #define PyBytes_Repr PyString_Repr
    #define PyBytes_Concat PyString_Concat
    #define PyBytes_ConcatAndDel PyString_ConcatAndDel
    #define _PyBytes_Resize _PyString_Resize
    #define _PyBytes_Eq _PyString_Eq
    #define PyBytes_Format PyString_Format
    #define _PyBytes_FormatLong _PyString_FormatLong
    #define PyBytes_DecodeEscape PyString_DecodeEscape
    #define _PyBytes_Join _PyString_Join
    #define PyBytes_Decode PyString_Decode
    #define PyBytes_Encode PyString_Encode
    #define PyBytes_AsEncodedObject PyString_AsEncodedObject
    #define PyBytes_AsEncodedString PyString_AsEncodedString
    #define PyBytes_AsDecodedObject PyString_AsDecodedObject
    #define PyBytes_AsDecodedString PyString_AsDecodedString
    #define PyBytes_AsStringAndSize PyString_AsStringAndSize
    #define _PyBytes_InsertThousandsGrouping _PyString_InsertThousandsGrouping
#else
    #include <bytesobject.h>
#endif

/* PyCapsule definitions for old python */

#if (    (PY_VERSION_HEX <  0x02070000) \
     || ((PY_VERSION_HEX >= 0x03000000) \
      && (PY_VERSION_HEX <  0x03010000)) )

#define PYCAPSULE_OLD

#define __PyCapsule_GetField(capsule, field, default_value) \
    ( PyCapsule_CheckExact(capsule) \
        ? (((PyCObject *)capsule)->field) \
        : (default_value) \
    ) \

#define __PyCapsule_SetField(capsule, field, value) \
    ( PyCapsule_CheckExact(capsule) \
        ? (((PyCObject *)capsule)->field = value), 1 \
        : 0 \
    ) \


#define PyCapsule_Type PyCObject_Type

#define PyCapsule_CheckExact(capsule) (PyCObject_Check(capsule))
#define PyCapsule_IsValid(capsule, name) (PyCObject_Check(capsule))


#define PyCapsule_New(pointer, name, destructor) \
    (PyCObject_FromVoidPtr(pointer, destructor))


#define PyCapsule_GetPointer(capsule, name) \
    (PyCObject_AsVoidPtr(capsule))

/* Don't call PyCObject_SetPointer here, it fails if there's a destructor */
#define PyCapsule_SetPointer(capsule, pointer) \
    __PyCapsule_SetField(capsule, cobject, pointer)


#define PyCapsule_GetDestructor(capsule) \
    __PyCapsule_GetField(capsule, destructor)

#define PyCapsule_SetDestructor(capsule, dtor) \
    __PyCapsule_SetField(capsule, destructor, dtor)


/*
 * Sorry, there's simply no place
 * to store a Capsule "name" in a CObject.
 */
#define PyCapsule_GetName(capsule) NULL

int PyCapsule_SetName(PyObject *capsule, const char *unused);

#define PyCapsule_GetContext(capsule) \
    __PyCapsule_GetField(capsule, descr)

#define PyCapsule_SetContext(capsule, context) \
    __PyCapsule_SetField(capsule, descr, context)


void * PyCapsule_Import(const char *name, int no_block);

#endif /* #if PY_VERSION_HEX < 0x02070000 */

PyObject* from_char_to_python_str(const char* in, Py_ssize_t size=-1,
                                  const char* encoding=NULL, /* defaults to latin-1 */
                                  const char* errors="strict");

PyObject* from_char_to_python_str(const std::string& in,
                                  const char* encoding=NULL, /* defaults to latin-1 */
                                  const char* errors="strict");

bopy::object from_char_to_boost_str(const char* in, Py_ssize_t size=-1,
                                    const char* encoding=NULL, /* defaults to latin-1 */
                                    const char* errors="strict");

bopy::object from_char_to_boost_str(const std::string& in,
                                    const char* encoding=NULL, /* defaults to latin-1 */
                                    const char* errors="strict");


void from_str_to_char(const bopy::object& in, std::string& out);
void from_str_to_char(PyObject* in, std::string& out);
char* from_str_to_char(PyObject* in);
char* from_str_to_char(const bopy::object& in);

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

    inline void giveup()
    {
        if (m_save)
        {
            PyEval_RestoreThread(m_save);
            m_save = 0;
        }
    }

    inline AutoPythonAllowThreads()
    {
        m_save = PyEval_SaveThread();
    }

    inline ~AutoPythonAllowThreads()
    {
        giveup();
    }
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
    boost::python::object pytango((boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("tango")))));

#define CALL_METHOD(retType, self, name, ...) \
    boost::python::call_method<retType>(self, name , __VA_ARGS__);


bool hasattr(boost::python::object &, const std::string &);

#if _MSC_VER > 1800
namespace boost
{
	template <>
	Tango::ApiUtil const volatile * get_pointer<class Tango::ApiUtil const volatile >(
		class Tango::ApiUtil const volatile *c)
	{
		return c;
	}
    template <>
	Tango::Pipe const volatile * get_pointer<class Tango::Pipe const volatile >(
		class Tango::Pipe const volatile *c)
	{
		return c;
	}
	template <>
	Tango::WPipe const volatile * get_pointer<class Tango::WPipe const volatile >(
		class Tango::WPipe const volatile *c)
	{
		return c;
	}
    template <>
	Tango::WAttribute const volatile * get_pointer<class Tango::WAttribute const volatile >(
		class Tango::WAttribute const volatile *c)
	{
		return c;
	}
}
#endif