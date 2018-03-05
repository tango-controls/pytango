/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include "from_py.h"

/**
 * Translation between python object to Tango data type.
 *
 * Example:
 * Tango::DevLong tg_value;
 * try
 * {
 *     from_py<Tango::DEV_LONG>::convert(py_obj, tg_value);
 * }
 * catch(boost::python::error_already_set &eas)
 * {
 *     handle_error(eas);
 * }
 */
template<long tangoTypeConst>
struct from_py
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;

    static inline void convert(const py::object &o, TangoScalarType &tg)
    {
        convert(o.ptr(), tg);
    }

    static inline void convert(PyObject *o, TangoScalarType &tg)
    {
        tg = o.cast<TangoScalartype>();
//        Tango::Except::throw_exception( \
//                        "PyDs_WrongPythonDataTypeForAttribute",
//                        "Unsupported attribute type translation",
//                        "from_py::convert()");
    }
};

#define DEFINE_FAST_TANGO_FROMPY(tangoTypeConst, FN) \
template<> \
struct from_py<tangoTypeConst> \
{ \
    typedef TANGO_const2type(tangoTypeConst) TangoScalarType; \
\
    static inline void convert(const boost::python::object &o, TangoScalarType &tg) \
    { \
        convert(o.ptr(), tg); \
    } \
\
    static inline void convert(PyObject *o, TangoScalarType &tg) \
    { \
        tg = static_cast<TangoScalarType>(FN(o));  \
        if(PyErr_Occurred()) \
            boost::python::throw_error_already_set();  \
    } \
};

#undef max 
#undef min

// DEFINE_FAST_TANGO_FROMPY should be enough. However, as python does not
// provide conversion from python integers to all the data types accepted
// by tango we must check the ranges manually. Also now we can add numpy
// support to some extent...
#ifdef DISABLE_PYTANGO_NUMPY
# define DEFINE_FAST_TANGO_FROMPY_NUM(tangoTypeConst, cpy_type, FN) \
    template<> \
    struct from_py<tangoTypeConst> \
    { \
        typedef TANGO_const2type(tangoTypeConst) TangoScalarType; \
        typedef numeric_limits<TangoScalarType> TangoScalarTypeLimits; \
    \
        static inline void convert(const boost::python::object &o, TangoScalarType &tg) \
        { \
            convert(o.ptr(), tg); \
        } \
    \
        static inline void convert(PyObject *o, TangoScalarType &tg) \
        { \
            cpy_type cpy_value = FN(o); \
            if(PyErr_Occurred()) { \
	        PyErr_Clear(); \
                PyErr_SetString(PyExc_TypeError, "Expecting a numeric type, it is not."); \
                boost::python::throw_error_already_set();  \
            } \
            if (TangoScalarTypeLimits::is_integer) { \
                if (cpy_value > TangoScalarTypeLimits::max()) {	\
                    PyErr_SetString(PyExc_OverflowError, "Value is too large."); \
                    boost::python::throw_error_already_set(); \
                } \
                if (cpy_value < TangoScalarTypeLimits::min()) {	\
                    PyErr_SetString(PyExc_OverflowError, "Value is too small."); \
                    boost::python::throw_error_already_set(); \
                } \
            } \
            tg = static_cast<TangoScalarType>(cpy_value);  \
        } \
    };
#else // DISABLE_PYTANGO_NUMPY
# define DEFINE_FAST_TANGO_FROMPY_NUM(tangoTypeConst, cpy_type, FN) \
    template<> \
    struct from_py<tangoTypeConst> \
    { \
        typedef TANGO_const2type(tangoTypeConst) TangoScalarType; \
        typedef numeric_limits<TangoScalarType> TangoScalarTypeLimits; \
    \
        static inline void convert(const boost::python::object &o, TangoScalarType &tg) \
        { \
            convert(o.ptr(), tg); \
        } \
    \
        static inline void convert(PyObject *o, TangoScalarType &tg) \
        { \
            cpy_type cpy_value = FN(o); \
            if(PyErr_Occurred()) { \
	        PyErr_Clear(); \
                if(PyArray_CheckScalar(o) && \
                ( PyArray_DescrFromScalar(o) \
                    == PyArray_DescrFromType(TANGO_const2numpy(tangoTypeConst)))) \
                { \
                    PyArray_ScalarAsCtype(o, reinterpret_cast<void*>(&tg)); \
                    return; \
                } else { \
                    PyErr_SetString(PyExc_TypeError, "Expecting a numeric type," \
                        " but it is not. If you use a numpy type instead of" \
                        " python core types, then it must exactly match (ex:" \
                        " numpy.int32 for PyTango.DevLong)"); \
                    boost::python::throw_error_already_set();  \
		} \
            } \
            if (TangoScalarTypeLimits::is_integer) { \
                if (cpy_value > (cpy_type)TangoScalarTypeLimits::max()) { \
                    PyErr_SetString(PyExc_OverflowError, "Value is too large."); \
                    boost::python::throw_error_already_set(); \
                } \
                if (cpy_value < (cpy_type)TangoScalarTypeLimits::min()) { \
                    PyErr_SetString(PyExc_OverflowError, "Value is too small."); \
                    boost::python::throw_error_already_set(); \
                } \
            } \
            tg = static_cast<TangoScalarType>(cpy_value);  \
        } \
    };
#endif // !DISABLE_PYTANGO_NUMPY


/* Allow for downcast */

inline unsigned PY_LONG_LONG PyLong_AsUnsignedLongLong_2(PyObject *pylong)
{
  unsigned PY_LONG_LONG result = PyLong_AsUnsignedLongLong(pylong);
  if(PyErr_Occurred())
  {
    PyErr_Clear();
    result = PyLong_AsUnsignedLong(pylong);
  }
  return result;
}

DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_BOOLEAN, long, PyLong_AsLong)
DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_UCHAR, unsigned long, PyLong_AsUnsignedLong)
DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_SHORT, long, PyLong_AsLong)
DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_USHORT, unsigned long, PyLong_AsUnsignedLong)
DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_LONG, long, PyLong_AsLong)
DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_ULONG, unsigned long, PyLong_AsUnsignedLong)
DEFINE_FAST_TANGO_FROMPY(Tango::DEV_STATE, PyLong_AsLong)

DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_LONG64, Tango::DevLong64, PyLong_AsLongLong)
DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_ULONG64, Tango::DevULong64, PyLong_AsUnsignedLongLong_2)
DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_FLOAT, double, PyFloat_AsDouble)
DEFINE_FAST_TANGO_FROMPY_NUM(Tango::DEV_DOUBLE, double, PyFloat_AsDouble)

// DEFINE_FAST_TANGO_FROMPY(Tango::DEV_STRING, PyString_AsString)
DEFINE_FAST_TANGO_FROMPY(Tango::DEV_STRING, PyString_AsCorbaString)
DEFINE_FAST_TANGO_FROMPY(Tango::DEV_ENUM, PyLong_AsUnsignedLong)

template<long tangoArrayTypeConst>
struct array_element_from_py : public from_py<TANGO_const2scalarconst(tangoArrayTypeConst)>
{ };

template<>
struct array_element_from_py<Tango::DEVVAR_CHARARRAY>
{
    static const long tangoArrayTypeConst = Tango::DEVVAR_CHARARRAY;

    typedef TANGO_const2scalartype(tangoArrayTypeConst) TangoScalarType;
    typedef numeric_limits<TangoScalarType> TangoScalarTypeLimits;

    static inline void convert(const boost::python::object &o, TangoScalarType &tg)
    {
        convert(o.ptr(), tg);
    }

#ifdef DISABLE_PYTANGO_NUMPY
    static inline void convert(PyObject *o, TangoScalarType &tg)
    {
        long cpy_value = PyLong_AsLong(o);
        if(PyErr_Occurred()) {
            PyErr_Clear();
            PyErr_SetString(PyExc_TypeError, "Expecting a numeric type,"
                " but it is not");
            boost::python::throw_error_already_set(); 
        }
        if (TangoScalarTypeLimits::is_integer) {
            if (cpy_value > TangoScalarTypeLimits::max()) {
                PyErr_SetString(PyExc_OverflowError, "Value is too large.");
                boost::python::throw_error_already_set();
            }
            if (cpy_value < TangoScalarTypeLimits::min()) {
                PyErr_SetString(PyExc_OverflowError, "Value is too small.");
                boost::python::throw_error_already_set();
            }
        }
        tg = static_cast<TangoScalarType>(cpy_value);
    }
#else
    static inline void convert(PyObject *o, TangoScalarType &tg)
    {
        long cpy_value = PyLong_AsLong(o);
        if(PyErr_Occurred()) {
	    PyErr_Clear();
            if(PyArray_CheckScalar(o) &&
            ( PyArray_DescrFromScalar(o)
                == PyArray_DescrFromType(TANGO_const2scalarnumpy(tangoArrayTypeConst))))
            {
                PyArray_ScalarAsCtype(o, reinterpret_cast<void*>(&tg));
                return;
            } else {
                PyErr_SetString(PyExc_TypeError, "Expecting a numeric type,"
                    " but it is not. If you use a numpy type instead of"
                    " python core types, then it must exactly match (ex:"
                    " numpy.int32 for PyTango.DevLong)");
                boost::python::throw_error_already_set();
            }
        }
        if (TangoScalarTypeLimits::is_integer) {
            if (cpy_value > TangoScalarTypeLimits::max()) {
                PyErr_SetString(PyExc_OverflowError, "Value is too large.");
                boost::python::throw_error_already_set();
            }
            if (cpy_value < TangoScalarTypeLimits::min()) {
                PyErr_SetString(PyExc_OverflowError, "Value is too small.");
                boost::python::throw_error_already_set();
            }
        }
        tg = static_cast<TangoScalarType>(cpy_value);
    }
#endif // DISABLE_PYTANGO_NUMPY

};


template<long tangoTypeConst>
inline void fast_python_to_tango_buffer_deleter__(typename TANGO_const2type(tangoTypeConst)* data_buffer, long processedElements)
{
    delete [] data_buffer;
}

template<>
inline void fast_python_to_tango_buffer_deleter__<Tango::DEV_STRING>(Tango::DevString* data_buffer, long processedElements)
{
    for (long i =0; i < processedElements; ++i) {
        delete [] data_buffer[i];
    }
    delete [] data_buffer;
}

template<long tangoTypeConst>
inline typename TANGO_const2type(tangoTypeConst)*
    fast_python_to_tango_buffer_sequence(PyObject* py_val, long* pdim_x, long *pdim_y, const std::string &fname, bool isImage, long& res_dim_x, long& res_dim_y)
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
 
    long dim_x;
    long dim_y = 0;
    Py_ssize_t len = PySequence_Size(py_val);
    bool expectFlatSource;

    if (isImage) {
        if (pdim_y) {
            expectFlatSource = true;
            dim_x = *pdim_x;
            dim_y = *pdim_y;
            long len2 = dim_x*dim_y;
            if (len2 < len)
                len = len2;
        } else {
            expectFlatSource = false;

            if (len > 0) {
                PyObject* py_row0 = PySequence_ITEM(py_val, 0);
                if (!py_row0 || !PySequence_Check(py_row0)) {
                    Py_XDECREF(py_row0);
                    Tango::Except::throw_exception(
                        "PyDs_WrongParameters",
                        "Expecting a sequence of sequences.",
                        fname + "()");
                }

                dim_y = static_cast<long>(len);
                dim_x = static_cast<long>(PySequence_Size(py_row0));
                Py_XDECREF(py_row0);
            } else {
                dim_x = 0;
            }
        }
        len = dim_x*dim_y;
    } else {
        expectFlatSource = true;
        if (pdim_x) {
            if (*pdim_x > len)
                Tango::Except::throw_exception(
                    "PyDs_WrongParameters",
                    "Specified dim_x is larger than the sequence size",
                    fname + "()");
            len = *pdim_x;
        }
        if (pdim_y && (*pdim_y!=0))
            Tango::Except::throw_exception(
                    "PyDs_WrongParameters",
                    "You should not specify dim_y for an spectrum attribute!",
                    fname + "()");
        dim_x = static_cast<long>(len);
    }

    res_dim_x = dim_x;
    res_dim_y = dim_y;

    if (!PySequence_Check(py_val))
        Tango::Except::throw_exception(
            "PyDs_WrongParameters",
            "Expecting a sequence!",
            fname + "()");

    /// @bug Why not TangoArrayType::allocbuf(len)? Because
    /// I will use it in set_value(tg_ptr,...,release=true).
    /// Tango API makes delete[] tg_ptr instead of freebuf(tg_ptr).
    /// This is usually the same, but for Tango::DevStringArray the
    /// behaviour seems different and causes weirdtroubles..
    TangoScalarType *tg_ptr;
    tg_ptr = new TangoScalarType[len];

    // The boost extract could be used:
    // TangoScalarType val = boost::python::extract<TangoScalarType>(elt_ptr);
    // instead of the code below.
    // the problem is that extract is considerably slower than our
    // convert function which only has to deal with the specific tango
    // data types

    PyObject * py_el = 0;
    PyObject * py_row = 0;
    TangoScalarType tg_scalar;
    long idx = 0;
    try {
        if (expectFlatSource) {
            for (idx = 0; idx < len; ++idx)
            {
                py_el = PySequence_ITEM(py_val, idx);
                if (!py_el)
                        boost::python::throw_error_already_set();
                
                from_py<tangoTypeConst>::convert(py_el, tg_scalar);
                tg_ptr[idx] = tg_scalar;
                
                Py_DECREF(py_el);
                py_el = 0;
            }
        } else {
            for (long y=0; y < dim_y; ++y) {
                py_row = PySequence_ITEM(py_val, y);
                if (!py_row)
                        boost::python::throw_error_already_set();
                if (!PySequence_Check(py_row)) {
                    Tango::Except::throw_exception(
                        "PyDs_WrongParameters",
                        "Expecting a sequence of sequences!",
                        fname + "()");
                }
                for (long x=0; x < dim_x; ++x, ++idx) {
                    py_el = PySequence_ITEM(py_row, x);
                    if (!py_el)
                        boost::python::throw_error_already_set();
                    
                    from_py<tangoTypeConst>::convert(py_el, tg_scalar);
                    tg_ptr[x + y*dim_x] = tg_scalar;
                    
                    Py_DECREF(py_el);
                    py_el = 0;
                }
                Py_DECREF(py_row);
                py_row = 0;
            }
        }
    } catch(...) {
        Py_XDECREF(py_el);
        Py_XDECREF(py_row);
        fast_python_to_tango_buffer_deleter__<tangoTypeConst>(tg_ptr, idx);
        throw;
    }
    return tg_ptr;
}

template<long tangoArrayTypeConst>
inline typename TANGO_const2scalartype(tangoArrayTypeConst)*
    fast_python_to_corba_buffer_sequence(PyObject* py_val, long* pdim_x, const std::string &fname, long& res_dim_x)
{
    typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
    typedef typename TANGO_const2scalartype(tangoArrayTypeConst) TangoScalarType;

    long dim_x;
    Py_ssize_t len = PySequence_Size(py_val);

    if (pdim_x) {
        if (*pdim_x > len)
            Tango::Except::throw_exception(
                "PyDs_WrongParameters",
                "Specified dim_x is larger than the sequence size",
                fname + "()");
        len = *pdim_x;
    }
    dim_x = static_cast<long>(len);

    res_dim_x = dim_x;

    if (!PySequence_Check(py_val))
        Tango::Except::throw_exception(
            "PyDs_WrongParameters",
            "Expecting a sequence!",
            fname + "()");

	TangoScalarType* tg_ptr = TangoArrayType::allocbuf(static_cast<Tango::DevULong>(len));

    // The boost extract could be used:
    // TangoScalarType val = boost::python::extract<TangoScalarType>(elt_ptr);
    // instead of the code below.
    // the problem is that extract is considerably slower than our
    // convert function which only has to deal with the specific tango
    // data types

    PyObject * py_el = 0;
    TangoScalarType tg_scalar;
    long idx = 0;
    try {
        for (idx = 0; idx < len; ++idx)
        {
            py_el = PySequence_ITEM(py_val, idx);
            if (!py_el)
                    boost::python::throw_error_already_set();

            array_element_from_py<tangoArrayTypeConst>::convert(py_el, tg_scalar);
            tg_ptr[idx] = tg_scalar;
            
            Py_DECREF(py_el);
            py_el = 0;
        }
    } catch(...) {
        Py_XDECREF(py_el);
        TangoArrayType::freebuf(tg_ptr);
        throw;
    }
    return tg_ptr;
}

template<>
inline TANGO_const2type(Tango::DEV_ENCODED)*
    fast_python_to_tango_buffer_sequence<Tango::DEV_ENCODED>(PyObject*, long*, long*, const std::string & fname, bool isImage, long& res_dim_x, long& res_dim_y)
{
    TangoSys_OMemStream o;
    o << "DevEncoded is only supported for SCALAR attributes." << ends;
    Tango::Except::throw_exception(
            "PyDs_WrongPythonDataTypeForAttribute",
            o.str(), fname + "()");
    return 0;
}

# ifndef DISABLE_PYTANGO_NUMPY
#   include "fast_from_py_numpy.hpp"
#   define fast_python_to_tango_buffer fast_python_to_tango_buffer_numpy
#   define fast_python_to_corba_buffer fast_python_to_corba_buffer_numpy
# else
#   define fast_python_to_tango_buffer fast_python_to_tango_buffer_sequence
#   define fast_python_to_corba_buffer fast_python_to_corba_buffer_sequence
# endif

template<long tangoArrayTypeConst>
inline typename TANGO_const2type(tangoArrayTypeConst)* fast_convert2array(boost::python::object o)
{
    typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
    typedef typename TANGO_const2scalartype(tangoArrayTypeConst) TangoScalarType;

    long res_dim_x;
    
    // Last parameter false: destruction will be handled by CORBA, not by
    // Tango. So, when we destroy it manually later, we also have to use the
    // CORBA style (TangoArrayType are defines by CORBA idl)
    TangoScalarType* array = fast_python_to_corba_buffer<tangoArrayTypeConst>(o.ptr(), 0, "insert_array", res_dim_x);

    try {
        // not a bug: res_dim_y means nothing to us, we are unidimensional
        // here we have max_len and currebt_len = res_dim_x
        return new TangoArrayType(res_dim_x, res_dim_x, array, true);
    } catch(...) {
        TangoArrayType::freebuf(array);
        throw;
    }
    return 0;
}

template<>
inline TANGO_const2type(Tango::DEVVAR_LONGSTRINGARRAY)* fast_convert2array<Tango::DEVVAR_LONGSTRINGARRAY>(boost::python::object py_value)
{
    const long tangoArrayTypeConst = Tango::DEVVAR_LONGSTRINGARRAY;
    typedef TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

    if (!PySequence_Check(py_value.ptr()))
    {
        raise_convert2array_DevVarLongStringArray();
    }
    
    size_t size = boost::python::len(py_value);
    if (size != 2)
    {
        raise_convert2array_DevVarLongStringArray();
    }
    
    const boost::python::object
        &py_lng = py_value[0],
        &py_str = py_value[1];

    unique_pointer<Tango::DevVarLongArray> a_lng(
        fast_convert2array<Tango::DEVVAR_LONGARRAY>(py_lng));

    unique_pointer<Tango::DevVarStringArray> a_str(
        fast_convert2array<Tango::DEVVAR_STRINGARRAY>(py_str));

    unique_pointer<TangoArrayType> result(new TangoArrayType());

    result->lvalue = *a_lng;
    result->svalue = *a_str;

    return result.release();
}

template<>
inline TANGO_const2type(Tango::DEVVAR_DOUBLESTRINGARRAY)* fast_convert2array<Tango::DEVVAR_DOUBLESTRINGARRAY>(boost::python::object py_value)
{
    const long tangoArrayTypeConst = Tango::DEVVAR_DOUBLESTRINGARRAY;
    typedef TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

    if (!PySequence_Check(py_value.ptr()))
    {
        raise_convert2array_DevVarDoubleStringArray();
    }
    
    size_t size = boost::python::len(py_value);
    if (size != 2)
    {
        raise_convert2array_DevVarDoubleStringArray();
    }

    const boost::python::object
        &py_dbl = py_value[0],
        &py_str = py_value[1];

    unique_pointer<Tango::DevVarDoubleArray> a_dbl(
        fast_convert2array<Tango::DEVVAR_DOUBLEARRAY>(py_dbl));

    unique_pointer<Tango::DevVarStringArray> a_str(
        fast_convert2array<Tango::DEVVAR_STRINGARRAY>(py_str));

    unique_pointer<TangoArrayType> result(new TangoArrayType());

    result->dvalue = *a_dbl;
    result->svalue = *a_str;

    return result.release();
}

