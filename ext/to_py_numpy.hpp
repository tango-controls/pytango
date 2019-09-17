/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include <tango.h>
#include <pybind11/pybind11.h>
#include <tgutils.h>

namespace py = pybind11;

/// @name Array extraction
/// @{

//template <long tangoArrayTypeConst>
//inline py::object to_py_numpy(const typename TANGO_const2type(tangoArrayTypeConst)* tg_array, py::object parent)
//{
//    static const int typenum = TANGO_const2scalarnumpy(tangoArrayTypeConst);
//
//    if (tg_array == 0) {
//        // Empty
//        PyObject* value = PyArray_SimpleNew(0, 0, typenum);
//        if (!value)
//            py::error_already_set();
//        return py::cast(value); //py::object(py::handle(value));
//    }
//
//    // Create a new numpy.ndarray() object. It uses ch_ptr as the data,
//    // so no costy memory copies when handling big images.
//    const void *ch_ptr = reinterpret_cast<const void *>(tg_array->get_buffer());
//    int nd = 1;
//    npy_intp dims[1];
//    dims[0]= tg_array->length();
//    PyObject* py_array = PyArray_SimpleNewFromData(nd, dims, typenum, const_cast<void*>(ch_ptr));
//    if (!py_array) {
//        py::error_already_set();
//    }
//
//    // numpy.ndarray() does not own it's memory, so we need to manage it.
//    // We can assign a 'parent' object that will be informed (decref'd)
//    // when the last copy of numpy.ndarray() disappears. That should
//    // actually destroy the memory in its destructor
//    PyObject* guard = parent.ptr();
//    Py_INCREF(guard);
//    PyArray_BASE(py_array) = guard;
//
//    return py::cast(py_array); //py::object(py::handle(py_array));
//}
//
//template <>
//inline py::object to_py_numpy<Tango::DEVVAR_STRINGARRAY>(const Tango::DevVarStringArray* tg_array, py::object parent)
//{
//    return to_py_list(tg_array);
//}
//
//template <>
//inline py::object to_py_numpy<Tango::DEVVAR_STATEARRAY>(const Tango::DevVarStateArray* tg_array, py::object parent)
//{
//    return to_py_list(tg_array);
//}
//
//template <>
//inline py::object to_py_numpy<Tango::DEVVAR_LONGSTRINGARRAY>(const Tango::DevVarLongStringArray* tg_array, py::object parent)
//{
//    py::list result;
//
//    result.append(to_py_numpy<Tango::DEVVAR_LONGARRAY>(&tg_array->lvalue, parent));
//    result.append(to_py_numpy<Tango::DEVVAR_STRINGARRAY>(&tg_array->svalue, parent));
//
//    return result;
//}
//
//template <>
//inline py::object to_py_numpy<Tango::DEVVAR_DOUBLESTRINGARRAY>(const Tango::DevVarDoubleStringArray* tg_array, py::object parent)
//{
//    py::list result;
//
//    result.append(to_py_numpy<Tango::DEVVAR_DOUBLEARRAY>(&tg_array->dvalue, parent));
//    result.append(to_py_numpy<Tango::DEVVAR_STRINGARRAY>(&tg_array->svalue, parent));
//
//    return result;
//}
/// @}
// ~Array Extraction
// -----------------------------------------------------------------------

template <long tangoTypeConst>
inline py::object to_py_numpy(typename TANGO_const2type(tangoTypeConst)* tg_array)
{
    typedef typename TANGO_const2arrayelementstype(tangoTypeConst) TangoScalarType;

    if (tg_array == 0) {
        // Empty pipe
        py::array value = py::array_t<TangoScalarType>(0, nullptr);
        if (!value)
            throw py::error_already_set();
        return value;
    }
    //    TangoScalarType* buffer = (TangoScalarType*)tg_array->get_buffer();
    // Had to use a copy cos tg_array went out of scope. Not ideal!!!!
    TangoScalarType* buffer = new TangoScalarType[tg_array->length()];
    memcpy(buffer, tg_array->get_buffer(), tg_array->length()*sizeof(TangoScalarType));

    // numpy.ndarray() does not own it's memory, so we need to manage it.
    // We can assign a 'base' object that will be informed (decref'd) when
    // the last copy of numpy.ndarray() disappears.
    // PyCObject is intended for that kind of things. It's seen as a
    // black box object from python. We assign him a function to be called
    // when it is deleted -> the function deletes the data.
    py::capsule free_when_done(reinterpret_cast<void*>(buffer), [](void* f) {
        TangoScalarType *buffer = reinterpret_cast<TangoScalarType *>(f);
        delete[] buffer;
    });

    // Create a new numpy.ndarray() object. It uses a pointer to the data,
    py::array array;
    int dims[1];
    dims[0]= tg_array->length();
    array = py::array_t<TangoScalarType>(dims, buffer, free_when_done);
    if (!array) {
        delete tg_array;
        throw py::error_already_set();
    }
    return array;
}

template <>
inline py::object to_py_numpy<Tango::DEVVAR_STRINGARRAY>(Tango::DevVarStringArray* tg_array)
{
    return to_py_list(tg_array);
}

template <>
inline py::object to_py_numpy<Tango::DEVVAR_STATEARRAY>(Tango::DevVarStateArray* tg_array)
{
    return to_py_list(tg_array);
}

template <>
inline py::object to_py_numpy<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::DevVarLongStringArray* tg_array)
{
    py::list result;

    result.append(to_py_numpy<Tango::DEVVAR_LONGARRAY>(&tg_array->lvalue));
    result.append(to_py_numpy<Tango::DEVVAR_STRINGARRAY>(&tg_array->svalue));

    return result;
}

template <>
inline py::object to_py_numpy<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::DevVarDoubleStringArray* tg_array)
{
    py::list result;

    result.append(to_py_numpy<Tango::DEVVAR_DOUBLEARRAY>(&tg_array->dvalue));
    result.append(to_py_numpy<Tango::DEVVAR_STRINGARRAY>(&tg_array->svalue));

    return result;
}
