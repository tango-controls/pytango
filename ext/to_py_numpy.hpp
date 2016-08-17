/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

/// @name Array extraction
/// @{

template <long tangoArrayTypeConst>
inline boost::python::object to_py_numpy(const typename TANGO_const2type(tangoArrayTypeConst)* tg_array, boost::python::object parent)
{
    static const int typenum = TANGO_const2scalarnumpy(tangoArrayTypeConst);

    if (tg_array == 0) {
        // Empty
        PyObject* value = PyArray_SimpleNew(0, 0, typenum);
        if (!value)
            boost::python::throw_error_already_set();
        return boost::python::object(boost::python::handle<>(value));
    }
    
    // Create a new numpy.ndarray() object. It uses ch_ptr as the data,
    // so no costy memory copies when handling big images.
    const void *ch_ptr = reinterpret_cast<const void *>(tg_array->get_buffer());
    int nd = 1;
    npy_intp dims[1];
    dims[0]= tg_array->length();
    PyObject* py_array = PyArray_SimpleNewFromData(nd, dims, typenum, const_cast<void*>(ch_ptr));
    if (!py_array) {
        boost::python::throw_error_already_set();
    }

    // numpy.ndarray() does not own it's memory, so we need to manage it.
    // We can assign a 'parent' object that will be informed (decref'd)
    // when the last copy of numpy.ndarray() disappears. That should
    // actually destroy the memory in its destructor
    PyObject* guard = parent.ptr();
    Py_INCREF(guard);
    PyArray_BASE(py_array) = guard;
    
    return boost::python::object(boost::python::handle<>(py_array));
}

template <>
inline boost::python::object to_py_numpy<Tango::DEVVAR_STRINGARRAY>(const Tango::DevVarStringArray* tg_array, boost::python::object parent)
{
    return to_py_list(tg_array);
}

template <>
inline boost::python::object to_py_numpy<Tango::DEVVAR_STATEARRAY>(const Tango::DevVarStateArray* tg_array, boost::python::object parent)
{
    return to_py_list(tg_array);
}

template <>
inline boost::python::object to_py_numpy<Tango::DEVVAR_LONGSTRINGARRAY>(const Tango::DevVarLongStringArray* tg_array, boost::python::object parent)
{
    boost::python::list result;
    
    result.append(to_py_numpy<Tango::DEVVAR_LONGARRAY>(&tg_array->lvalue, parent));
    result.append(to_py_numpy<Tango::DEVVAR_STRINGARRAY>(&tg_array->svalue, parent));
    
    return result;
}

template <>
inline boost::python::object to_py_numpy<Tango::DEVVAR_DOUBLESTRINGARRAY>(const Tango::DevVarDoubleStringArray* tg_array, boost::python::object parent)
{
    boost::python::list result;
    
    result.append(to_py_numpy<Tango::DEVVAR_DOUBLEARRAY>(&tg_array->dvalue, parent));
    result.append(to_py_numpy<Tango::DEVVAR_STRINGARRAY>(&tg_array->svalue, parent));
    
    return result;
}
/// @}
// ~Array Extraction
// -----------------------------------------------------------------------

template <long tangoArrayTypeConst>
inline boost::python::object to_py_numpy(typename TANGO_const2type(tangoArrayTypeConst)* tg_array, int orphan)
{
    static const int typenum = TANGO_const2scalarnumpy(tangoArrayTypeConst);

    if (tg_array == 0) {
        // Empty
        PyObject* value = PyArray_SimpleNew(0, 0, typenum);
        if (!value)
            boost::python::throw_error_already_set();
        return boost::python::object(boost::python::handle<>(value));
    }

    // Create a new numpy.ndarray() object. It uses ch_ptr as the data,
    // so no costy memory copies when handling big images.
    int nd = 1;
    npy_intp dims[1];
    dims[0]= tg_array->length();
    void *ch_ptr = (void *)(tg_array->get_buffer(orphan));
    PyObject* py_array = PyArray_New(&PyArray_Type, nd, dims, typenum, NULL,
				     ch_ptr, -1, 0, NULL);
    if (!py_array) {
        boost::python::throw_error_already_set();
    }

    return boost::python::object(boost::python::handle<>(py_array));
}

template <>
inline boost::python::object to_py_numpy<Tango::DEVVAR_STRINGARRAY>(Tango::DevVarStringArray* tg_array,
								    int orphan)
{
    return to_py_list(tg_array);
}

template <>
inline boost::python::object to_py_numpy<Tango::DEVVAR_STATEARRAY>(Tango::DevVarStateArray* tg_array,
								   int orphan)
{
    return to_py_list(tg_array);
}

template <>
inline boost::python::object to_py_numpy<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::DevVarLongStringArray* tg_array,
									int orphan)
{
    boost::python::list result;

    result.append(to_py_numpy<Tango::DEVVAR_LONGARRAY>(&tg_array->lvalue, orphan));
    result.append(to_py_numpy<Tango::DEVVAR_STRINGARRAY>(&tg_array->svalue, orphan));

    return result;
}

template <>
inline boost::python::object to_py_numpy<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::DevVarDoubleStringArray* tg_array,
									  int orphan)
{
    boost::python::list result;

    result.append(to_py_numpy<Tango::DEVVAR_DOUBLEARRAY>(&tg_array->dvalue, orphan));
    result.append(to_py_numpy<Tango::DEVVAR_STRINGARRAY>(&tg_array->svalue, orphan));

    return result;
}
