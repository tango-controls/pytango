/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#ifdef DISABLE_PYTANGO_NUMPY

template<long tangoTypeConst>
struct convert_numpy_to_integer {
    convert_numpy_to_integer() {}
};

template<long tangoTypeConst>
struct convert_numpy_to_float {
    convert_numpy_to_float() {}
};

#else

template<long tangoTypeConst>
struct convert_numpy_to_integer
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    static const long NumpyType = TANGO_const2numpy(tangoTypeConst);

    convert_numpy_to_integer()
    {
        boost::python::converter::registry::push_back(
            &convertible,
            &construct,
            boost::python::type_id<TangoScalarType>());
    }

    static void* convertible(PyObject* obj)
    {
        if (!PyArray_CheckScalar(obj))
            return 0;

        PyArray_Descr* type = PyArray_DescrFromScalar(obj);
        if (PyDataType_ISINTEGER(type)) {
            return obj;
        }
        return 0;
    }

    static void construct(PyObject* obj,
                          boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        typedef boost::python::converter::rvalue_from_python_storage<TangoScalarType> tango_storage;
        void* const storage = reinterpret_cast<tango_storage*>(data)->storage.bytes;
        TangoScalarType *ptr = new (storage) TangoScalarType();

        PyObject* native_obj = PyObject_CallMethod(obj, const_cast<char*>("__int__"), NULL);
        if (native_obj == NULL) {
            boost::python::throw_error_already_set();
        }
        from_py<tangoTypeConst>::convert(native_obj, *ptr);
        Py_DECREF(native_obj);

        data->convertible = storage;
    }

};



template<long tangoTypeConst>
struct convert_numpy_to_float
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    static const long NumpyType = TANGO_const2numpy(tangoTypeConst);

    convert_numpy_to_float()
    {
        boost::python::converter::registry::push_back(
            &convertible,
            &construct,
            boost::python::type_id<TangoScalarType>());
    }

    static void* convertible(PyObject* obj)
    {
        if (!PyArray_CheckScalar(obj))
            return 0;

        PyArray_Descr* type = PyArray_DescrFromScalar(obj);
        if (PyDataType_ISINTEGER(type) || PyDataType_ISFLOAT(type)) {
            return obj;
        }
        return 0;
    }

    static void construct(PyObject* obj,
                          boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        typedef boost::python::converter::rvalue_from_python_storage<TangoScalarType> tango_storage;
        void* const storage = reinterpret_cast<tango_storage*>(data)->storage.bytes;
        TangoScalarType *ptr = new (storage) TangoScalarType();

        PyObject* native_obj = PyObject_CallMethod(obj, const_cast<char*>("__float__"), NULL);
        if (native_obj == NULL) {
            boost::python::throw_error_already_set();
        }
        from_py<tangoTypeConst>::convert(native_obj, *ptr);
        Py_DECREF(native_obj);

        data->convertible = storage;
    }
};

#endif
