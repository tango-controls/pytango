/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>
#include <pyutils.h>
#include <defs.h>
#include <iostream>
#include <tgutils.h>
//#include <to_py_numpy.hpp>

namespace py = pybind11;

namespace PyDeviceData {

    template <long tangoTypeConst>
    void insert_scalar(Tango::DeviceData &self, py::object py_value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType value;
//        from_py<tangoTypeConst>::convert(py_value.ptr(), value);
        value = py_value.cast<TangoScalarType>();
        self << value;
    }

    template <>
    void insert_scalar<Tango::DEV_STRING>(Tango::DeviceData &self, py::object py_value)
    {
        assert(false);
//        PyObject* py_value_ptr = py_value.ptr();
//        if(PyUnicode_Check(py_value_ptr)) {
//            PyObject* obj_bytes_ptr = PyUnicode_AsLatin1String(py_value_ptr);
//            Tango::DevString value = PyBytes_AsString(obj_bytes_ptr);
//            self << value;
//            Py_DECREF(obj_bytes_ptr);
//        } else {
//            Tango::DevString value = PyBytes_AsString(py_value_ptr);
//            self << value;
//        }
    }

    template <>
    void insert_scalar<Tango::DEV_ENCODED>(Tango::DeviceData &self, py::object py_value)
    {
        assert(false);
//        object p0 = py_value[0];
//        object p1 = py_value[1];
//        const char* encoded_format = extract<const char*>(p0.ptr());
//        const char* encoded_data = extract<const char*>(p1.ptr());
//
//        CORBA::ULong nb = static_cast<CORBA::ULong>(boost::python::len(p1));
//        Tango::DevVarCharArray arr(nb, nb, (CORBA::Octet*) encoded_data, false);
//        Tango::DevEncoded val;
//        val.encoded_format = CORBA::string_dup(encoded_format);
//        val.encoded_data = arr;
//        self << val;
    }

    template <>
    void insert_scalar<Tango::DEV_VOID>(Tango::DeviceData &self, py::object py_value)
    {
        assert(false);
//        raise_(PyExc_TypeError, "Trying to insert a value in a DEV_VOID DeviceData!");
    }

    template <>
    void insert_scalar<Tango::DEV_PIPE_BLOB>(Tango::DeviceData &self, py::object py_value)
    {
        assert(false);
    }

    template <long tangoArrayTypeConst>
    void insert_array(Tango::DeviceData &self, py::object py_value)
    {
        typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

            // self << val; -->> This ends up doing:
            // inline void operator << (DevVarUShortArray* datum)
            // { any.inout() <<= datum;}
            // So:
            //  - We loose ownership of the pointer, should not remove it
            //  - it's a CORBA object who gets ownership, not a buggy Tango
            //    thing. So the last parameter to fast_convert2array is false
//        TangoArrayType* val = fast_convert2array<tangoArrayTypeConst>(py_value);
//        self << val;
    }

    template <long tangoTypeConst>
    py::object extract_scalar(Tango::DeviceData &self)
    {
        typedef typename TANGO_const2type(tangoTypeConst)TangoScalarType;
        /// @todo CONST_DEV_STRING ell el tracta com DEV_STRING
        TangoScalarType val;
        self >> val;
        return py::cast(val);
    }

    template<>
    py::object extract_scalar<Tango::DEV_VOID>(Tango::DeviceData &self) {
        assert(false);
        return py::none();
    }

    template<>
    py::object extract_scalar<Tango::DEV_STRING>(Tango::DeviceData &self) {
        std::string val;
        self >> val;
        return py::str(val);
    }

    template<>
    py::object extract_scalar<Tango::DEV_PIPE_BLOB>(Tango::DeviceData &self) {
        assert(false);
        return py::none();
    }

    template<long tangoArrayTypeConst>
    py::object extract_array(Tango::DeviceData &self, PyTango::ExtractAs extract_as) {
        typedef typename TANGO_const2type(tangoArrayTypeConst)TangoArrayType;

        // const is the pointed, not the pointer. So cannot modify the data.
        // And that's because the data is still inside "self" after extracting.
        // This also means that we are not supposed to "delete" tmp_ptr.
        const TangoArrayType* tmp_ptr;
        self >> tmp_ptr;

        switch (extract_as)
        {
            default:
            case PyTango::ExtractAsNumpy:
//              return to_py_numpy<tangoArrayTypeConst>(tmp_ptr, py_self);
            case PyTango::ExtractAsList:
            case PyTango::ExtractAsPyTango3:
//            return to_py_list(tmp_ptr);
            case PyTango::ExtractAsTuple:
//            return to_py_tuple(tmp_ptr);
            case PyTango::ExtractAsString:/// @todo
            case PyTango::ExtractAsNothing:
            return py::none();
        }
    }

        template <>
        py::object extract_array<Tango::DEVVAR_STATEARRAY>(Tango::DeviceData &self, PyTango::ExtractAs extract_as)
        {
            assert(False);
            return py::none();
        }
}

void export_device_data(py::module& m) {

    py::class_<Tango::DeviceData>(m, "DeviceData")
        .def(py::init<>())
        .def(py::init<const Tango::DeviceData &>())

        .def("extract", [](Tango::DeviceData& self, PyTango::ExtractAs extract_as) -> py::object {
//            Tango::DeviceData& self = py_self.cast<Tango::DeviceData>();
            TANGO_DO_ON_DEVICE_DATA_TYPE_ID(self.get_type(),
                    return PyDeviceData::extract_scalar<tangoTypeConst>(self);,
                    return PyDeviceData::extract_array<tangoTypeConst>(self, extract_as);
            );
            return py::none();
        })

        .def("insert", [](Tango::DeviceData &self, const long data_type, py::object py_value) -> void {
            TANGO_DO_ON_DEVICE_DATA_TYPE_ID(data_type,
                PyDeviceData::insert_scalar<tangoTypeConst>(self, py_value);,
                PyDeviceData::insert_array<tangoTypeConst>(self, py_value);
            );
        })
        /// @todo do not throw exceptions!!
        .def("is_empty", &Tango::DeviceData::is_empty)
        .def("get_type", [](Tango::DeviceData &self) {
            /// @todo This should change in Tango itself, get_type should not return int!!
            return static_cast<Tango::CmdArgType>(self.get_type());
        })
    ;
}
