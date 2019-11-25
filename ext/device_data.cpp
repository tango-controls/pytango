/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pyutils.h>
#include <defs.h>
#include <iostream>
#include <tgutils.h>

namespace py = pybind11;

namespace PyDeviceData {

template <long tangoTypeConst>
    void insert_scalar(Tango::DeviceData& self, py::object py_value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType value = py_value.cast<TangoScalarType>();
        self << value;
    }

    template <>
    void insert_scalar<Tango::DEV_STRING>(Tango::DeviceData& self, py::object py_value)
    {
        std::string val_str = py_value.cast<std::string>();
        self << val_str;
    }

    template <>
    void insert_scalar<Tango::DEV_ENCODED>(Tango::DeviceData& self, py::object py_value)
    {
        py::tuple tup(py_value);
        std::string encoded_format = tup[0].cast<std::string>();
        py::list encoded_data = tup[1];
        long len = py::len(encoded_data);
        unsigned char* bptr = new unsigned char[len];
        for (auto& item : encoded_data) {
            *bptr++ = item.cast<unsigned char>();
        }
        Tango::DevVarCharArray array(len, len, bptr-len, false);
        Tango::DevEncoded value;
        value.encoded_format = strdup(encoded_format.c_str());
        value.encoded_data = array;
        self << value;
    }

    template <>
    void insert_scalar<Tango::DEV_VOID>(Tango::DeviceData& self, py::object py_value)
    {
        raise_(PyExc_TypeError, "Trying to insert a value in a DEV_VOID DeviceData!");
    }

    template <>
    void insert_scalar<Tango::DEV_PIPE_BLOB>(Tango::DeviceData& self, py::object py_value)
    {
        assert(false);
    }

    template <long tangoArrayTypeConst>
    void insert_array(Tango::DeviceData& self, py::list py_value)
    {
        typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
        typedef typename TANGO_const2scalartype(tangoArrayTypeConst) TangoScalarType;

        long len = py::len(py_value);
        TangoArrayType* array = new TangoArrayType();
        TangoScalarType value;
        array->length(len);
        for (auto i=0; i<len; i++) {
            value = py_value[i].cast<TangoScalarType>();
            (*array)[i] = value;
        }
        self << array;
    }

    template <>
    void insert_array<Tango::DEVVAR_STRINGARRAY>(Tango::DeviceData& self, py::list py_value)
    {
        Tango::DevVarStringArray *array = new Tango::DevVarStringArray();
        long len = py::len(py_value);
        array->length(len);
        for (auto i=0; i<len; i++) {
            std::string ss = py_value[i].cast<std::string>();
            (*array)[i] = strdup(ss.c_str());
        }
        self << array;
    }

    template <>
    void insert_array<Tango::DEVVAR_STATEARRAY>(Tango::DeviceData& self, py::list py_value)
    {
        assert(false);
    }

    template <>
    void insert_array<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::DeviceData& self, py::list py_value)
    {
        py::tuple tup = py_value;
        py::list long_data = tup[0];
        py::list string_data = tup[1];
        long llen = py::len(long_data);
        long slen = py::len(string_data);
        Tango::DevVarLongStringArray *array = new Tango::DevVarLongStringArray();
        array->lvalue.length(llen);
        for (auto i=0; i<llen; i++) {
            (array->lvalue)[i] = long_data[i].cast<long>();
        }
        array->svalue.length(slen);
        for (auto i=0; i<slen; i++) {
            std::string ss = string_data[i].cast<std::string>();
            (array->svalue)[i] = strdup(ss.c_str());
        }
        self << array;
    }

    template <>
    void insert_array<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::DeviceData& self, py::list py_value)
    {
        py::tuple tup = py_value;
        py::list double_data = tup[0];
        py::list string_data = tup[1];
        long dlen = py::len(double_data);
        long slen = py::len(string_data);
        Tango::DevVarDoubleStringArray *array = new Tango::DevVarDoubleStringArray();
        array->dvalue.length(dlen);
        for (auto i=0; i<dlen; i++) {
            (array->dvalue)[i] = double_data[i].cast<double>();
        }
        array->svalue.length(slen);
        for (auto i=0; i<slen; i++) {
            std::string ss = string_data[i].cast<std::string>();
            (array->svalue)[i] = strdup(ss.c_str());
        }
        self << array;
    }

    template <long tangoTypeConst>
    py::object extract_scalar(Tango::DeviceData& self)
    {
        typedef typename TANGO_const2type(tangoTypeConst)TangoScalarType;
        /// @todo CONST_DEV_STRING ell el tracta com DEV_STRING
        TangoScalarType val;
        self >> val;
        return py::cast(val);
    }

    template<>
    py::object extract_scalar<Tango::DEV_VOID>(Tango::DeviceData& self) {
        assert(false);
        return py::none();
    }

    template<>
    py::object extract_scalar<Tango::DEV_STRING>(Tango::DeviceData& self) {
        std::string val;
        self >> val;
        return py::str(val);
    }

    template<>
    py::object extract_scalar<Tango::DEV_ENCODED>(Tango::DeviceData& self) {
        Tango::DevEncoded val;
        self >> val;
        py::str encoded_format = strdup(val.encoded_format);
        py::list encoded_data;
        unsigned int len = val.encoded_data.length();
        for (auto i=0; i<len; i++) {
            encoded_data.append(val.encoded_data[i]);
        }
        return py::make_tuple(encoded_format, encoded_data);
   }

    template<>
    py::object extract_scalar<Tango::DEV_PIPE_BLOB>(Tango::DeviceData& self) {
        assert(false);
        return py::none();
    }

    template<long tangoArrayTypeConst>
    py::object extract_array(Tango::DeviceData& self) {
       typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
       typedef typename TANGO_const2scalartype(tangoArrayTypeConst) TangoScalarType;
       const TangoArrayType* tmp_arr;
       self >> tmp_arr;
       py::array array;
       int dims[1];
       dims[0] = tmp_arr->length();
       TangoScalarType* buffer = const_cast<TangoScalarType*>(tmp_arr->get_buffer());
       array = py::array_t<TangoScalarType>(dims,
               reinterpret_cast<TangoScalarType*>(buffer));
       return array;
    }

    template<>
    py::object extract_array<Tango::DEVVAR_STRINGARRAY>(Tango::DeviceData& self) {
        py::list data;
        const Tango::DevVarStringArray* array;
        self >> array;
        for (auto i=0; i<array->length(); i++) {
            data.append(py::str((*array)[i]));
        }
        return data;
    }

    template <>
    py::object extract_array<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::DeviceData& self)
    {
        py::list long_data;
        py::list string_data;
        const Tango::DevVarLongStringArray *array = NULL;
        self >> array;
        for (auto i=0; i<array->lvalue.length(); i++) {
            long_data.append(py::cast((array->lvalue)[i]));
        }
        for (auto i=0; i<array->svalue.length(); i++) {
            string_data.append(py::str((array->svalue)[i]));
        }
        return py::make_tuple(long_data, string_data);
    }

    template <>
    py::object extract_array<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::DeviceData& self)
    {
        py::list double_data;
        py::list string_data;
        const Tango::DevVarDoubleStringArray *array = NULL;
        self >> array;
        for (auto i=0; i<array->dvalue.length(); i++) {
            double_data.append(py::cast((array->dvalue)[i]));
        }
        for (auto i=0; i<array->svalue.length(); i++) {
            string_data.append(py::str((array->svalue)[i]));
        }
        return py::make_tuple(double_data, string_data);
    }

    template <>
    py::object extract_array<Tango::DEVVAR_STATEARRAY>(Tango::DeviceData& self)
    {
        assert(false);
        return py::none();
    }
}

void export_device_data(py::module& m) {

    py::class_<Tango::DeviceData>(m, "DeviceData")
        .def(py::init<>())
        .def(py::init<const Tango::DeviceData &>())

        .def("extract", [](Tango::DeviceData& self) -> py::object {
            TANGO_DO_ON_DEVICE_DATA_TYPE_ID(self.get_type(),
                    return PyDeviceData::extract_scalar<tangoTypeConst>(self);,
                    return PyDeviceData::extract_array<tangoTypeConst>(self);
            );
            return py::none();
        })
        .def("insert", [](Tango::DeviceData& self, const long data_type, py::object py_value) -> void {
                TANGO_DO_ON_DEVICE_DATA_TYPE_ID(data_type,
                PyDeviceData::insert_scalar<tangoTypeConst>(self, py_value);,
                PyDeviceData::insert_array<tangoTypeConst>(self, py_value);
            );
        })
        .def("is_empty", [](Tango::DeviceData& self) {
            // do not throw exceptions return a boolean
            self.reset_exceptions(Tango::DeviceData::isempty_flag);
            bool empty = self.is_empty();
            self.set_exceptions(Tango::DeviceData::isempty_flag);
            return empty;
        })
        .def("get_type", [](Tango::DeviceData& self) {
            // This should change in Tango itself, get_type should not return int!!
            return static_cast<Tango::CmdArgType>(self.get_type());
        })
    ;
}
