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
#include <pybind11/stl.h>
#include "tgutils.h"
//#include "wattribute_numpy.hpp"

namespace py = pybind11;

///**
// * Helper method to Limit the max number of element to send to C++
// *
// * @param[in,out] len the length. if x*y is lower the len, the len is updated to x*y
// * @param[in] x the x dimension
// * @param[in] y the y dimension
// */
//static inline void twod2oned(long &len, long x, long y)
//{
//    if (y <= 0)
//    {
//        if (x < len)
//        {
//            len = x;
//        }
//    }
//    else
//    {
//        long max_elt = x * y;
//        if (max_elt < len)
//            len = max_elt;
//    }
//}
//
//inline static void throw_wrong_python_data_type(const std::string& att_name,
//                                         const std::string& method)
//{
//    std::stringstream o;
//    o << "Wrong Python type for attribute " << att_name << ends;
//    Tango::Except::throw_exception(
//            "PyDs_WrongPythonDataTypeForAttribute",
//            o.str(),
//            method);
//}

namespace PyWAttribute
{
    template<long tangoTypeConst>
    py::object __get_min_value(Tango::WAttribute &self)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType tg_val;
        self.get_min_value(tg_val);
        py::object py_value = py::cast(tg_val);
        return py_value;
    }

    py::object get_min_value(Tango::WAttribute& self)
    {
        long type = self.get_data_type();
        if(type == Tango::DEV_ENCODED)
            type = Tango::DEV_UCHAR;

        py::object minval;
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, minval = __get_min_value, self);
        return minval;
    }

    template<long tangoTypeConst>
    py::object __get_max_value(Tango::WAttribute &self)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;

        TangoScalarType tg_val;
        self.get_max_value(tg_val);
        py::object py_value = py::cast(tg_val);
        return py_value;
    }

    py::object get_max_value(Tango::WAttribute& self)
    {
        long type = self.get_data_type();
        if(type == Tango::DEV_ENCODED)
            type = Tango::DEV_UCHAR;

        py::object maxval;
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, maxval = __get_max_value, self);
        return maxval;
    }

    template<long tangoTypeConst>
    inline void _set_min_value(Tango::WAttribute& self, py::object value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType c_value = value.cast<TangoScalarType>();
        self.set_min_value(c_value);
    }

    template<>
    inline void _set_min_value<Tango::DEV_STRING>(Tango::WAttribute& self, py::object value)
    {
        std::string ss = value.cast<std::string>();
        self.set_min_value(ss.c_str());
    }

    template<>
    inline void _set_min_value<Tango::DEV_ENCODED>(Tango::WAttribute& self, py::object value)
    {
        string err_msg = "Attribute properties cannot be set with Tango::DevEncoded data type";
        Tango::Except::throw_exception("API_MethodArgument", err_msg,
                      "WAttribute::set_min_value()");
    }

    inline void set_min_value(Tango::WAttribute& self, py::object& value)
    {
        long tangoTypeConst = self.get_data_type();
        // TODO: the below line is a neat trick to properly raise a Tango exception if a property is set
        // for one of the forbidden attribute data types; code dependent on Tango C++ implementation
        if(tangoTypeConst == Tango::DEV_BOOLEAN || tangoTypeConst == Tango::DEV_STATE)
            tangoTypeConst = Tango::DEV_DOUBLE;
        else if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(tangoTypeConst, _set_min_value, self, value);
    }

    template<long tangoTypeConst>
    inline void _set_max_value(Tango::WAttribute& self, py::object value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType c_value = value.cast<TangoScalarType>();
        self.set_max_value(c_value);
    }

    template<>
    inline void _set_max_value<Tango::DEV_STRING>(Tango::WAttribute& self, py::object value)
    {
        std::string ss = value.cast<std::string>();
        self.set_max_value(ss.c_str());
    }

    template<>
    inline void _set_max_value<Tango::DEV_ENCODED>(Tango::WAttribute& self, py::object value)
    {
        string err_msg = "Attribute properties cannot be set with Tango::DevEncoded data type";
        Tango::Except::throw_exception((const char *)"API_MethodArgument",
                      (const char *)err_msg.c_str(),
                      (const char *)"WAttribute::set_max_value()");
    }

    inline void set_max_value(Tango::WAttribute& self, py::object& value)
    {
        long tangoTypeConst = self.get_data_type();
        // TODO: the below line is a neat trick to properly raise a Tango exception if a property is set
        // for one of the forbidden attribute data types; code dependent on Tango C++ implementation
        if(tangoTypeConst == Tango::DEV_STRING || tangoTypeConst == Tango::DEV_BOOLEAN || tangoTypeConst == Tango::DEV_STATE)
            tangoTypeConst = Tango::DEV_DOUBLE;
        else if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(tangoTypeConst, _set_max_value, self, value);
    }

    template<long tangoTypeConst>
    inline void __set_write_value_scalar(Tango::WAttribute& att,
                                         py::object& value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType cpp_value = value.cast<TangoScalarType>();
        att.set_write_value(cpp_value);
    }
    template<>
    inline void __set_write_value_scalar<Tango::DEV_STRING>(Tango::WAttribute& att,
                                                             py::object& value)
    {
        std::string ss = value.cast<std::string>();
        att.set_write_value(const_cast<char*>(ss.c_str()));
    }

    template<>
    inline void __set_write_value_scalar<Tango::DEV_ENCODED>(Tango::WAttribute& att,
                                                             py::object& value)
    {
        Tango::Except::throw_exception(
                "PyDs_WrongPythonDataTypeForAttribute",
                "set_write_value is not supported for DEV_ENCODED attributes.",
                "set_write_value()");
    }

//    template<long tangoTypeConst>
//    inline void __set_write_value_array(Tango::WAttribute &att,
//                                        boost::python::object &seq,
//                                        long x_dim, long y_dim)
//    {
//        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
//        typedef typename TANGO_const2arraytype(tangoTypeConst) TangoArrayType;
//
//        PyObject *seq_ptr = seq.ptr();
//        long len = (long) PySequence_Size(seq_ptr);
//        twod2oned(len, x_dim, y_dim);
//
//        TangoScalarType *tg_ptr = TangoArrayType::allocbuf(len);
//
//        for (long idx = 0; idx < len; ++idx)
//        {
//            PyObject *elt_ptr = PySequence_GetItem(seq_ptr, idx);
//
//            // The boost extract could be used:
//            // TangoScalarType val = boost::python::extract<TangoScalarType>(elt_ptr);
//            // instead of the code below.
//            // the problem is that extract is considerably slower than our
//            // convert function which only has to deal with the specific tango
//            // data types
//            try
//            {
//                TangoScalarType tg_scalar;
//                from_py<tangoTypeConst>::convert(elt_ptr, tg_scalar);
//                tg_ptr[idx] = tg_scalar;
//                Py_DECREF(elt_ptr);
//            }
//            catch(...)
//            {
//                Py_DECREF(elt_ptr);
//                delete [] tg_ptr;
//                throw;
//            }
//        }
//
//        try
//        {
//            att.set_write_value(tg_ptr, x_dim, y_dim);
//            delete [] tg_ptr;
//        }
//        catch(...)
//        {
//            delete [] tg_ptr;
//            throw;
//        }
//    }
//
//    template<>
//    inline void __set_write_value_array<Tango::DEV_STRING>(Tango::WAttribute &att,
//                               boost::python::object &seq,
//                               long x_dim, long y_dim)
//    {
//        PyObject *seq_ptr = seq.ptr();
//        long len = (long) PySequence_Size(seq_ptr);
//        twod2oned(len, x_dim, y_dim);
//
//    Tango::DevString* tg_ptr = Tango::DevVarStringArray::allocbuf(len);
//
//        for (long idx = 0; idx < len; ++idx)
//        {
//            PyObject *elt_ptr = PySequence_GetItem(seq_ptr, idx);
//
//            // The boost extract could be used:
//            // TangoScalarType val = boost::python::extract<TangoScalarType>(elt_ptr);
//            // instead of the code below.
//            // the problem is that extract is considerably slower than our
//            // convert function which only has to deal with the specific tango
//            // data types
//            try
//            {
//        Tango::DevString tg_scalar;
//                from_py<Tango::DEV_STRING>::convert(elt_ptr, tg_scalar);
//                tg_ptr[idx] = Tango::string_dup(tg_scalar);
//                Py_DECREF(elt_ptr);
//            }
//            catch(...)
//            {
//                Py_DECREF(elt_ptr);
//                delete [] tg_ptr;
//                throw;
//            }
//        }
//
//        try
//        {
//            att.set_write_value(tg_ptr, x_dim, y_dim);
////            delete [] tg_ptr;
//        }
//        catch(...)
//        {
//            delete [] tg_ptr;
//            throw;
//        }
//    }
//
//    template<>
//    inline void __set_write_value_array<Tango::DEV_ENCODED>(Tango::WAttribute &att,
//                                                            boost::python::object &seq,
//                                                            long x_dim, long y_dim)
//    {
//        Tango::Except::throw_exception(
//                "PyDs_WrongPythonDataTypeForAttribute",
//                "set_write_value is not supported for DEV_ENCODED attributes.",
//                "set_write_value()");
//    }

    inline void set_write_value(Tango::WAttribute& self, py::object& value)
    {
        long type = self.get_data_type();
        Tango::AttrDataFormat format = self.get_data_format();

        if (format == Tango::SCALAR)
        {
            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, __set_write_value_scalar, self, value);
        }
//        else
//        {
//            if (!PySequence_Check(value.ptr()))
//            {
//                std::stringstream o;
//                o << "Wrong Python type for attribute " << att.get_name()
//                  << "of type " << Tango::CmdArgTypeName[type]
//                  << ". Expected a sequence." << ends;
//
//                Tango::Except::throw_exception(
//                        "PyDs_WrongPythonDataTypeForAttribute",
//                        o.str(),
//                        "set_value()");
//            }
//            long size = static_cast<long>(PySequence_Size(value.ptr()));
//            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, __set_write_value_array,
//                                              self, value, size, 0);
//        }
    }

    inline void set_write_value(Tango::WAttribute& self,
                                py::object& value,
                                long x)
    {
        long type = self.get_data_type();
        Tango::AttrDataFormat format = self.get_data_format();
//
//        if (format == Tango::SCALAR)
//        {
//            std::stringstream o;
//            o << "Cannot call set_value(data, dim_x) on scalar attribute "
//              << att.get_name() << ". Use set_write_value(data) instead"
//              << ends;
//
//            Tango::Except::throw_exception(
//                    "PyDs_WrongPythonDataTypeForAttribute",
//                    o.str(),
//                    "set_write_value()");
//        }
//        else
//        {
//            if (!PySequence_Check(value.ptr()))
//            {
//                std::stringstream o;
//                o << "Wrong Python type for attribute " << att.get_name()
//                  << "of type " << Tango::CmdArgTypeName[type]
//                  << ". Expected a sequence" << ends;
//
//                Tango::Except::throw_exception(
//                        "PyDs_WrongPythonDataTypeForAttribute",
//                        o.str(),
//                        "set_write_value()");
//            }
//            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, __set_write_value_array,
//                                              self, value, x, 0);
//        }
    }

    inline void set_write_value(Tango::WAttribute& self,
                                py::object& value,
                                long x, long y)
    {
        long type = self.get_data_type();
        Tango::AttrDataFormat format = self.get_data_format();
//
//        if (format == Tango::SCALAR)
//        {
//            std::stringstream o;
//            o << "Cannot call set_write_value(data, dim_x, dim_y) "
//              << "on scalar attribute " << att.get_name()
//              << ". Use set_write_value(data) instead" << ends;
//
//            Tango::Except::throw_exception(
//                    (const char *)"PyDs_WrongPythonDataTypeForAttribute",
//                    o.str(),
//                    (const char *)"set_write_value()");
//        }
//        else
//        {
//            if (!PySequence_Check(value.ptr()))
//            {
//                std::stringstream o;
//                o << "Wrong Python type for attribute " << att.get_name()
//                  << "of type " << Tango::CmdArgTypeName[type]
//                  << ". Expected a sequence" << ends;
//
//                Tango::Except::throw_exception(
//                        (const char *)"PyDs_WrongPythonDataTypeForAttribute",
//                        o.str(),
//                        (const char *)"set_write_value()");
//            }
//            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, __set_write_value_array,
//                                              self, value, x, y);
//        }
    }
    //
    // PyTango 3 compatibility
    //

//    template<long tangoTypeConst>
//    void __get_write_value_pytango3(Tango::WAttribute &att, boost::python::list &seq)
//    {
//        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
//
//        const TangoScalarType *ptr;
//
//        long length = att.get_write_value_length();
//
//        att.get_write_value(ptr);
//
//        for (long l = 0; l < length; ++l)
//        {
//            seq.append(ptr[l]);
//        }
//    }
//
//    template<>
//    void __get_write_value_pytango3<Tango::DEV_STRING>(Tango::WAttribute &att,
//                                              boost::python::list &seq)
//    {
//        const Tango::ConstDevString *ptr = NULL;
//
//        att.get_write_value(ptr);
//
//    if (ptr == NULL) {
//        return;
//    }
//
//        long length = att.get_write_value_length();
//        for (long l = 0; l < length; ++l) {
//            seq.append(ptr[l]);
//        }
//    }
//
//    inline void get_write_value_pytango3(Tango::WAttribute &att,
//                                boost::python::list &value)
//    {
//        long type = att.get_data_type();
//        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, __get_write_value_pytango3, att, value);
//    }

    template<long tangoTypeConst>
    void __get_write_value_scalar(Tango::WAttribute &att, py::object& obj)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType v;
        att.get_write_value(v);
        obj = py::cast(v);
    }

    template<>
    void __get_write_value_scalar<Tango::DEV_STRING>(Tango::WAttribute &att, py::object& obj)
    {
        Tango::DevString v = nullptr;
        att.get_write_value(v);
        if (v == nullptr) {
            obj = py::none();
        } else {
            obj = py::cast(std::string(v));
        }
    }
//
//    template<long tangoTypeConst>
//    void __get_write_value_array_pytango3(Tango::WAttribute &att, boost::python::object* obj)
//    {
//        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
//
//        const TangoScalarType *buffer = NULL;
//        att.get_write_value(buffer);
//
//    if (buffer == NULL) {
//        *obj = boost::python::object();
//        return;
//    }
//
//        size_t length = att.get_write_value_length();
//
//        boost::python::list o;
//        for (size_t n = 0; n < length; ++n) {
//            o.append(buffer[n]);
//    }
//        *obj = o;
//    }
//
//    template<>
//    void __get_write_value_array_pytango3<Tango::DEV_STRING>(Tango::WAttribute &att, boost::python::object* obj)
//    {
//        const Tango::ConstDevString *ptr = NULL;
//
//    if (ptr == NULL) {
//        *obj = boost::python::object();
//        return;
//    }
//
//        long length = att.get_write_value_length();
//        att.get_write_value(ptr);
//        boost::python::list o;
//        for (long l = 0; l < length; ++l) {
//            o.append(ptr[l]);
//    }
//    }
//
//    template<long tangoTypeConst>
//    void __get_write_value_array_lists(Tango::WAttribute &att, boost::python::object* obj)
//    {
//        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
//
//        const TangoScalarType *buffer = NULL;
//        att.get_write_value(buffer);
//
//    if (buffer == NULL) {
//        *obj = boost::python::object();
//        return;
//    }
//
//        size_t dim_x = att.get_w_dim_x();
//        size_t dim_y = att.get_w_dim_y();
//
//        boost::python::list result;
//
//        if (att.get_data_format() == Tango::SPECTRUM) {
//            for (size_t x=0; x<dim_x; ++x) {
//                result.append(buffer[x]);
//            }
//        } else {
//            for (size_t y=0; y<dim_y; ++y) {
//                boost::python::list row;
//                for (size_t x=0; x<dim_x; ++x) {
//                    row.append(buffer[x + y*dim_x]);
//                }
//                result.append(row);
//            }
//        }
//        *obj = result;
//    }
//
//    template<>
//    void __get_write_value_array_lists<Tango::DEV_STRING>(Tango::WAttribute &att, boost::python::object* obj)
//    {
//        const Tango::ConstDevString* buffer= NULL;
//        att.get_write_value(buffer);
//
//    if (buffer == NULL) {
//        *obj = boost::python::object();
//        return;
//    }
//
//        size_t dim_x = att.get_w_dim_x();
//        size_t dim_y = att.get_w_dim_y();
//
//        boost::python::list result;
//
//        if (att.get_data_format() == Tango::SPECTRUM) {
//            for (size_t x=0; x<dim_x; ++x) {
//                result.append(buffer[x]);
//            }
//        } else {
//            for (size_t y=0; y<dim_y; ++y) {
//                boost::python::list row;
//                for (size_t x=0; x<dim_x; ++x) {
//                    row.append(buffer[x + y*dim_x]);
//                }
//                result.append(row);
//            }
//        }
//        *obj = result;
//    }

    inline py::object get_write_value(Tango::WAttribute& self)
    {
        long type = self.get_data_type();
        py::object value;

        Tango::AttrDataFormat fmt = self.get_data_format();
        const bool isScalar = fmt == Tango::SCALAR;

        if (isScalar)
        {
            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, __get_write_value_scalar, self, value);
        }
//        else
//        {
//              TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type,
//              __get_write_value_array_numpy, self, &value);
//              break;
//        }
        return value;
    }
};

void export_wattribute(py::module& m) {
    py::class_<Tango::WAttribute, Tango::Attribute>(m, "WAttribute")
        .def("get_min_value", [](Tango::WAttribute& self) -> py::object {
            return PyWAttribute::get_min_value(self);
        })
        .def("get_max_value", [](Tango::WAttribute& self) -> py::object {
            return PyWAttribute::get_max_value(self);
        })
        .def("set_min_value", [](Tango::WAttribute& self, py::object& min_value) -> void {
            PyWAttribute::set_min_value(self, min_value);
        })
        .def("set_max_value", [](Tango::WAttribute& self, py::object& max_value) -> void {
            PyWAttribute::set_max_value(self, max_value);
        })
        .def("is_min_value", [](Tango::WAttribute& self) -> bool {
            return self.is_min_value();
        })
        .def("is_max_value", [](Tango::WAttribute& self) -> bool {
            return self.is_max_value();
        })
        .def("get_write_value_length", [](Tango::WAttribute& self) -> long {
            return self.get_write_value_length();
        })
        .def("set_write_value", [](Tango::WAttribute& self, py::object& value) -> void {
            PyWAttribute::set_write_value(self, value);
        })
        .def("set_write_value", [](Tango::WAttribute& self, py::object& value, long x) -> void {
            PyWAttribute::set_write_value(self, value, x);
        })
        .def("set_write_value", [](Tango::WAttribute& self, py::object& value, long x, long y) -> void {
            PyWAttribute::set_write_value(self, value, x, y);
        })
        // new style get_write_value
        .def("get_write_value", [](Tango::WAttribute& self) {
            std::cerr << "gets to get_write_value in pybind" << std::endl;
//            py::object obj = PyWAttribute::get_write_value(self);
        })
  ;
}
