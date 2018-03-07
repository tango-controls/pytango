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
#include <pybind11/numpy.h>
#include <pyutils.h>
#include <device_attribute.h>

namespace py = pybind11;

//extern const char *non_valid_image;
//extern const char *non_valid_spectrum;
//
//// Why am I storing 'type' as a python attribute with object::attr
//// instead of as a property calling DeviceAttribute::get_type here?
//// Because after 'extract'ing, any call to get_type() will fail. Same
//// for "value" and "w_value". And for has_failed and is_empty...
static const char* value_attr_name = "value";
static const char* w_value_attr_name = "w_value";
//static const std::string type_attr_name = "type";
//static const std::string is_empty_attr_name = "is_empty";
//static const std::string has_failed_attr_name = "has_failed";


template<long tangoTypeConst>
struct python_tangocpp
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;

    static inline void to_cpp(const py::object & py_value, TangoScalarType & result)
    {
        result = py_value.cast<TangoScalarType>();
    }

    static inline py::object to_python(const TangoScalarType & value)
    {
        return py::cast(value);
    }
};

template<>
struct python_tangocpp<Tango::DEV_STRING>
{
    static const long tangoTypeConst = Tango::DEV_STRING;
    typedef TANGO_const2type(tangoTypeConst) TangoScalarType;

    static inline void to_cpp(const py::object & py_value, TangoScalarType & result)
    {
//        result = py_value.cast<char*>();
    }

//    static inline py::object to_python(const TangoScalarType & value)
//    {
//        return py::object(std::string(value));
//    }
};
//
//#ifndef DISABLE_PYTANGO_NUMPY
//#   include "device_attribute_numpy.hpp"
//#endif


#define EXTRACT_VALUE(self, value_ptr) \
try { \
    self >> value_ptr; \
} catch (Tango::DevFailed &e ) { \
    if (strcmp(e.errors[0].reason.in(),"API_EmptyDeviceAttribute") != 0) \
        throw; \
}


namespace PyDeviceAttribute {

    template<long tangoTypeConst> static inline void
    _update_value_as_bin(Tango::DeviceAttribute &self,
                         py::object py_value, bool read_only)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        typedef typename TANGO_const2arraytype(tangoTypeConst) TangoArrayType;

        // Extract the actual data from Tango::DeviceAttribute (self)
        TangoArrayType* value_ptr = nullptr;
        EXTRACT_VALUE(self, value_ptr)
//        unique_pointer<TangoArrayType> guard_value_ptr(value_ptr);

        py_value.attr(w_value_attr_name) = Py_None;

//        if (value_ptr == 0)
//        {
//            if (read_only)
//            {
//                py_value.attr(value_attr_name) =
//                    py::object(py::handle<>(_PyObject_New(&PyBytes_Type)));
//            }
//            else
//            {
//                py_value.attr(value_attr_name) =
//                    py::object(py::handle<>(_PyObject_New(&PyByteArray_Type)));
//            }
//            return;
//        }
//
        py::list py_list;
        TangoScalarType* buffer = value_ptr->get_buffer();
        for (int i=0; i<value_ptr->length(); i++) {
            py_list.append(py::cast(buffer[i]));
        }
//
//        const char *ch_ptr = reinterpret_cast<char *>(buffer);
//        Py_ssize_t nb_bytes = (Py_ssize_t)value_ptr->length() * sizeof(TangoScalarType);
//
//        PyObject* data_ptr = nullptr;
//        if (read_only)
//        {
//            data_ptr = PyBytes_FromStringAndSize(ch_ptr, nb_bytes);
//        }
//        else
//        {
//            data_ptr = PyByteArray_FromStringAndSize(ch_ptr, nb_bytes);
//        }
        py_value.attr(value_attr_name) = py_list;
    }

    template<> inline void
    _update_value_as_bin<Tango::DEV_ENCODED>(Tango::DeviceAttribute &self,
                                             py::object py_value,
                                             bool read_only)
    {
        char* r_encoded_format;
        unsigned char* data_ptr;
        unsigned int data_size;
        py::list r_encoded_data;
        self.extract(r_encoded_format, data_ptr, data_size);
        for (auto i=0; i<(int)data_size; i++) {
            r_encoded_data.append(data_ptr[i]);
        }
        py_value.attr(value_attr_name) = py::make_tuple(py::str(r_encoded_format),
                                                                r_encoded_data);
        if (self.get_written_dim_x() > 0)
        {
//            bool is_write_type = self.get_written_dim_x() && (value_ptr->length() < 2);
//            if (is_write_type)
//            {
                py_value.attr(w_value_attr_name) = py::make_tuple(r_encoded_format, r_encoded_data);
//            }
//            else
//            {
//                py::list w_encoded_data;
//                for(int i=0; i<dim_x; i++) {
//                    r_encoded_data[i] = py::cast(data_ptr[i])
//                }
//                Tango::DevEncoded& w_buffer = buffer[1];
//                py::str w_encoded_format(w_buffer.encoded_format);
//
//                Tango::DevVarCharArray& w_encoded_data_array = w_buffer.encoded_data;
//                char* w_ch_ptr = (char*) w_encoded_data_array.get_buffer();
//                PyObject* w_encoded_data_ptr = nullptr;
//                    PyByteArray_FromStringAndSize(w_ch_ptr, w_encoded_data_array.length());
//                Py_ssize_t w_size = w_encoded_data_array.length();
//                if (read_only)
//                {
//                    w_encoded_data_ptr = PyBytes_FromStringAndSize(w_ch_ptr, w_size);
//                }
//                else
//                {
//                    w_encoded_data_ptr = PyByteArray_FromStringAndSize(w_ch_ptr, w_size);
//                }
//                py::object w_encoded_data = py::object(py::handle<>(w_encoded_data_ptr));
//
//                py_value.attr(value_attr_name) =
//                    py::make_tuple(r_encoded_format, w_encoded_data);
//            }
        }
        else
        {
            py_value.attr(w_value_attr_name) = Py_None;
        }
    }

//    template<> inline void
//    _update_value_as_bin<Tango::DEV_PIPE_BLOB>(Tango::DeviceAttribute &self,
//					       py::object py_value,
//					       bool read_only)
//    {
//	assert(false);
//    }

//    template<long tangoTypeConst> static inline void
//    _update_value_as_string(Tango::DeviceAttribute &self,
//                            py::object py_value)
//    {
//        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
//        typedef typename TANGO_const2arraytype(tangoTypeConst) TangoArrayType;
//
//        // Extract the actual data from Tango::DeviceAttribute (self)
//        TangoArrayType* value_ptr = 0;
//        EXTRACT_VALUE(self, value_ptr)
//        unique_pointer<TangoArrayType> guard_value_ptr(value_ptr);
//
//        if (value_ptr == 0)
//        {
//            py_value.attr(value_attr_name) = py::str();
//            py_value.attr(w_value_attr_name) = py::object();
//            return;
//        }
//
//        TangoScalarType* buffer = value_ptr->get_buffer();
//
//        const char* ch_ptr = reinterpret_cast<char *>(buffer);
//        size_t nb_bytes = value_ptr->length() * sizeof(TangoScalarType);
//
//        py_value.attr(value_attr_name) = py::str(ch_ptr, (size_t)nb_bytes);
//        py_value.attr(w_value_attr_name) = py::object();
//    }

//    template<> inline void
//    _update_value_as_string<Tango::DEV_ENCODED>(Tango::DeviceAttribute &self,
//                                                py::object py_value)
//    {
//        Tango::DevVarEncodedArray* value_ptr;
//        EXTRACT_VALUE(self, value_ptr)
//        unique_pointer<Tango::DevVarEncodedArray> guard(value_ptr);
//
//        Tango::DevEncoded* buffer = value_ptr->get_buffer();
//
//        Tango::DevEncoded& r_buffer = buffer[0];
//        py::str r_encoded_format(r_buffer.encoded_format);
//
//        Tango::DevVarCharArray& r_encoded_data_array = r_buffer.encoded_data;
//        char* r_ch_ptr = (char*)r_encoded_data_array.get_buffer();
//        py::str r_encoded_data(r_ch_ptr, r_encoded_data_array.length());
//
//        py_value.attr(value_attr_name) =
//            py::make_tuple(r_encoded_format, r_encoded_data);
//
//        if (self.get_written_dim_x() > 0)
//        {
//            bool is_write_type = self.get_written_dim_x() && (value_ptr->length() < 2);
//            if (is_write_type)
//            {
//                py::object w_encoded_format(r_encoded_format);
//                py::object w_encoded_data(r_encoded_data);
//                py_value.attr(w_value_attr_name) =
//                    py::make_tuple(w_encoded_format, w_encoded_data);
//            }
//            else
//            {
//                Tango::DevEncoded& w_buffer = buffer[1];
//                py::str w_encoded_format(w_buffer.encoded_format);
//
//                Tango::DevVarCharArray& w_encoded_data_array = w_buffer.encoded_data;
//                char* w_ch_ptr = (char*)w_encoded_data_array.get_buffer();
//                py::str w_encoded_data(w_ch_ptr, w_encoded_data_array.length());
//                py_value.attr(w_value_attr_name) =
//                    py::make_tuple(w_encoded_format, w_encoded_data);
//            }
//        }
//        else
//        {
//            py_value.attr(w_value_attr_name) = py::object();
//        }
//    }
//
//    template<> inline void
//    _update_value_as_string<Tango::DEV_PIPE_BLOB>(Tango::DeviceAttribute &self,
//						  py::object py_value)
//    {
//	assert(false);
//    }
//
    template<long tangoTypeConst> static inline void
    _update_scalar_values(Tango::DeviceAttribute &self, py::object py_value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;

        if (self.get_written_dim_x() > 0)
        {
            std::vector<TangoScalarType> val;
            self.extract_read(val);
            // In the following lines, the cast is absolutely necessary because
            // vector<TangoScalarType> may not be a vector<TangoScalarType> at
            // compile time. For example, for vector<DevBoolean>, the compiler
            // may create a std::_Bit_reference type.
            py_value.attr(value_attr_name) = py::cast((TangoScalarType)val[0]);
            self.extract_set(val);
            py_value.attr(w_value_attr_name) = py::cast((TangoScalarType)val[0]);
        }
        else
        {
            TangoScalarType rvalue;
            EXTRACT_VALUE(self, rvalue)
            py_value.attr(value_attr_name) = rvalue;
            py_value.attr(w_value_attr_name) = Py_None;
        }
    }

    template<> inline void
    _update_scalar_values<Tango::DEV_ENCODED>(Tango::DeviceAttribute &self,
                                              py::object py_value)
    {
//        _update_value_as_string<Tango::DEV_ENCODED>(self, py_value);
    }

    template<> inline void
    _update_scalar_values<Tango::DEV_STRING>(Tango::DeviceAttribute &self,
                                             py::object py_value)
    {
        if (self.get_written_dim_x() > 0)
        {
            std::vector<std::string> r_val, w_val;
            self.extract_read(r_val);
            py_value.attr(value_attr_name) = py::cast(r_val[0]);
            self.extract_set(w_val);
            py_value.attr(w_value_attr_name) = py::cast(w_val[0]);
        }
        else
        {
            std::string rvalue;
            EXTRACT_VALUE(self, rvalue)
            py_value.attr(value_attr_name) = rvalue;
            py_value.attr(w_value_attr_name) = Py_None;
        }
    }

    template<> inline void
    _update_scalar_values<Tango::DEV_PIPE_BLOB>(Tango::DeviceAttribute &self,
						py::object py_value)
    {
        assert(false);
    }
//
//    template<long tangoTypeConst> static inline void
//    _update_array_values_as_lists(Tango::DeviceAttribute &self, bool isImage,
//                                  py::object py_value)
//    {
//        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
//        typedef typename TANGO_const2arraytype(tangoTypeConst) TangoArrayType;
//
//        // Extract the actual data from Tango::DeviceAttribute (self)
//        TangoArrayType* value_ptr = 0;
//        EXTRACT_VALUE(self, value_ptr)
//        unique_pointer<TangoArrayType> guard_value_ptr(value_ptr);
//
//        if (value_ptr == 0) {
//            // Empty device attribute
//            py_value.attr(value_attr_name) = py::list();
//            py_value.attr(w_value_attr_name) = object();
//            return;
//        }
//
//        TangoScalarType* buffer = value_ptr->get_buffer();
//        int total_length = value_ptr->length();
//
//        // Determine if the attribute is AttrWriteType.WRITE
//        int read_size =0, write_size = 0;
//        if (isImage) {
//            read_size = self.get_dim_x() * self.get_dim_y();
//            write_size = self.get_written_dim_x() * self.get_written_dim_y();
//        } else {
//            read_size = self.get_dim_x();
//            write_size = self.get_written_dim_x();
//        }
//        bool is_write_type = (read_size + write_size) > total_length;
//
//        // Convert to a list of lists
//        long offset = 0;
//        for(int it=1; it>=0; --it) { // 2 iterations: read part/write part
//            if ((!it) && is_write_type) {
//                py_value.attr(w_value_attr_name) = py_value.attr(value_attr_name);
//                continue;
//            }
//            py::list result;
//
//            if (isImage) {
//                const int dim_x = it? self.get_dim_x() : self.get_written_dim_x();
//                const int dim_y = it? self.get_dim_y() : self.get_written_dim_y();
//
//                for (int y=0; y < dim_y; ++y) {
//                    py::list row;
//                    for (int x=0; x < dim_x; ++x)
//                        row.append(python_tangocpp<tangoTypeConst>::to_python(buffer[offset + x + y*dim_x]));
//                    result.append(row);
//                }
//                offset += dim_x*dim_y;
//            } else {
//                const int dim_x = it? self.get_dim_x() : self.get_written_dim_x();
//                for (int x=0; x < dim_x; ++x)
//                    result.append(python_tangocpp<tangoTypeConst>::to_python(buffer[offset + x]));
//                offset += dim_x;
//            }
//            py_value.attr(it? value_attr_name : w_value_attr_name) = result;
//        }
//    }

//    template<> inline void
//    _update_array_values_as_lists<Tango::DEV_PIPE_BLOB>(Tango::DeviceAttribute &self,
//                                                        bool isImage, py::object py_value)
//    {
//        assert(false);
//    }

//    template<long tangoTypeConst> static void
//    _update_array_values_as_tuples(Tango::DeviceAttribute &self, bool isImage,
//                                   py::object py_value)
//    {
//        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
//        typedef typename TANGO_const2arraytype(tangoTypeConst) TangoArrayType;
//
//        // Extract the actual data from Tango::DeviceAttribute (self)
//        TangoArrayType* value_ptr = 0;
//        EXTRACT_VALUE(self, value_ptr)
//        unique_pointer<TangoArrayType> guard_value_ptr(value_ptr);
//
//        if (value_ptr == 0) {
//            // Empty device attribute
//            py_value.attr(value_attr_name) = py::tuple();
//            py_value.attr(w_value_attr_name) = object();
//            return;
//        }
//
//        TangoScalarType* buffer = value_ptr->get_buffer();
//        int total_length = value_ptr->length();
//
//        // Determine if the attribute is AttrWriteType.WRITE
//        int read_size =0, write_size = 0;
//        if (isImage) {
//            read_size = self.get_dim_x() * self.get_dim_y();
//            write_size = self.get_written_dim_x() * self.get_written_dim_y();
//        } else {
//            read_size = self.get_dim_x();
//            write_size = self.get_written_dim_x();
//        }
//        bool is_write_type = (read_size + write_size) > total_length;
//
//        // Convert to a tuple of tuples
//        long offset = 0;
//        for(int it=1; it>=0; --it) { // 2 iterations: read part/write part
//            if ((!it) && is_write_type) {
//                py_value.attr(w_value_attr_name) = py_value.attr(value_attr_name);
//                continue;
//            }
//
//            object result_guard;
//            if (isImage) {
//                const int dim_x = it? self.get_dim_x() : self.get_written_dim_x();
//                const int dim_y = it? self.get_dim_y() : self.get_written_dim_y();
//
//                PyObject * result = PyTuple_New(dim_y);
//                if (!result)
//                    py::throw_error_already_set();
//                result_guard = object(handle<>(result));
//
//                for (int y=0; y < dim_y; ++y) {
//                    PyObject * row = PyTuple_New(dim_x);
//                    if (!row)
//                        py::throw_error_already_set();
//                    object row_guard = object(handle<>(row));
//                    for (int x=0; x < dim_x; ++x) {
//                        object el = python_tangocpp<tangoTypeConst>::to_python(buffer[offset + x + y*dim_x]);
//                        PyTuple_SetItem(row, x, el.ptr());
//                        incref(el.ptr());
//                    }
//                    PyTuple_SetItem(result, y, row);
//                    incref(row);
//                }
//                offset += dim_x*dim_y;
//            } else {
//                const int dim_x = it? self.get_dim_x() : self.get_written_dim_x();
//
//                PyObject * result = PyTuple_New(dim_x);
//                if (!result)
//                    py::throw_error_already_set();
//                result_guard = object(handle<>(result));
//
//                for (int x=0; x < dim_x; ++x) {
//                    object el = python_tangocpp<tangoTypeConst>::to_python(buffer[offset +x]);
//                    PyTuple_SetItem(result, x, el.ptr());
//                    incref(el.ptr());
//                }
//                offset += dim_x;
//            }
//            py_value.attr(it? value_attr_name : w_value_attr_name) = result_guard;
//        }
//    }

//    template<> inline void
//    _update_array_values_as_tuples<Tango::DEV_PIPE_BLOB>(Tango::DeviceAttribute &self,
//                                                         bool isImage, py::object py_value)
//    {
//        assert(false);
//    }

    void
    update_values(Tango::DeviceAttribute &self, py::object& py_value,
                  PyTango::ExtractAs extract_as)
    {
        // We do not want is_empty to launch an exception!!
        self.reset_exceptions(Tango::DeviceAttribute::isempty_flag);

        const int data_type = self.get_type();
        const bool is_empty = data_type < 0;
        const bool has_failed = self.has_failed();
        Tango::AttrDataFormat data_format = self.get_data_format();

        py_value.attr("is_empty") = is_empty;
        py_value.attr("has_failed") = has_failed;
        py_value.attr("type") = static_cast<Tango::CmdArgType>(data_type);

        if (has_failed || is_empty) {
            // In neither of these cases is 'data_type' valid so we cannot extract
            py_value.attr("value") = Py_None;
            py_value.attr("w_value") = Py_None;
            return;
        }

        bool is_image = false;
        switch (data_format) {
            case Tango::SCALAR:
                if (data_type == Tango::DEV_ENCODED)
                {
                    switch (extract_as)
                    {
                        default:
//                        case PyTango::ExtractAsNumpy:
//                        case PyTango::ExtractAsTuple:
//                        case PyTango::ExtractAsList:
//                            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
//                                _update_scalar_values, self, py_value);
//                            break;
//                        case PyTango::ExtractAsBytes:
//                            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
//                                _update_value_as_bin, self, py_value, true);
//                            break;
                        case PyTango::ExtractAsByteArray:
                            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
                                _update_value_as_bin, self, py_value, false);
                            break;
//                        case PyTango::ExtractAsString:
//                            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
//                                _update_value_as_string, self, py_value);
//                            break;
//                        case PyTango::ExtractAsNothing:
//                            break;
                    }
                }
                else
                {
                    if (extract_as != PyTango::ExtractAsNothing)
                    {
                        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
                            _update_scalar_values, self, py_value);
                    }
                }
                break;
            case Tango::IMAGE:
                is_image = true;
            case Tango::SPECTRUM:
                switch (extract_as)
                {
                    default:
//                    case PyTango::ExtractAsNumpy:
//                        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
//                            _update_array_values, self, is_image, py_value);
//                        break;
//                    case PyTango::ExtractAsTuple:
//                        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
//                            _update_array_values_as_tuples, self, is_image, py_value);
//                        break;
//                    case PyTango::ExtractAsList:
//                        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
//                            _update_array_values_as_lists, self, is_image, py_value);
//                        break;
//                    case PyTango::ExtractAsBytes:
//                        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
//                            _update_value_as_bin, self, py_value, true);
//                        break;
                    case PyTango::ExtractAsByteArray:
                        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
                            _update_value_as_bin, self, py_value, false);
                        break;
//                    case PyTango::ExtractAsString:
//                        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(data_type,
//                            _update_value_as_string, self, py_value);
//                        break;
                    case PyTango::ExtractAsNothing:
                        break;
                }
                break;
            case Tango::FMT_UNKNOWN:
                default:
                raise_(PyExc_ValueError, "Can't extract data because: self.get_data_format()=FMT_UNKNOWN");
                assert(false);
        }
    }

    template<long tangoTypeConst> static inline void
    _fill_list_attribute(Tango::DeviceAttribute & dev_attr, const bool isImage,
                         py::object& py_value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        typedef typename TANGO_const2arraytype(tangoTypeConst) TangoArrayType;
        unsigned long dim_x, dim_y, nelems;

        // -- Check the dimensions
        if (isImage) {
            py::list py_list = py_value;
            py::list py_sub = py_list[0];
            dim_y = len(py_list);
            dim_x = len(py_sub);
            nelems = dim_x * dim_y;
        } else {
            dim_x = py::len(py_value);
            dim_y = 0;
            nelems = dim_x;
        }

        // -- Allocate memory
        std::unique_ptr<TangoArrayType> value;
        TangoScalarType* buffer = TangoArrayType::allocbuf(nelems);
        try {
            value.reset(new TangoArrayType(nelems, nelems, buffer, true));
        } catch(...) {
            TangoArrayType::freebuf(buffer);
            throw;
        }
        // -- Copy the sequence to the newly created buffer
        if (isImage) {
            py::list py_list = py_value;
            for(int y=0; y<dim_y; y++) {
                py::list py_sub = py_list[y];
                for(int x=0; x<dim_x; x++) {
                    python_tangocpp<tangoTypeConst>::to_cpp(py_sub[x], buffer[x+y*dim_x]);
                }
            }
        } else {
            py::list py_list = py_value;
            for(int i=0; i<dim_x; i++) {
                python_tangocpp<tangoTypeConst>::to_cpp(py_list[i], buffer[i]);
            }
        }
        // -- Insert it into the dev_attr
        dev_attr.insert(value.get(), dim_x, dim_y);
        // -- Final cleaning
        value.release(); // Do not delete value, it is handled by dev_attr now!
    }

    template<> inline void
    _fill_list_attribute<Tango::DEV_ENCODED>(Tango::DeviceAttribute & dev_attr,
                                             const bool isImage,
                                             py::object& py_value)
    {
        /// @todo really? This is really not gonna happen?
        // Unsupported
        assert(false);
    }

    void reset_values(Tango::DeviceAttribute & self, int data_type,
                 Tango::AttrDataFormat data_format, py::object py_value)
    {
        bool isImage = false;
        switch(data_format)
        {
            case Tango::SCALAR:
                TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID( data_type, _fill_scalar_attribute, self, py_value );
                break;
            case Tango::IMAGE:
                isImage = true;
            case Tango::SPECTRUM:
//                // Why on earth? Why do we define _fill_numpy_attribute instead
//                // of just using _fill_list_attribute, if the latter accepts
//                // anything with operators "[]" and "len" defined?
//                // Well it seems that PyArray_GETITEM does something diferent
//                // and I get a value transformed into a python basic type,
//                // while py_value[y][x] gives me a numpy type (ej: numpy.bool_).
//                // Then the conversions between numpy types to c++ are not
//                // defined by boost while the conversions between python
//                // standard types and C++ are.
//#               ifdef DISABLE_PYTANGO_NUMPY
//                {
                    TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID( data_type, _fill_list_attribute, self, isImage, py_value );
//                }
//#               else
//                {
//                    if (PyArray_Check(py_value.ptr()))
//                        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID( data_type, _fill_numpy_attribute, self, isImage, py_value );
//                    else
//                        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID( data_type, _fill_list_attribute, self, isImage, py_value );
//                }
//#               endif
                break;
            default:
                raise_(PyExc_TypeError, "unsupported data_format.");
        }
    }

    void reset(Tango::DeviceAttribute & self, const Tango::AttributeInfo &attr_info, py::object py_value) {
        self.set_name(attr_info.name.c_str());
        reset_values(self, attr_info.data_type, attr_info.data_format, py_value);
    }

    void reset(Tango::DeviceAttribute & self, const std::string &attr_name, Tango::DeviceProxy &dev_proxy, py::object py_value) {
        self.set_name(attr_name.c_str());
        Tango::AttributeInfoEx attr_info;
        {
            AutoPythonAllowThreads guard;
            try {
                attr_info = dev_proxy.get_attribute_config(attr_name);
            } catch (...) {
            }
        }
        reset_values(self, attr_info.data_type, attr_info.data_format, py_value);
    }
};

void export_device_attribute(py::module &m)
{
    py::class_<Tango::DeviceAttribute>(m, "DeviceAttribute", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<const Tango::DeviceAttribute &>())
        .def_readwrite("name", &Tango::DeviceAttribute::name)
        .def_readwrite("quality", &Tango::DeviceAttribute::quality)
        .def_readwrite("time", &Tango::DeviceAttribute::time)
        .def_property_readonly("dim_x", &Tango::DeviceAttribute::get_dim_x)
        .def_property_readonly("dim_y", &Tango::DeviceAttribute::get_dim_y)
        .def_property_readonly("w_dim_x", &Tango::DeviceAttribute::get_written_dim_x)
        .def_property_readonly("w_dim_y", &Tango::DeviceAttribute::get_written_dim_y)
        .def_property_readonly("r_dimension", &Tango::DeviceAttribute::get_r_dimension)
        .def_property_readonly("w_dimension", &Tango::DeviceAttribute::get_w_dimension)
        .def_property_readonly("nb_read", &Tango::DeviceAttribute::get_nb_read)
        .def_property_readonly("nb_written", &Tango::DeviceAttribute::get_nb_written)
        .def_property_readonly("data_format", &Tango::DeviceAttribute::get_data_format)
        .def("get_date", &Tango::DeviceAttribute::get_date,
            py::return_value_policy::reference)
        .def("get_err_stack", &Tango::DeviceAttribute::get_err_stack,
            py::return_value_policy::copy)
        .def("set_w_dim_x", &Tango::DeviceAttribute::set_w_dim_x)
        .def("set_w_dim_y", &Tango::DeviceAttribute::set_w_dim_y)
        ;
        py::enum_<Tango::DeviceAttribute::except_flags>(m, "except_flags")
            .value("isempty_flag", Tango::DeviceAttribute::isempty_flag)
            .value("wrongtype_flag", Tango::DeviceAttribute::wrongtype_flag)
            .value("failed_flag", Tango::DeviceAttribute::failed_flag)
            .value("numFlags", Tango::DeviceAttribute::numFlags)
        ;
}
