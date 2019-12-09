/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <server/command.h>
#include <server/device_impl.h>
#include <pytgutils.h>
#include <pyutils.h>
#include <tgutils.h>
#include <exception.h>
#include <memory>

namespace py = pybind11;

bool PyCmd::is_allowed(Tango::DeviceImpl *dev, const CORBA::Any &any) {
    if (py_allowed_defined) {
        DeviceImplWrap *dev_ptr = (DeviceImplWrap*) dev;
        AutoPythonGILEnsure __py_lock;
        bool returned_value = true;
        try {
            returned_value = dev_ptr->py_self.attr(py_allowed_name.c_str())().cast<bool>();
        } catch (py::error_already_set &eas) {
            handle_python_exception(eas);
        }
        return returned_value;
    }
    return true;
}

void allocate_any(CORBA::Any *&any_ptr) {
    try {
        any_ptr = new CORBA::Any();
    } catch (bad_alloc) {
        Tango::Except::throw_exception("API_MemoryAllocation",
                "Can't allocate memory in server", "PyCmd::allocate_any()");
    }
}

void throw_bad_type(const char *type) {
    std::stringstream o;
    o << "Incompatible command argument type, expected type is : Tango::"
            << type << std::ends;
    Tango::Except::throw_exception("API_IncompatibleCmdArgumentType",
            o.str(), "PyCmd::extract()");
}

template<long tangoTypeConst>
void insert_scalar(CORBA::Any& any, py::object& py_value)
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    TangoScalarType value =  py_value.cast<TangoScalarType>();
    any <<= value;
}

template<>
void insert_scalar<Tango::DEV_BOOLEAN>(CORBA::Any& any, py::object& py_value)
{
    Tango::DevBoolean value = py_value.cast<Tango::DevBoolean>();
    CORBA::Any::from_boolean any_value(value);
    any <<= any_value;
}

template<>
void insert_scalar<Tango::DEV_STRING>(CORBA::Any& any, py::object& py_value)
{
    std::string value = py_value.cast<std::string>();
    any <<= CORBA::string_dup(value.c_str());
}

template<>
void insert_scalar<Tango::DEV_ENCODED>(CORBA::Any& any, py::object& py_value)
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
    any <<= value;
}

template<>
void insert_scalar<Tango::DEV_VOID>(CORBA::Any& any, py::object& py_value)
{}

template<>
void insert_scalar<Tango::DEV_PIPE_BLOB>(CORBA::Any& any, py::object& py_value)
{
    assert(false);
}

template<long tangoArrayTypeConst>
void insert_array(CORBA::Any &any, py::object& py_value) {
    typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
    typedef typename TANGO_const2scalartype(tangoArrayTypeConst) TangoScalarType;

    py::list py_list = py_value;
    long len = py::len(py_list);
    TangoScalarType* data_buffer = new TangoScalarType[len];
    TangoScalarType value;
    for (int i=0; i<len; i++) {
        value = py_list[i].cast<TangoScalarType>();
        data_buffer[i] = value;
    }
    TangoArrayType array(len, len, data_buffer, true);
    any <<= array;
}

template<>
void insert_array<Tango::DEVVAR_STRINGARRAY>(CORBA::Any& any, py::object& py_value)
{
    py::list py_list = py_value;
    long len = py::len(py_list);
    Tango::DevString* data_buffer = new Tango::DevString[len];
    for (int i=0; i<len; i++) {
        std::string value = py_list[i].cast<std::string>();
        data_buffer[i] = strdup(value.c_str());
    }
    Tango::DevVarStringArray array(len, len, data_buffer, false);
    any <<= array;
}

template<>
void insert_array<Tango::DEVVAR_LONGSTRINGARRAY>(CORBA::Any& any, py::object& py_value)
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
    any <<= array;
}

template<>
void insert_array<Tango::DEVVAR_DOUBLESTRINGARRAY>(CORBA::Any& any, py::object& py_value)
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
    any <<= array;
}

template<long tangoTypeConst>
void extract_scalar(const CORBA::Any &any, py::object& o) {
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    TangoScalarType data;

    if ((any >>= data) == false) {
        throw_bad_type(Tango::CmdArgTypeName[tangoTypeConst]);
    }
    o = py::cast(data);
}

template<>
void extract_scalar<Tango::DEV_STRING>(const CORBA::Any &any, py::object& o) {
    Tango::ConstDevString data;

    if ((any >>= data) == false) {
        throw_bad_type(Tango::CmdArgTypeName[Tango::DEV_STRING]);
    }
    o = py::cast(data);
}

template<>
void extract_scalar<Tango::DEV_VOID>(const CORBA::Any &any, py::object& o)
{}

template<>
void extract_scalar<Tango::DEV_PIPE_BLOB>(const CORBA::Any &any, py::object& o) {
    assert(false);
}

template<>
void extract_scalar<Tango::DEV_ENCODED>(const CORBA::Any &any, py::object& obj) {
    Tango::DevEncoded* val;
    any >>= val;
    py::str encoded_format = strdup(val->encoded_format);
    py::list encoded_data;
    int len = val->encoded_data.length();
    for (auto i=0; i<len; i++) {
        encoded_data.append(val->encoded_data[i]);
    }
    obj = py::make_tuple(encoded_format, encoded_data);
}

template<long tangoArrayTypeConst>
void extract_array(const CORBA::Any &any, py::object& py_result)
{
    typedef typename TANGO_const2scalartype(tangoArrayTypeConst) TangoScalarType;
    typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

    TangoArrayType *tmp_arr;
    if ((any >>= tmp_arr) == false)
        throw_bad_type(Tango::CmdArgTypeName[tangoArrayTypeConst]);

    // For numpy we need a 'guard' object that handles the memory used
    // by the numpy object (releases it).
    // But I cannot manage memory inside our 'any' object, because it is
    // const and handles it's memory itself. So I need a copy before
    // creating the object.
    TangoArrayType* copy_ptr = new TangoArrayType(*tmp_arr);

    // numpy.ndarray() does not own it's memory, so we need to manage it.
    // We can assign a 'base' object that will be informed (decref'd) when
    // the last copy of numpy.ndarray() disappears.
    py::capsule free_when_done(reinterpret_cast<void*>(copy_ptr), [](void* f) {
         TangoScalarType *ptr = reinterpret_cast<TangoScalarType *>(f);
         delete[] ptr;
    });
    py::array array;
    int dims[1];
    dims[0] = copy_ptr->length();
    array = py::array_t<TangoScalarType>(dims,
            reinterpret_cast<TangoScalarType*>(copy_ptr->get_buffer()), free_when_done);

    py_result = py::object(array);
}

template<>
void extract_array<Tango::DEVVAR_STRINGARRAY>(const CORBA::Any &any, py::object& py_result)
{
    Tango::DevVarStringArray *array;
    if ((any >>= array) == false)
        throw_bad_type(Tango::CmdArgTypeName[Tango::DEVVAR_STRINGARRAY]);

    py::list result;
    int len = array->length();
    for (auto i=0; i<len; i++) {
        result.append(py::str((*array)[i]));
    }

    py_result = result;
}

template<>
void extract_array<Tango::DEVVAR_LONGSTRINGARRAY>(const CORBA::Any &any, py::object& py_result)
{
    py::list long_data;
    py::list string_data;
    const Tango::DevVarLongStringArray *array = NULL;
    if ((any >>= array) == false)
        throw_bad_type(Tango::CmdArgTypeName[Tango::DEVVAR_LONGSTRINGARRAY]);
    int llen = array->lvalue.length();
    for (auto i=0; i<llen; i++) {
        long_data.append(py::cast((array->lvalue)[i]));
    }
    int slen = array->svalue.length();
    for (auto i=0; i<slen; i++) {
        string_data.append(py::str((array->svalue)[i]));
    }
    py_result = py::make_tuple(long_data, string_data);
}

template<>
void extract_array<Tango::DEVVAR_DOUBLESTRINGARRAY>(const CORBA::Any &any, py::object& py_result)
{
    py::list double_data;
    py::list string_data;
    const Tango::DevVarDoubleStringArray *array = NULL;
    any >>= array;
    int dlen = array->dvalue.length();
    for (auto i=0; i<dlen; i++) {
        double_data.append(py::cast((array->dvalue)[i]));
    }
    int slen = array->svalue.length();
    for (auto i=0; i<slen; i++) {
        string_data.append(py::str((array->svalue)[i]));
    }
    py_result = py::make_tuple(double_data, string_data);
}

template<>
void extract_array<Tango::DEV_PIPE_BLOB>(const CORBA::Any &any,
                     py::object& py_result)
{
    assert(false);
}

CORBA::Any *PyCmd::execute(Tango::DeviceImpl *dev, const CORBA::Any &param_any)
{
    DeviceImplWrap *dev_ptr = (DeviceImplWrap*)dev;

    AutoPythonGILEnsure __py_lock;
    try
    {
        // This call extracts the CORBA any into a python object.
        // So, the result is that param_py = param_any.
        // It is done with some template magic.
        py::object param_py;
        TANGO_DO_ON_DEVICE_DATA_TYPE_ID(in_type,
            extract_scalar<tangoTypeConst>(param_any, param_py);
        ,
            extract_array<tangoTypeConst>(param_any, param_py);
        );

        // Execute the python call for the command
        py::object ret_py_obj;
        if (in_type == Tango::DEV_VOID)
        {
            ret_py_obj = (dev_ptr->py_self).attr(name.c_str())();
        }
        else
        {
            ret_py_obj = dev_ptr->py_self.attr(name.c_str())(param_py);
        }
        CORBA::Any *ret_any;
        allocate_any(ret_any);
        std::unique_ptr<CORBA::Any> ret_any_guard(ret_any);

        // It does: ret_any = ret_py_obj
        TANGO_DO_ON_DEVICE_DATA_TYPE_ID(out_type,
            insert_scalar<tangoTypeConst>(*ret_any, ret_py_obj);
        ,
            insert_array<tangoTypeConst>(*ret_any, ret_py_obj);
        );
        return ret_any_guard.release();
    } catch(py::error_already_set &eas) {
        handle_python_exception(eas);
        return 0; // Should not happen, handle_python_exception rethrows in
                  // a Tango friendly manner
    }
}
