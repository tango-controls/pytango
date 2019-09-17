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
        Device_5ImplWrap *dev_ptr = (Device_5ImplWrap*) dev;
        AutoPythonGILEnsure __py_lock;
        bool returned_value = true;
        std::cout << "does it do this?" << std::endl;
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
void insert_scalar(CORBA::Any& any, py::object& o)
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    TangoScalarType value =  o.cast<TangoScalarType>();
    py::print(value);
    py::print(tangoTypeConst);
    any <<= value;
}

void insert_scalar_boolean(CORBA::Any& any, py::object& o)
{
    Tango::DevBoolean value = o.cast<Tango::DevBoolean>();
    CORBA::Any::from_boolean any_value(value);
    any <<= any_value;
}

void insert_scalar_string(CORBA::Any& any, py::object& o)
{
    std::string value = o.cast<std::string>();
    any <<= CORBA::string_dup(value.c_str());
}

void insert_scalar_encoded(CORBA::Any& any, py::object& obj)
{
    py::tuple tup = obj;
    py::object p0 = tup[0];
    py::array_t<unsigned char> p1(tup[1]);
    std::string encoded_format = p0.cast<std::string>();
    Py_buffer view;
    if (PyObject_GetBuffer(p1.ptr(), &view, PyBUF_FULL_RO) < 0)
    {
        throw_bad_type(Tango::CmdArgTypeName[Tango::DEV_ENCODED]);
    }
    CORBA::ULong nb = static_cast<CORBA::ULong>(view.len);
    Tango::DevVarCharArray arr(nb, nb, (CORBA::Octet*)view.buf, false);
    Tango::DevEncoded *data = new Tango::DevEncoded;
    data->encoded_format = CORBA::string_dup(encoded_format.c_str());
    data->encoded_data = arr;
    any <<= data;
    PyBuffer_Release(&view);
}

template<long tangoArrayTypeConst>
void insert_array(CORBA::Any &any, py::object& obj) {
    typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
    typedef typename TANGO_const2scalartype(tangoArrayTypeConst) TangoScalarType;

    // Destruction will be handled by CORBA, not by Tango.
    // TangoArrayType* data = fast_convert2array<tangoArrayTypeConst>(o);
    // By giving a pointer to <<= we are giving ownership of the data
    // buffer to CORBA
    // any <<= data;
    py::array_t<TangoScalarType> p1(obj);
    Py_buffer view;
    if (PyObject_GetBuffer(p1.ptr(), &view, PyBUF_FULL_RO) < 0)
    {
        throw_bad_type(Tango::CmdArgTypeName[tangoArrayTypeConst]);
    }
    CORBA::ULong nb = static_cast<CORBA::ULong>(view.len) / sizeof(TangoScalarType);
    TangoArrayType arr(nb, nb, (TangoScalarType*)view.buf, false);
    any <<= arr;
    PyBuffer_Release(&view);
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
void extract_scalar<Tango::DEV_ENCODED>(const CORBA::Any &any, py::object& o) {
    Tango::DevEncoded* data;

    if ((any >>= data) == false) {
        throw_bad_type(Tango::CmdArgTypeName[Tango::DEV_ENCODED]);
    }
    py::str encoded_format(data[0].encoded_format);
    py::str encoded_data((const char*)data[0].encoded_data.get_buffer(),
                           data[0].encoded_data.length());

    o = py::make_tuple(encoded_format, encoded_data);
}

//#ifndef DISABLE_PYTANGO_NUMPY
///// This callback is run to delete Tango::DevVarXArray* objects.
///// It is called by python. The array was associated with an attribute
///// value object that is not being used anymore.
///// @param ptr_ The array object.
///// @param type_ The type of the array objects. We need it to convert ptr_
/////              to the proper type before deleting it.
/////              ex: Tango::DEVVAR_SHORTARRAY.
//#    ifdef PYCAPSULE_OLD
//         template<long type>
//         static void dev_var_x_array_deleter__(void * ptr_)
//         {
//             TANGO_DO_ON_DEVICE_ARRAY_DATA_TYPE_ID(type,
//                 delete static_cast<TANGO_const2type(tangoTypeConst)*>(ptr_);
//             );
//         }
//#    else
//         template<long type>
//         static void dev_var_x_array_deleter__(PyObject* obj)
//         {
//             void * ptr_ = PyCapsule_GetPointer(obj, NULL);
//             TANGO_DO_ON_DEVICE_ARRAY_DATA_TYPE_ID(type,
//                 delete static_cast<TANGO_const2type(tangoTypeConst)*>(ptr_);
//             );
//         }
//#endif
//#endif

template<long tangoArrayTypeConst>
void extract_array(const CORBA::Any &any, py::object& py_result)
{
    typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

    TangoArrayType *tmp_ptr;

    if ((any >>= tmp_ptr) == false)
        throw_bad_type(Tango::CmdArgTypeName[tangoArrayTypeConst]);

//#ifndef DISABLE_PYTANGO_NUMPY
//      // For numpy we need a 'guard' object that handles the memory used
//      // by the numpy object (releases it).
//      // But I cannot manage memory inside our 'any' object, because it is
//      // const and handles it's memory itself. So I need a copy before
//      // creating the object.
//      TangoArrayType* copy_ptr = new TangoArrayType(*tmp_ptr);
//
//      // numpy.ndarray() does not own it's memory, so we need to manage it.
//      // We can assign a 'base' object that will be informed (decref'd) when
//      // the last copy of numpy.ndarray() disappears.
//      // PyCObject is intended for that kind of things. It's seen as a
//      // black box object from python. We assign him a function to be called
//      // when it is deleted -> the function deletes de data.

//    py::capsule free_when_done(reinterpret_cast<void*>(value_ptr), [](void* f) {
//        TangoScalarType *ptr = reinterpret_cast<TangoScalarType *>(f);
//        delete[] ptr;
//    });

    //      PyObject* guard = PyCapsule_New(
//              static_cast<void*>(copy_ptr),
//              NULL,
//              dev_var_x_array_deleter__<tangoArrayTypeConst>);
//      if (!guard ) {
//          delete copy_ptr;
//          throw_error_already_set();
//      }
//
//      py_result = to_py_numpy<tangoArrayTypeConst>(copy_ptr, object(handle<>(guard)));
//#else
//      py_result = to_py_list(tmp_ptr);
      py_result = py::none();
//#endif
}

template<>
void extract_array<Tango::DEV_PIPE_BLOB>(const CORBA::Any &any,
                     py::object& py_result)
{
    assert(false);
}

void __insert(CORBA::Any& any, py::object& py_value, const Tango::CmdArgType type)
{
    // This might be a bit verbose but at least WYSIWYG
    switch (type)
    {
    case Tango::DEV_BOOLEAN:
        insert_scalar_boolean(any, py_value);
        break;
    case Tango::DEV_SHORT:
        insert_scalar<Tango::DEV_SHORT>(any, py_value);
        break;
    case Tango::DEV_LONG:
        insert_scalar<Tango::DEV_LONG>(any, py_value);
        break;
    case Tango::DEV_FLOAT:
        insert_scalar<Tango::DEV_FLOAT>(any, py_value);
        break;
    case Tango::DEV_DOUBLE:
        insert_scalar<Tango::DEV_DOUBLE>(any, py_value);
        break;
    case Tango::DEV_USHORT:
        insert_scalar<Tango::DEV_USHORT>(any, py_value);
        break;
    case Tango::DEV_ULONG:
        insert_scalar<Tango::DEV_ULONG>(any, py_value);
        break;
    case Tango::DEV_STRING:
        insert_scalar_string(any, py_value);
        break;
    case Tango::DEV_STATE:
        insert_scalar<Tango::DEV_STATE>(any, py_value);
        break;
    case Tango::DEV_LONG64:
        insert_scalar<Tango::DEV_LONG64>(any, py_value);
        break;
    case Tango::DEV_ULONG64:
        insert_scalar<Tango::DEV_ULONG64>(any, py_value);
        break;
    case Tango::DEV_ENCODED:
        insert_scalar_encoded(any, py_value);
        break;
    case Tango::DEV_ENUM:
        insert_scalar<Tango::DEV_ENUM>(any, py_value);
        break;
//    case Tango::DEVVAR_CHARARRAY:
//        insert_array<Tango::DEV_UCHAR>(any, py_value);
//        break;
    case Tango::DEVVAR_SHORTARRAY:
        insert_array<Tango::DEVVAR_SHORTARRAY>(any, py_value);
        break;
    case Tango::DEVVAR_LONGARRAY:
        insert_array<Tango::DEVVAR_LONGARRAY>(any, py_value);
        break;
    case Tango::DEVVAR_FLOATARRAY:
        insert_array<Tango::DEVVAR_FLOATARRAY>(any, py_value);
        break;
    case Tango::DEVVAR_DOUBLEARRAY:
        insert_array<Tango::DEVVAR_DOUBLEARRAY>(any, py_value);
        break;
    case Tango::DEVVAR_USHORTARRAY:
        insert_array<Tango::DEVVAR_USHORTARRAY>(any, py_value);
        break;
    case Tango::DEVVAR_ULONGARRAY:
        insert_array<Tango::DEVVAR_ULONGARRAY>(any, py_value);
        break;
    case Tango::DEVVAR_BOOLEANARRAY:

        insert_array<Tango::DEVVAR_BOOLEANARRAY>(any, py_value);
        break;
    case Tango::DEVVAR_LONG64ARRAY:
        insert_array<Tango::DEVVAR_LONG64ARRAY>(any, py_value);
        break;
    case Tango::DEVVAR_ULONG64ARRAY:
        insert_array<Tango::DEVVAR_ULONG64ARRAY>(any, py_value);
        break;
    case Tango::DEVVAR_STATEARRAY:
        insert_array<Tango::DEVVAR_STATEARRAY>(any, py_value);
        break;
//        __TANGO_DEPEND_ON_TYPE_AUX_ID(DEVVAR_STRINGARRAY, DOIT_ARRAY) \
//        __TANGO_DEPEND_ON_TYPE_AUX_ID(DEVVAR_LONGSTRINGARRAY, DOIT_ARRAY) \
//        __TANGO_DEPEND_ON_TYPE_AUX_ID(DEVVAR_DOUBLESTRINGARRAY, DOIT_ARRAY) \
    default:
        throw;
    }
}

CORBA::Any *PyCmd::execute(Tango::DeviceImpl *dev, const CORBA::Any &param_any)
{
    Device_5ImplWrap *dev_ptr = (Device_5ImplWrap*)dev;

    AutoPythonGILEnsure __py_lock;
    std::cout << "Got to execute in command.cpp" << std::endl;
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
        // Insert the python object into CORBA any for return
        __insert(*ret_any, ret_py_obj, out_type);

        // It does: ret_any = ret_py_obj
//        TANGO_DO_ON_DEVICE_DATA_TYPE_ID(out_type,
//            insert_scalar<tangoTypeConst>(ret_py_obj, *ret_any);
//        ,
//            insert_array<tangoTypeConst>(ret_py_obj, *ret_any);
//        );

        return ret_any_guard.release();
    } catch(py::error_already_set &eas) {
        handle_python_exception(eas);
        return 0; // Should not happen, handle_python_exception rethrows in
                  // a Tango friendly manner
    }
}
