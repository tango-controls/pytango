/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "exception.h"
#include "pytgutils.h"
#include "fast_from_py.h"
#include "server/device_impl.h"
#include "server/command.h"

#ifndef DISABLE_PYTANGO_NUMPY
#   include "to_py_numpy.hpp"
#endif

#include <memory>

using namespace boost::python;

//+-------------------------------------------------------------------------
//
// method : 		PyCmd::is_allowed
//
// description : 	Decide if it is allowed to execute the command
//
// argin : - dev : The device on which the command has to be excuted
//	   - any : The input data
//
// This method returns a boolean set to True if it is allowed to execute
// the command. Otherwise, returns false
//
//--------------------------------------------------------------------------
bool PyCmd::is_allowed(Tango::DeviceImpl *dev, const CORBA::Any &any)
{
    if (py_allowed_defined == true)
    {
        PyDeviceImplBase *dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
        //Device_4ImplWrap *dev_ptr = static_cast<Device_4ImplWrap *>(dev);

        AutoPythonGIL __py_lock;

        bool returned_value = true;
        try
        {
            returned_value = call_method<bool>(dev_ptr->the_self, py_allowed_name.c_str());
        }
        catch(error_already_set &eas)
        {
            handle_python_exception(eas);
        }

        return returned_value;
    }
    else
        return true;
}


void allocate_any(CORBA::Any *&any_ptr)
{
    try
    {
        any_ptr = new CORBA::Any();
    }
    catch (bad_alloc)
    {
        Tango::Except::throw_exception(
            "API_MemoryAllocation",
            "Can't allocate memory in server",
            "PyCmd::allocate_any()");
    }
}

void throw_bad_type(const char *type)
{
    TangoSys_OMemStream o;

    o << "Incompatible command argument type, expected type is : Tango::" 
      << type << ends;
    Tango::Except::throw_exception(
            "API_IncompatibleCmdArgumentType",
            o.str(),
            "PyCmd::extract()");
}

template<long tangoTypeConst>
void insert_scalar(boost::python::object &o, CORBA::Any &any)
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    
    any <<= extract<TangoScalarType>(o);
}

template<>
void insert_scalar<Tango::DEV_STRING>(boost::python::object &o, CORBA::Any &any)
{
    PyObject *o_ptr = o.ptr();
    if (PyUnicode_Check(o_ptr))
    {
        PyObject *bytes_o_ptr = PyUnicode_AsLatin1String(o_ptr);
	any <<= PyBytes_AsString(bytes_o_ptr);
        Py_DECREF(bytes_o_ptr);
    }
    else
    {
        any <<= PyBytes_AsString(o_ptr);
    }
}

template<>
void insert_scalar<Tango::DEV_VOID>(boost::python::object &o, CORBA::Any &any)
{}

template<>
void insert_scalar<Tango::DEV_BOOLEAN>(boost::python::object &o, CORBA::Any &any)
{
    Tango::DevBoolean value = extract<Tango::DevBoolean>(o);
    CORBA::Any::from_boolean any_value(value);

    any <<= any_value;
}

template<>
void insert_scalar<Tango::DEV_ENCODED>(boost::python::object &o, CORBA::Any &any)
{
    bopy::object p0 = o[0];
    bopy::object p1 = o[1];

    const char* encoded_format = bopy::extract<const char *> (p0.ptr());
    
    PyObject* data_ptr = p1.ptr();
    Py_buffer view;
    
    if (PyObject_GetBuffer(data_ptr, &view, PyBUF_FULL_RO) < 0)
    {
        throw_bad_type(Tango::CmdArgTypeName[Tango::DEV_ENCODED]);
    }
    
    CORBA::ULong nb = static_cast<CORBA::ULong>(view.len);
    Tango::DevVarCharArray arr(nb, nb, (CORBA::Octet*)view.buf, false);
    Tango::DevEncoded *data = new Tango::DevEncoded;
    data->encoded_format = CORBA::string_dup(encoded_format);
    data->encoded_data = arr;    
    
    any <<= data;

    PyBuffer_Release(&view);
}

template<>
void insert_scalar<Tango::DEV_PIPE_BLOB>(boost::python::object &o, CORBA::Any &any)
{
    assert(false);
}

template<long tangoArrayTypeConst>
void insert_array(boost::python::object &o, CORBA::Any &any)
{   
    typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

    // Destruction will be handled by CORBA, not by Tango.
    TangoArrayType* data = fast_convert2array<tangoArrayTypeConst>(o);
    
    // By giving a pointer to <<= we are giving ownership of the data 
    // buffer to CORBA
    any <<= data;
}

template<>
void insert_array<Tango::DEV_PIPE_BLOB>(boost::python::object &o, CORBA::Any &any)
{
    assert(false);
}

template<long tangoTypeConst>
void extract_scalar(const CORBA::Any &any, boost::python::object &o)
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    
    TangoScalarType data;
    
    if ((any >>= data) == false)
        throw_bad_type(Tango::CmdArgTypeName[tangoTypeConst]);

    o = object(data);
}

template<>
void extract_scalar<Tango::DEV_STRING>(const CORBA::Any &any, boost::python::object &o)
{
    Tango::ConstDevString data;
    
    if ((any >>= data) == false)
        throw_bad_type(Tango::CmdArgTypeName[Tango::DEV_STRING]);

    o = from_char_to_boost_str(data);
}

template<>
void extract_scalar<Tango::DEV_VOID>(const CORBA::Any &any, boost::python::object &o)
{}

template<>
void extract_scalar<Tango::DEV_PIPE_BLOB>(const CORBA::Any &any, boost::python::object &o)
{
    assert(false);
}

template<>
void extract_scalar<Tango::DEV_ENCODED>(const CORBA::Any &any, boost::python::object &o)
{
    Tango::DevEncoded* data;

    if ((any >>= data) == false)
        throw_bad_type(Tango::CmdArgTypeName[Tango::DEV_ENCODED]);
    
    bopy::str encoded_format(data[0].encoded_format);

#if PY_MAJOR_VERSION >= 3
    boost::python::object encoded_data(
        boost::python::handle<>(
            PyBytes_FromStringAndSize(
                (char*)data[0].encoded_data.get_buffer(),
                data[0].encoded_data.length()
            )
        )
    );
#else
    bopy::str encoded_data((const char*)data[0].encoded_data.get_buffer(),
                           data[0].encoded_data.length());
#endif

    o = boost::python::make_tuple(encoded_format, encoded_data);
}

#ifndef DISABLE_PYTANGO_NUMPY
/// This callback is run to delete Tango::DevVarXArray* objects.
/// It is called by python. The array was associated with an attribute
/// value object that is not being used anymore.
/// @param ptr_ The array object.
/// @param type_ The type of the array objects. We need it to convert ptr_
///              to the proper type before deleting it.
///              ex: Tango::DEVVAR_SHORTARRAY.
#    ifdef PYCAPSULE_OLD
         template<long type>
         static void dev_var_x_array_deleter__(void * ptr_)
         {
             TANGO_DO_ON_DEVICE_ARRAY_DATA_TYPE_ID(type,
                 delete static_cast<TANGO_const2type(tangoTypeConst)*>(ptr_);
             );
         }
#    else
         template<long type>
         static void dev_var_x_array_deleter__(PyObject* obj)
         {
             void * ptr_ = PyCapsule_GetPointer(obj, NULL);
             TANGO_DO_ON_DEVICE_ARRAY_DATA_TYPE_ID(type,
                 delete static_cast<TANGO_const2type(tangoTypeConst)*>(ptr_);
             );
         }
#endif
#endif

template<long tangoArrayTypeConst>
void extract_array(const CORBA::Any &any, boost::python::object &py_result)
{
    typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
    
    TangoArrayType *tmp_ptr;

    if ((any >>= tmp_ptr) == false)
        throw_bad_type(Tango::CmdArgTypeName[tangoArrayTypeConst]);

#ifndef DISABLE_PYTANGO_NUMPY
      // For numpy we need a 'guard' object that handles the memory used
      // by the numpy object (releases it).
      // But I cannot manage memory inside our 'any' object, because it is
      // const and handles it's memory itself. So I need a copy before
      // creating the object.
      TangoArrayType* copy_ptr = new TangoArrayType(*tmp_ptr);

      // numpy.ndarray() does not own it's memory, so we need to manage it.
      // We can assign a 'base' object that will be informed (decref'd) when
      // the last copy of numpy.ndarray() disappears.
      // PyCObject is intended for that kind of things. It's seen as a
      // black box object from python. We assign him a function to be called
      // when it is deleted -> the function deletes de data.
      PyObject* guard = PyCapsule_New(
              static_cast<void*>(copy_ptr),
              NULL,
              dev_var_x_array_deleter__<tangoArrayTypeConst>);
      if (!guard ) {
          delete copy_ptr;
          throw_error_already_set();
      }

      py_result = to_py_numpy<tangoArrayTypeConst>(copy_ptr, object(handle<>(guard)));
#else
      py_result = to_py_list(tmp_ptr);
#endif
}

template<>
void extract_array<Tango::DEV_PIPE_BLOB>(const CORBA::Any &any,
					 boost::python::object &py_result)
{
    assert(false);
}

CORBA::Any *PyCmd::execute(Tango::DeviceImpl *dev, const CORBA::Any &param_any)
{
    PyDeviceImplBase *dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
    
    AutoPythonGIL python_guard;
    try 
    {   
        // This call extracts the CORBA any into a python object.
        // So, the result is that param_py = param_any.
        // It is done with some template magic.
        boost::python::object param_py;
        TANGO_DO_ON_DEVICE_DATA_TYPE_ID(in_type,
            extract_scalar<tangoTypeConst>(param_any, param_py);
        , 
            extract_array<tangoTypeConst>(param_any, param_py);
        );

        // Execute the python call for the command
        object ret_py_obj;

        if (in_type == Tango::DEV_VOID)
        {
            ret_py_obj = call_method<object>(dev_ptr->the_self, name.c_str());
        }
        else
        {
            ret_py_obj = call_method<object>(dev_ptr->the_self, name.c_str(), param_py);
        }
        
        CORBA::Any *ret_any;
        allocate_any(ret_any);
        unique_pointer<CORBA::Any> ret_any_guard(ret_any);

        // It does: ret_any = ret_py_obj
        TANGO_DO_ON_DEVICE_DATA_TYPE_ID(out_type,
            insert_scalar<tangoTypeConst>(ret_py_obj, *ret_any);
        ,
            insert_array<tangoTypeConst>(ret_py_obj, *ret_any);
        );
        
        return ret_any_guard.release();
    } catch(error_already_set &eas) {
        handle_python_exception(eas);
        return 0; // Should not happen, handle_python_exception rethrows in
                  // a Tango friendly manner
    }
}
