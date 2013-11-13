/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "pytgutils.h"
#include "fast_from_py.h"

using namespace boost::python;

#ifndef DISABLE_PYTANGO_NUMPY
#   include "to_py_numpy.hpp"
#endif

namespace PyDeviceData {

    Tango::CmdArgType get_type(Tango::DeviceData &self)
    {
        /// @todo This should change in Tango itself, get_type should not return int!!
        return static_cast<Tango::CmdArgType>(self.get_type());
    }
    
    /// @name Scalar Insertion
    /// @{
        template <long tangoTypeConst>
        void insert_scalar(Tango::DeviceData &self, object py_value)
        {
            typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
            TangoScalarType val;
            val = boost::python::extract<TangoScalarType>(py_value);
            self << val;
        }
        template <>
        void insert_scalar<Tango::DEV_ENCODED>(Tango::DeviceData &self, object py_value)
        {
            object p0 = py_value[0];
            object p1 = py_value[1];
            const char* encoded_format = extract<const char*> (p0.ptr());
            const char* encoded_data = extract<const char*> (p1.ptr());
            
            CORBA::ULong nb = static_cast<CORBA::ULong>(boost::python::len(p1));
            Tango::DevVarCharArray arr(nb, nb, (CORBA::Octet*)encoded_data, false);
            Tango::DevEncoded val;
            val.encoded_format = CORBA::string_dup(encoded_format);
            val.encoded_data = arr;
            self << val;
        }
        template <>
        void insert_scalar<Tango::DEV_VOID>(Tango::DeviceData &self, object py_value)
        {
            raise_(PyExc_TypeError, "Trying to insert a value in a DEV_VOID DeviceData!");
        }
    /// @}
    // ~Scalar Insertion
    // -----------------------------------------------------------------------

    /// @name Array Insertion
    /// @{
        template <long tangoArrayTypeConst>
        void insert_array(Tango::DeviceData &self, object py_value)
        {
            typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

            // self << val; -->> This ends up doing:
            // inline void operator << (DevVarUShortArray* datum) 
            // { any.inout() <<= datum;}
            // So:
            //  - We loose ownership of the pointer, should not remove it
            //  - it's a CORBA object who gets ownership, not a buggy Tango
            //    thing. So the last parameter to fast_convert2array is false
            TangoArrayType* val = fast_convert2array<tangoArrayTypeConst>(py_value);
            self << val;
        }
    /// @}
    // ~Array Insertion
    // -----------------------------------------------------------------------

    /// @name Scalar Extraction
    /// @{
        template <long tangoTypeConst>
        object extract_scalar(Tango::DeviceData &self)
        {
            typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
            /// @todo CONST_DEV_STRING ell el tracta com DEV_STRING
            TangoScalarType val;
            self >> val;
            return boost::python::object(val);
        }

        template <>
        object extract_scalar<Tango::DEV_VOID>(Tango::DeviceData &self)
        {
            return boost::python::object();
        }

        template <>
        object extract_scalar<Tango::DEV_STRING>(Tango::DeviceData &self)
        {
            typedef std::string TangoScalarType;
            TangoScalarType val;
            self >> val;
            return boost::python::object(val);
        }
    /// @}
    // ~Scalar Extraction
    // -----------------------------------------------------------------------

    /// @name Array extraction
    /// @{       

        template <long tangoArrayTypeConst>
        object extract_array(Tango::DeviceData &self, object &py_self, PyTango::ExtractAs extract_as)
        {
            typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
            
            // const is the pointed, not the pointer. So cannot modify the data.
            // And that's because the data is still inside "self" after extracting.
            // This also means that we are not supposed to "delete" tmp_ptr.
            const TangoArrayType* tmp_ptr;
            self >> tmp_ptr;

            switch (extract_as)
            {
                default:
                case PyTango::ExtractAsNumpy:
#                 ifndef DISABLE_PYTANGO_NUMPY
                    return to_py_numpy<tangoArrayTypeConst>(tmp_ptr, py_self);
#                 endif
                case PyTango::ExtractAsList:
                case PyTango::ExtractAsPyTango3:
                    return to_py_list(tmp_ptr);
                case PyTango::ExtractAsTuple:
                    return to_py_tuple(tmp_ptr);
                case PyTango::ExtractAsString: /// @todo
                case PyTango::ExtractAsNothing:
                    return object();
            }
        }
    /// @}
    // ~Array Extraction
    // -----------------------------------------------------------------------

    object extract(object py_self, PyTango::ExtractAs extract_as)
    {
        Tango::DeviceData &self = boost::python::extract<Tango::DeviceData &>(py_self);
        
        TANGO_DO_ON_DEVICE_DATA_TYPE_ID(self.get_type(),
            return extract_scalar<tangoTypeConst>(self);
        ,
            return extract_array<tangoTypeConst>(self, py_self, extract_as);
        );
        return object();
    }

    void insert(Tango::DeviceData &self, long data_type, object py_value)
    {
        TANGO_DO_ON_DEVICE_DATA_TYPE_ID(data_type,
            insert_scalar<tangoTypeConst>(self, py_value);
        , 
            insert_array<tangoTypeConst>(self, py_value);
        );
    }
}

void export_device_data()
{
    class_<Tango::DeviceData> DeviceData("DeviceData",
        init<>())
    ;

    scope scope_dd = DeviceData;

    /// @todo get rid of except_flags everywhere... or really use and export them everywhere!
    enum_<Tango::DeviceData::except_flags>("except_flags")
        .value("isempty_flag", Tango::DeviceData::isempty_flag)
        .value("wrongtype_flag", Tango::DeviceData::wrongtype_flag)
        .value("numFlags", Tango::DeviceData::numFlags)
    ;

    DeviceData
        .def(init<const Tango::DeviceData &>())

        .def("extract",
            &PyDeviceData::extract,
            ( arg_("self"), arg_("extract_as")=PyTango::ExtractAsNumpy ))

        .def("insert", &PyDeviceData::insert,
            (arg_("self"), arg_("data_type"), arg_("value")))

        /// @todo do not throw exceptions!!
        .def("is_empty", &Tango::DeviceData::is_empty)

// TODO
//	void exceptions(bitset<numFlags> fl) {exceptions_flags = fl;}
//	bitset<numFlags> exceptions() {return exceptions_flags;}
//	void reset_exceptions(except_flags fl) {exceptions_flags.reset((size_t)fl);}
//	void set_exceptions(except_flags fl) {exceptions_flags.set((size_t)fl);}

        .def("get_type", &PyDeviceData::get_type)
    ;

}
