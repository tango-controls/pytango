/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "device_pipe.h"
#include "tgutils.h"
#include "pytgutils.h"
#include "tango_numpy.h"
#include "fast_from_py.h"

#ifndef DISABLE_PYTANGO_NUMPY
#   include "to_py_numpy.hpp"
#endif

namespace PyTango 
{ 
    namespace DevicePipe 
    {
        bopy::object extract(Tango::DevicePipeBlob&, PyTango::ExtractAs);

        template<long tangoTypeConst>
        bopy::object
        __update_scalar_values(Tango::DevicePipe& self, size_t elt_idx)
        {
            typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
            TangoScalarType val;
            bopy::str name(self.get_data_elt_name(elt_idx));
            self >> val;
            bopy::object data(val);
            return bopy::make_tuple(name, data);
        }

        template<>
        bopy::object
        __update_scalar_values<Tango::DEV_VOID>(Tango::DevicePipe& self,
                                          size_t elt_idx)
        {
            bopy::str name(self.get_data_elt_name(elt_idx));
            return bopy::make_tuple(name, bopy::object());
        }

        template<>
        bopy::object
        __update_scalar_values<Tango::DEV_STRING>(Tango::DevicePipe& self,
                                            size_t elt_idx)
        {
            typedef std::string TangoScalarType;
            TangoScalarType val;
            bopy::str name(self.get_data_elt_name(elt_idx));
            self >> val;
            bopy::object data(val);
            return bopy::make_tuple(name, data);
        }        

        template<>
        bopy::object
        __update_scalar_values<Tango::DEV_PIPE_BLOB>(Tango::DevicePipe& self,
                                               size_t elt_idx)
        {
            Tango::DevicePipeBlob val;
            bopy::str name(self.get_data_elt_name(elt_idx));
            self >> val;
            bopy::object data = extract(val, PyTango::ExtractAsNumpy);
            return bopy::make_tuple(name, data);
        }        

        template <long tangoArrayTypeConst>
        bopy::object
        __update_array_values(Tango::DevicePipe &self, bopy::object &py_self,
                        size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

            TangoArrayType tmp_arr;
            self >> (&tmp_arr);
            bopy::object data;
            switch (extract_as)
            {
                default:
                case PyTango::ExtractAsNumpy:
#                 ifndef DISABLE_PYTANGO_NUMPY
                    data = to_py_numpy<tangoArrayTypeConst>(&tmp_arr, py_self);
                    tmp_arr.get_buffer(1);
                    break;
#                 endif
                case PyTango::ExtractAsList:
                case PyTango::ExtractAsPyTango3:
                    data = to_py_list(&tmp_arr);
                    break;
                case PyTango::ExtractAsTuple:
                    data = to_py_tuple(&tmp_arr);
                    break;
                case PyTango::ExtractAsString: /// @todo
                case PyTango::ExtractAsNothing:
                    data = bopy::object();
                    break;
            }

            bopy::str name(self.get_data_elt_name(elt_idx));
            return bopy::make_tuple(name, data);
        }

        template <>
        bopy::object
        __update_array_values<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::DevicePipe &self,
                                                       bopy::object &py_self,
                                                       size_t elt_idx, 
                                                       PyTango::ExtractAs extract_as)
        {
            assert(false);
            return bopy::object();
        }

        template <>
        bopy::object
        __update_array_values<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::DevicePipe &self,
                                                         bopy::object &py_self,
                                                         size_t elt_idx, 
                                                         PyTango::ExtractAs extract_as)
        {
            assert(false);
            return bopy::object();
        }

        bopy::object
        update_value(Tango::DevicePipe &self, bopy::object& py_self,
                  size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            const int elt_type = self.get_data_elt_type(elt_idx);

            TANGO_DO_ON_DEVICE_DATA_TYPE_ID(elt_type,
                return __update_scalar_values<tangoTypeConst>(self, elt_idx);
            ,
                return __update_array_values<tangoTypeConst>(self, py_self, elt_idx, extract_as);
            );
            return bopy::object();
        }
        
        void
        update_values(Tango::DevicePipe& self, bopy::object& py_self,
                      PyTango::ExtractAs extract_as /*=PyTango::ExtractAsNumpy*/)
        {
            // We do not want is_empty to launch an exception!!
            //self.reset_exceptions(Tango::DevicePipe::isempty_flag);

            //py_self.attr("name") = self.get_name();
            bopy::list data;
            py_self.attr("data") = data;

            size_t elt_nb = self.get_data_elt_nb();
            for(size_t elt_idx = 0; elt_idx < elt_nb; ++elt_idx)
            {
                data.append(update_value(self, py_self, elt_idx, extract_as));
            }
        }

      ///////////////////////////////////////////////////////////////////////////////////////////
      
        template<typename T, long tangoTypeConst>
        bopy::object
        __extract_scalar(T& obj, size_t elt_idx)
	{
            typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
	    TangoScalarType val;
	    obj >> val;
	    return bopy::object(val);
	}

        template<>
        bopy::object
        __extract_scalar<Tango::DevicePipe, Tango::DEV_VOID>(Tango::DevicePipe& obj, size_t elt_idx)
	{
            return bopy::object();
	}

        template<>
        bopy::object
        __extract_scalar<Tango::DevicePipe, Tango::DEV_STRING>(Tango::DevicePipe& obj, size_t elt_idx)
	{
            std::string val;
	    obj >> val;
	    return bopy::object(val);
	}

        template<>
        bopy::object
        __extract_scalar<Tango::DevicePipe, Tango::DEV_PIPE_BLOB>(Tango::DevicePipe& obj, size_t elt_idx)
	{
	    Tango::DevicePipeBlob val;
	    obj >> val;
	    // TODO: propagate extract_as
	    return extract(val, PyTango::ExtractAsNumpy);
	}

        template<>
        bopy::object
        __extract_scalar<Tango::DevicePipeBlob, Tango::DEV_VOID>(Tango::DevicePipeBlob& obj, size_t elt_idx)
	{
            return bopy::object();
	}

        template<>
        bopy::object
        __extract_scalar<Tango::DevicePipeBlob, Tango::DEV_STRING>(Tango::DevicePipeBlob& obj, size_t elt_idx)
	{
            std::string val;
	    obj >> val;
	    return bopy::object(val);
	}

        template<>
        bopy::object
        __extract_scalar<Tango::DevicePipeBlob, Tango::DEV_PIPE_BLOB>(Tango::DevicePipeBlob& obj, size_t elt_idx)
	{
	    Tango::DevicePipeBlob val;
	    obj >> val;
	    // TODO: propagate extract_as
	    return extract(val, PyTango::ExtractAsNumpy);
	}
      
        template<long tangoTypeConst>
        bopy::object
        extract_scalar(Tango::DevicePipe& self, size_t elt_idx)
        {
 	    return __extract_scalar<Tango::DevicePipe, tangoTypeConst>(self, elt_idx);
        }

        template<long tangoTypeConst>
        bopy::object
        extract_scalar(Tango::DevicePipeBlob& self, size_t elt_idx)
        {
 	    return __extract_scalar<Tango::DevicePipeBlob, tangoTypeConst>(self, elt_idx);
        }

        template <typename T, long tangoArrayTypeConst>
        bopy::object
        __extract_array(T& obj, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

            TangoArrayType tmp_arr;
            obj >> (&tmp_arr);
            bopy::object data;
            switch (extract_as)
            {
                default:
                case PyTango::ExtractAsNumpy:

#                 ifndef DISABLE_PYTANGO_NUMPY
                    data = to_py_numpy<tangoArrayTypeConst>(&tmp_arr, 1);
                    break;
#                 endif

                case PyTango::ExtractAsList:
                case PyTango::ExtractAsPyTango3:
                    data = to_py_list(&tmp_arr);
                    break;
                case PyTango::ExtractAsTuple:
                    data = to_py_tuple(&tmp_arr);
                    break;
                case PyTango::ExtractAsString: /// @todo
                case PyTango::ExtractAsNothing:
                    data = bopy::object();
                    break;
            }
            return data;
        }

        template <>
        bopy::object
        __extract_array<Tango::DevicePipe, Tango::DEVVAR_LONGSTRINGARRAY>
	(Tango::DevicePipe& pipe, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            assert(false);
            return bopy::object();
        }

        template <>
        bopy::object
        __extract_array<Tango::DevicePipe, Tango::DEVVAR_DOUBLESTRINGARRAY>
	(Tango::DevicePipe& pipe, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            assert(false);
            return bopy::object();
        }

        template <>
        bopy::object
        __extract_array<Tango::DevicePipeBlob, Tango::DEVVAR_LONGSTRINGARRAY>
	(Tango::DevicePipeBlob& blob, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            assert(false);
            return bopy::object();
        }

        template <>
        bopy::object
        __extract_array<Tango::DevicePipeBlob, Tango::DEVVAR_DOUBLESTRINGARRAY>
	(Tango::DevicePipeBlob& blob, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            assert(false);
            return bopy::object();
        }

        template <long tangoArrayTypeConst>
        bopy::object
        extract_array(Tango::DevicePipe& self, size_t elt_idx,
		      PyTango::ExtractAs extract_as)
        {
	  return __extract_array<Tango::DevicePipe, tangoArrayTypeConst>(self, elt_idx,
									 extract_as);
	}

        template <long tangoArrayTypeConst>
        bopy::object
        extract_array(Tango::DevicePipeBlob& self, size_t elt_idx,
		      PyTango::ExtractAs extract_as)
        {
	  return __extract_array<Tango::DevicePipeBlob, tangoArrayTypeConst>(self, elt_idx,
									     extract_as);
	}
      
        template<typename T>
        bopy::object
        __extract_item(T& obj, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
	    const int elt_type = obj.get_data_elt_type(elt_idx);
            TANGO_DO_ON_DEVICE_DATA_TYPE_ID(elt_type,
                return extract_scalar<tangoTypeConst>(obj, elt_idx);
            ,
                return extract_array<tangoTypeConst>(obj, elt_idx, extract_as);
            );
            return bopy::object();     
	}

        template<typename T>
        bopy::object
        __extract(T& obj, PyTango::ExtractAs extract_as)
        {
            bopy::list data;
            size_t elt_nb = obj.get_data_elt_nb();
            for(size_t elt_idx = 0; elt_idx < elt_nb; ++elt_idx)
            {
	        bopy::dict elem;
		elem["name"] = obj.get_data_elt_name(elt_idx);
		elem["dtype"] = static_cast<Tango::CmdArgType>(obj.get_data_elt_type(elt_idx));
		elem["value"] = __extract_item(obj, elt_idx, extract_as);
	        data.append(elem);
            }
	    return data;
        }      

        bopy::object
	extract(Tango::DevicePipeBlob& blob, 
		PyTango::ExtractAs extract_as=PyTango::ExtractAsNumpy)
	{
	    bopy::object name = bopy::str(blob.get_name());
	    bopy::object value = __extract<Tango::DevicePipeBlob>(blob, extract_as);
	    return bopy::make_tuple(name, value);
	}

        bopy::object
	extract(Tango::DevicePipe& device_pipe, 
		PyTango::ExtractAs extract_as=PyTango::ExtractAsNumpy)
	{
	    bopy::object name = bopy::str(device_pipe.get_root_blob_name());
	    bopy::object value = __extract<Tango::DevicePipe>(device_pipe, extract_as);
	    return bopy::make_tuple(name, value);
	}
    }
}

void export_device_pipe()
{
    bopy::class_<Tango::DevicePipe> DevicePipe("DevicePipe");

    bopy::scope dp_scope = DevicePipe;

    DevicePipe
        .def(bopy::init<>())
        .def(bopy::init<const std::string &>())
        .def(bopy::init<const std::string &, const std::string &>())
        .def(bopy::init<const Tango::DevicePipe &>())
        .add_property("name", 
                      bopy::make_function(&Tango::DevicePipe::get_name,
                                          bopy::return_value_policy
                                          <bopy::copy_const_reference>()),
                      &Tango::DevicePipe::set_name)
        .add_property("root_blob_name",
                      bopy::make_function(&Tango::DevicePipe::get_root_blob_name,
                                          bopy::return_value_policy
                                          <bopy::copy_const_reference>()),
                      &Tango::DevicePipe::set_root_blob_name)
        .add_property("data_elt_nb",
                      &Tango::DevicePipe::get_data_elt_nb, 
                      &Tango::DevicePipe::set_data_elt_nb)        
        .add_property("data_elt_names",
                      &Tango::DevicePipe::get_data_elt_names, 
                      &Tango::DevicePipe::set_data_elt_names)        
        .def("get_data_elt_name", &Tango::DevicePipe::get_data_elt_name)
        .def("get_data_elt_type", &Tango::DevicePipe::get_data_elt_type)

        .def("extract", 
	     (bopy::object (*) (Tango::DevicePipe &, PyTango::ExtractAs))
	     PyTango::DevicePipe::extract)

		 .def("extract",
	     (bopy::object (*) (Tango::DevicePipeBlob &, PyTango::ExtractAs))
	     PyTango::DevicePipe::extract)
    ;
}
