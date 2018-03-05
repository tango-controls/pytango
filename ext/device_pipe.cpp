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
#include <defs.h>
#include <tgutils.h>
#include <to_py.h>
#include <device_pipe.h>
#include <tango_numpy.h>
#ifndef DISABLE_PYTANGO_NUMPY
#   include "to_py_numpy.hpp"
#endif

namespace py = pybind11;

namespace PyTango
{
    namespace DevicePipe
    {
        template<long tangoTypeConst>
        py::object __update_scalar_values(Tango::DevicePipe& self, size_t elt_idx)
        {
            typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
            TangoScalarType val;
            py::str name(self.get_data_elt_name(elt_idx));
            self >> val;
            py::object data = py::cast(val, py::return_value_policy::reference);
            return py::make_tuple(name, val);
        }

        template<>
        py::object __update_scalar_values<Tango::DEV_VOID>(Tango::DevicePipe& self,
                                          size_t elt_idx)
        {
            py::str name(self.get_data_elt_name(elt_idx));
            return py::make_tuple(name, py::object());
        }

        template<>
        py::object __update_scalar_values<Tango::DEV_STRING>(Tango::DevicePipe& self,
                                            size_t elt_idx)
        {
            typedef std::string TangoScalarType;
            TangoScalarType val;
            py::str name(self.get_data_elt_name(elt_idx));
            self >> val;
            py::object data = py::cast(val, py::return_value_policy::reference);
            return py::make_tuple(name, data);
        }

        template<>
        py::object __update_scalar_values<Tango::DEV_PIPE_BLOB>(Tango::DevicePipe& self,
                                               size_t elt_idx)
        {
            Tango::DevicePipeBlob val;
            py::str name(self.get_data_elt_name(elt_idx));
            self >> val;
            py::object data = extract(val, PyTango::ExtractAsNumpy);
            return py::make_tuple(name, data);
        }

        template <long tangoArrayTypeConst>
        py::object __update_array_values(Tango::DevicePipe &self, py::object &py_self,
                        size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

            TangoArrayType tmp_arr;
            self >> (&tmp_arr);
            py::object data;
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
                    data = py::object();
                    break;
            }

            py::str name(self.get_data_elt_name(elt_idx));
            return py::make_tuple(name, data);
        }

        template <>
        py::object __update_array_values<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::DevicePipe &self,
                                                       py::object &py_self,
                                                       size_t elt_idx,
                                                       PyTango::ExtractAs extract_as)
        {
            assert(false);
            return py::object();
        }

        template <>
        py::object __update_array_values<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::DevicePipe &self,
                                                         py::object &py_self,
                                                         size_t elt_idx,
                                                         PyTango::ExtractAs extract_as)
        {
            assert(false);
            return py::object();
        }

        py::object update_value(Tango::DevicePipe &self, py::object& py_self,
                  size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            const int elt_type = self.get_data_elt_type(elt_idx);

            TANGO_DO_ON_DEVICE_DATA_TYPE_ID(elt_type,
                return __update_scalar_values<tangoTypeConst>(self, elt_idx);
            ,
                return __update_array_values<tangoTypeConst>(self, py_self, elt_idx, extract_as);
            );
            return py::object();
        }

        void update_values(Tango::DevicePipe& self, py::object& py_self,
                      PyTango::ExtractAs extract_as /*=PyTango::ExtractAsNumpy*/)
        {
            // We do not want is_empty to launch an exception!!
            //self.reset_exceptions(Tango::DevicePipe::isempty_flag);

            //py_self.attr("name") = self.get_name();
            py::list data;
            py_self.attr("data") = data;

            size_t elt_nb = self.get_data_elt_nb();
            for(size_t elt_idx = 0; elt_idx < elt_nb; ++elt_idx)
            {
                data.append(update_value(self, py_self, elt_idx, extract_as));
            }
        }

      ///////////////////////////////////////////////////////////////////////////////////////////

        template<typename T, long tangoTypeConst>
        py::object __extract_scalar(T& obj, size_t elt_idx)
        {
            typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
            TangoScalarType val;
            obj >> val;
            py::object data = py::cast(val, py::return_value_policy::reference);
            return data;
        }

        template<>
        py::object __extract_scalar<Tango::DevicePipe, Tango::DEV_VOID>(Tango::DevicePipe& obj, size_t elt_idx)
        {
            return py::object();
        }

        template<>
        py::object __extract_scalar<Tango::DevicePipe, Tango::DEV_STRING>(Tango::DevicePipe& obj, size_t elt_idx)
        {
            std::string val;
            obj >> val;
            py::object data = py::cast(val, py::return_value_policy::reference);
           return data;
        }

        template<>
        py::object __extract_scalar<Tango::DevicePipe, Tango::DEV_PIPE_BLOB>(Tango::DevicePipe& obj, size_t elt_idx)
        {
            Tango::DevicePipeBlob val;
            obj >> val;
            // TODO: propagate extract_as
            return extract(val, PyTango::ExtractAsNumpy);
        }

        template<>
        py::object __extract_scalar<Tango::DevicePipeBlob, Tango::DEV_VOID>(Tango::DevicePipeBlob& obj, size_t elt_idx)
        {
            return py::object();
        }

        template<>
        py::object __extract_scalar<Tango::DevicePipeBlob, Tango::DEV_STRING>(Tango::DevicePipeBlob& obj, size_t elt_idx)
        {
            std::string val;
            obj >> val;
            py::object data = py::cast(val, py::return_value_policy::reference);
            return data;
        }

        template<>
        py::object __extract_scalar<Tango::DevicePipeBlob, Tango::DEV_PIPE_BLOB>(Tango::DevicePipeBlob& obj, size_t elt_idx)
        {
            Tango::DevicePipeBlob val;
            obj >> val;
            // TODO: propagate extract_as
            return extract(val, PyTango::ExtractAsNumpy);
        }

        template<long tangoTypeConst>
        py::object extract_scalar(Tango::DevicePipe& self, size_t elt_idx)
        {
            return __extract_scalar<Tango::DevicePipe, tangoTypeConst>(self, elt_idx);
        }

        template<long tangoTypeConst>
        py::object extract_scalar(Tango::DevicePipeBlob& self, size_t elt_idx)
        {
            return __extract_scalar<Tango::DevicePipeBlob, tangoTypeConst>(self, elt_idx);
        }

        template <typename T, long tangoArrayTypeConst>
        py::object __extract_array(T& obj, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;

            TangoArrayType tmp_arr;
            obj >> (&tmp_arr);
            py::object data;
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
                    data = py::object();
                    break;
            }
            return data;
        }

        template <>
        py::object __extract_array<Tango::DevicePipe, Tango::DEVVAR_LONGSTRINGARRAY>
            (Tango::DevicePipe& pipe, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            assert(false);
            return py::object();
        }

        template <>
        py::object __extract_array<Tango::DevicePipe, Tango::DEVVAR_DOUBLESTRINGARRAY>
            (Tango::DevicePipe& pipe, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            assert(false);
            return py::object();
        }

        template <>
        py::object
        __extract_array<Tango::DevicePipeBlob, Tango::DEVVAR_LONGSTRINGARRAY>
	(Tango::DevicePipeBlob& blob, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            assert(false);
            return py::object();
        }

        template <>
        py::object __extract_array<Tango::DevicePipeBlob, Tango::DEVVAR_DOUBLESTRINGARRAY>
            (Tango::DevicePipeBlob& blob, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            assert(false);
            return py::object();
        }

        template <long tangoArrayTypeConst>
        py::object extract_array(Tango::DevicePipe& self, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            return __extract_array<Tango::DevicePipe, tangoArrayTypeConst>(self, elt_idx, extract_as);
        }

        template <long tangoArrayTypeConst>
        py::object extract_array(Tango::DevicePipeBlob& self, size_t elt_idx,
                PyTango::ExtractAs extract_as)
        {
            return __extract_array<Tango::DevicePipeBlob, tangoArrayTypeConst>(self, elt_idx, extract_as);
        }

        template<typename T>
        py::object __extract_item(T& obj, size_t elt_idx, PyTango::ExtractAs extract_as)
        {
            const int elt_type = obj.get_data_elt_type(elt_idx);
            TANGO_DO_ON_DEVICE_DATA_TYPE_ID(elt_type,
                    return extract_scalar<tangoTypeConst>(obj, elt_idx); ,
                    return extract_array<tangoTypeConst>(obj, elt_idx, extract_as);
            );
            return py::object();
        }

        template<typename T>
        py::object __extract(T& obj, PyTango::ExtractAs extract_as)
        {
            py::list data;
            size_t elt_nb = obj.get_data_elt_nb();
            for(size_t elt_idx = 0; elt_idx < elt_nb; ++elt_idx)
            {
                py::dict elem;
                elem["name"] = obj.get_data_elt_name(elt_idx);
                elem["dtype"] = static_cast<Tango::CmdArgType>(obj.get_data_elt_type(elt_idx));
                elem["value"] = __extract_item(obj, elt_idx, extract_as);
                data.append(elem);
            }
            return data;
        }
        py::object extract(Tango::DevicePipeBlob& blob,
                PyTango::ExtractAs extract_as)
        {
            py::object name = py::str(blob.get_name());
            py::object value = __extract<Tango::DevicePipeBlob>(blob, extract_as);
            return py::make_tuple(name, value);
        }

        py::object extract(Tango::DevicePipe& device_pipe,
                PyTango::ExtractAs extract_as)
        {
            py::object name = py::str(device_pipe.get_root_blob_name());
            py::object value = __extract<Tango::DevicePipe>(device_pipe, extract_as);
            return py::make_tuple(name, value);
        }
    }
}

void export_device_pipe(py::module &m) {
    py::class_<Tango::DevicePipe>(m, "DevicePipe")
        .def(py::init<>())
        .def(py::init<const std::string &>())
        .def(py::init<const std::string &, const std::string &>())
        .def(py::init<const Tango::DevicePipe &>())
        .def_property("name",
                      &Tango::DevicePipe::get_name,
                      &Tango::DevicePipe::set_name,
                      py::return_value_policy::copy) //  <py::copy_const_reference>()),
        .def_property("root_blob_name",
                      &Tango::DevicePipe::get_root_blob_name,
                      &Tango::DevicePipe::set_root_blob_name,
                      py::return_value_policy::copy) //  <py::copy_const_reference>())
        .def_property("data_elt_nb",
                      &Tango::DevicePipe::get_data_elt_nb, 
                      &Tango::DevicePipe::set_data_elt_nb)
        .def_property("data_elt_names",
                      &Tango::DevicePipe::get_data_elt_names, 
                      &Tango::DevicePipe::set_data_elt_names)
        .def("get_data_elt_name", &Tango::DevicePipe::get_data_elt_name)
        .def("get_data_elt_type", &Tango::DevicePipe::get_data_elt_type)

        .def("extract", [](Tango::DevicePipe& device_pipe, PyTango::ExtractAs extract_as) {
                return PyTango::DevicePipe::extract(device_pipe, extract_as);
        }, py::arg("extract_as")=PyTango::ExtractAsNumpy)

        .def("extract", [](Tango::DevicePipeBlob& blob, PyTango::ExtractAs extract_as) {
                return PyTango::DevicePipe::extract(blob, extract_as);
        }, py::arg("extract_as")=PyTango::ExtractAsNumpy)
    ;
}
