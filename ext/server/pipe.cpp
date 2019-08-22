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
#include "pipe.h"
#include "pytgutils.h"
#include "device_impl.h"
#include "../exception.h"

namespace py = pybind11;

// THIS IS THE SAME CODE AS IN attr.cpp except it says PIPE not ATTR
// WE should move it to a common util file!!!!!!!!!!!!!!!!!!!!!!!!!!

//static inline void CALL_PIPE_METHOD(Tango::DeviceImpl *dev, const std::string& name) {
//    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
//    AutoPythonGIL __py_lock;
//    try {
//        py::call_method<void>(__dev_ptr->the_self, name);
//    } catch (py::error_already_set &eas) {
//        handle_python_exception(eas);
//    }
//}
//
//static inline void CALL_PIPE_METHOD_VARGS(Tango::DeviceImpl *dev, const std::string& name, ...) {
//    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
//    AutoPythonGIL __py_lock;
//    try {
//        py::call_method<void>(__dev_ptr->the_self, name, ...);
//    } catch (py::error_already_set &eas) {
//        handle_python_exception(eas);
//    }
//}
//
//static inline CALL_PIPE_METHOD_RET(retType, ret, Tango::DeviceImpl *dev, const std::string& name) {
//    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
//    AutoPythonGIL __py_lock;
//    try {
//        ret = py::call_method<retType>(__dev_ptr->the_self, name);
//    } catch (py::error_already_set &eas) {
//        handle_python_exception(eas);
//    }
//}
//
//static inline CALL_PIPE_METHOD_VARGS_RET(retType, ret, Tango::DeviceImpl *dev, const std::string& name, ...) {
//    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
//    AutoPythonGIL __py_lock;
//    try {
//        ret = py::call_method<retType>(__dev_ptr->the_self, name, ...);
//    } catch (py::error_already_set &eas) {
//        handle_python_exception(eas);
//    }
//}
//
//static inline RET_CALL_PIPE_METHOD(retType, Tango::DeviceImpl *dev, const std::string& name) {
//    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
//    AutoPythonGIL __py_lock;
//    try {
//        return py::call_method<retType>(__dev_ptr->the_self, name);
//    } catch (py::error_already_set &eas) {
//        handle_python_exception(eas);
//    }
//}
//
//static inline RET_CALL_PIPE_METHOD_VARGS(retType, Tango::DeviceImpl *dev, const std::string& name, ...) {
//    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
//    AutoPythonGIL __py_lock;
//    try {
//        return py::call_method<retType>(__dev_ptr->the_self, name, ...);
//    } catch (py::error_already_set &eas) {
//        handle_python_exception(eas);
//    }
//}
namespace PyTango { namespace Pipe {


    void _Pipe::read(Tango::DeviceImpl *dev, Tango::Pipe &pipe)
    {
//        if (!_is_method(dev, read_name))
//        {
//            std::stringstream o;
//            o << read_name << " method " << " not found for " << pipe.get_name();
//            Tango::Except::throw_exception("PyTango_ReadPipeMethodNotFound",
//                                           o.str(), "PyTango::Pipe::read");
//        }
//        CALL_PIPE_METHOD_VARGS(dev, read_name.c_str(), boost::ref(pipe))
    }

    void _Pipe::write(Tango::DeviceImpl* dev, Tango::WPipe &pipe)
    {
//        if (!_is_method(dev, write_name))
//        {
//            std::stringstream o;
//            o << write_name << " method not found for " << pipe.get_name();
//            Tango::Except::throw_exception("PyTango_WritePipeMethodNotFound",
//                   o.str(), "PyTango::Pipe::write");
//        }
//        PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
//        AutoPythonGIL __py_lock;
//        try {
////            py::call_method<py::object>(__dev_ptr->the_self, write_name.c_str(), boost::ref(pipe));
//        } catch(py::error_already_set &eas) {
//            handle_python_exception(eas);
//        }
    }

    bool _Pipe::is_allowed(Tango::DeviceImpl *dev, Tango::PipeReqType ty)
    {
//        if (_is_method(dev, py_allowed_name))
//        {
//            RET_CALL_PIPE_METHOD_VARGS(bool, dev, py_allowed_name.c_str(), ty)
//        }
        // keep compiler quiet
        return true;
    }

    bool _Pipe::_is_method(Tango::DeviceImpl *dev, const std::string& name)
    {
//        AutoPythonGIL __py_lock;
//        PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
//        PyObject *__dev_py = __dev_ptr->the_self;
//        return is_method_defined(__dev_py, name);
        return true;
    }

    static void throw_wrong_python_data_type(const std::string& name, const char *method)
    {
        std::stringstream o;
        o << "Wrong Python type for pipe " << name << ends;
        Tango::Except::throw_exception("PyDs_WrongPythonDataTypeForPipe", o.str(), method);
    }

template<typename T>
void append_scalar_encoded(T& obj, const std::string& name,
                           py::object& py_value)
    {
//        py::object p0 = py_value[0];
//        py::object p1 = py_value[1];
//
//        const char* encoded_format = py::extract<const char *> (p0.ptr());
//
//        PyObject* data_ptr = p1.ptr();
//        Py_buffer view;
//
//        if (PyObject_GetBuffer(data_ptr, &view, PyBUF_FULL_RO) < 0)
//        {
//            throw_wrong_python_data_type(obj.get_name(), "append_scalar_encoded");
//        }
//
//        CORBA::ULong nb = static_cast<CORBA::ULong>(view.len);
//        Tango::DevVarCharArray arr(nb, nb, (CORBA::Octet*)view.buf, false);
//        Tango::DevEncoded value;
//        value.encoded_format = CORBA::string_dup(encoded_format);
//        value.encoded_data = arr;
//        obj << value;
//        PyBuffer_Release(&view);
    }

    template<typename T, long tangoTypeConst>
    void __append_scalar(T &obj, const std::string& name, py::object& py_value)
    {
//        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
//        TangoScalarType value;
//        from_py<tangoTypeConst>::convert(py_value, value);
//        obj << value;
    }

    template<long tangoTypeConst>
    void append_scalar(Tango::Pipe& pipe, const std::string& name,
                       py::object& py_value)
    {
//        __append_scalar<Tango::Pipe, tangoTypeConst>(pipe, name, py_value);
    }

    template<>
    void append_scalar<Tango::DEV_VOID>(Tango::Pipe& pipe,
                                        const std::string& name,
                                        py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_scalar");
    }

    template<>
    void append_scalar<Tango::DEV_PIPE_BLOB>(Tango::Pipe& pipe,
                                             const std::string& name,
                                             py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_scalar");
    }

    template<>
    void append_scalar<Tango::DEV_ENCODED>(Tango::Pipe& pipe,
                                           const std::string& name,
                                           py::object& py_value)
    {
        append_scalar_encoded<Tango::Pipe>(pipe, name, py_value);
    }

    template<long tangoTypeConst>
    void append_scalar(Tango::DevicePipeBlob& blob, const std::string& name, py::object& py_value)
    {
        __append_scalar<Tango::DevicePipeBlob, tangoTypeConst>(blob, name, py_value);
    }

    template<>
    void append_scalar<Tango::DEV_VOID>(Tango::DevicePipeBlob& blob,
                                        const std::string& name,
                                        py::object& py_value)
    {
        throw_wrong_python_data_type(blob.get_name(), "append_scalar");
    }

    template<>
    void append_scalar<Tango::DEV_PIPE_BLOB>(Tango::DevicePipeBlob& blob,
                                             const std::string& name,
                                             py::object& py_value)
    {
        throw_wrong_python_data_type(blob.get_name(), "append_scalar");
    }

    template<>
    void append_scalar<Tango::DEV_ENCODED>(Tango::DevicePipeBlob& blob,
                                           const std::string& name,
                                           py::object& py_value)
    {
        append_scalar_encoded<Tango::DevicePipeBlob>(blob, name, py_value);
    }

    // -------------
    // Array version
    // -------------

    template<typename T, long tangoArrayTypeConst>
    void __append_array(T& obj, const std::string& name, py::object& py_value)
    {
//        typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
//        TangoArrayType* value = fast_convert2array<tangoArrayTypeConst>(py_value);
//        obj << value;
    }

    template<long tangoArrayTypeConst>
    void append_array(Tango::Pipe& pipe, const std::string& name,
                      py::object& py_value)
    {
        __append_array<Tango::Pipe, tangoArrayTypeConst>(pipe, name, py_value);
    }

    template<>
    void append_array<Tango::DEV_VOID>(Tango::Pipe& pipe,
                                       const std::string& name,
                                       py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_array");
    }

    template<>
    void append_array<Tango::DEV_PIPE_BLOB>(Tango::Pipe& pipe,
                                            const std::string& name,
                                            py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_array");
    }

    template<>
    void append_array<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::Pipe& pipe,
                                                     const std::string& name,
                                                     py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_array");
    }

    template<>
    void append_array<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::Pipe& pipe,
                                                       const std::string& name,
                                                       py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_array");
    }

    template<long tangoArrayTypeConst>
    void append_array(Tango::DevicePipeBlob& blob, const std::string& name,
                      py::object& py_value)
    {
        __append_array<Tango::DevicePipeBlob, tangoArrayTypeConst>(blob, name, py_value);
    }

    template<>
    void append_array<Tango::DEV_VOID>(Tango::DevicePipeBlob& blob,
                                       const std::string& name,
                                       py::object& py_value)
    {
        throw_wrong_python_data_type(blob.get_name(), "append_array");
    }

    template<>
    void append_array<Tango::DEV_PIPE_BLOB>(Tango::DevicePipeBlob& blob,
                                            const std::string& name,
                                            py::object& py_value)
    {
        throw_wrong_python_data_type(blob.get_name(), "append_array");
    }

    template<>
    void append_array<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::DevicePipeBlob& blob,
                                                     const std::string& name,
                                                     py::object& py_value)
    {
        throw_wrong_python_data_type(blob.get_name(), "append_array");
    }

    template<>
    void append_array<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::DevicePipeBlob& blob,
                                                       const std::string& name,
                                                       py::object& py_value)
    {
        throw_wrong_python_data_type(blob.get_name(), "append_array");
    }

//  );
//    }


//    template<typename T>
//    void __set_value(T& obj, py::object& py_value)
//    {
//        py::object items = py_value.attr("items")();
//        Py_ssize_t size = py::len(items);
//        obj.set_data_elt_nb(size);
//        for(size_t i = 0; i < size; ++i)
//        {
//            std::string item_name = py::extract<std::string>(items[i][0]);
//            py::object py_item_data = items[i][1];
//            if (PyDict_Check(py_item_data.ptr()))  // data element
//            {
//                py::object py_item_value = py_item_data["value"];
//                py::object py_item_dtype = py_item_data["dtype"];
//                Tango::CmdArgType item_dtype = py::extract<Tango::CmdArgType>(py_item_dtype);
//                __append(obj, item_name, py_item_value, item_dtype);
//            }
//            else
//            {
//                std::string blob_name = py::extract<std::string>(py_item_data[0]);
//                py::object py_blob_data = py_item_data[1];
//                Tango::DevicePipeBlob blob(blob_name);
//                __set_value(blob, py_blob_data);
//                obj << blob;
//            }
//        }
//    }


//    template<typename T>
//    void __set_value(T& obj, py::object& py_value)
//    {
//        py::str name_key("name");
//        Py_ssize_t size = py::len(py_value);
//        std::vector<std::string> elem_names;
//        for(size_t i = 0; i < size; ++i)
//        {
//            elem_names.push_back(py::extract<std::string>(py_value[i][0]));
//        }
//        obj.set_data_elt_names(elem_names);
//
//        for(size_t i = 0; i < size; ++i)
//        {
//            std::string item_name = py::extract<std::string>(py_value[i][0]);
//            py::dict py_item_data = py::extract<py::dict>(py_value[i][1]);
//            if (py_item_data.has_key(name_key)) // a sub-blob
//            {
//                std::string blob_name = py::extract<std::string>(py_item_data["name"]);
//                py::object py_blob_data = py_item_data["data"];
//                Tango::DevicePipeBlob blob(blob_name);
//                __set_value(blob, py_blob_data);
//                obj << blob;
//            }
//            else
//            {
//                py::object py_item_value = py_item_data["value"];
//                py::object py_item_dtype = py_item_data["dtype"];
//                Tango::CmdArgType item_dtype = py::extract<Tango::CmdArgType>(py_item_dtype);
//                __append(obj, item_name, py_item_value, item_dtype);
//            }
//        }
//    }


    template<typename T>
    void __set_value(T& obj, py::object& py_value)
    {
//        // need to fill item names first because in case it is a sub-blob,
//        // the Tango C++ API doesnt't provide a way to do it
//        Py_ssize_t size = py::len(py_value);
//        std::vector<std::string> elem_names;
//        for(ssize_t i = 0; i < size; ++i)
//        {
//            elem_names.push_back(py::extract<std::string>(py_value[i]["name"]));
//        }
//        obj.set_data_elt_names(elem_names);
//
//        for(ssize_t i = 0; i < size; ++i)
//        {
//            py::object item = py_value[i];
//            std::string item_name = py::extract<std::string>(item["name"]);
//            py::object py_item_data = item["value"];
//            Tango::CmdArgType item_dtype = py::extract<Tango::CmdArgType>(item["dtype"]);
//            if (item_dtype == Tango::DEV_PIPE_BLOB) // a sub-blob
//            {
//                std::string blob_name = py::extract<std::string>(py_item_data[0]);
//                py::object py_blob_data = py_item_data[1];
//                Tango::DevicePipeBlob blob(blob_name);
//                __set_value(blob, py_blob_data);
//                obj << blob;
//            }
//            else
//            {
//                __append(obj, item_name, py_item_data, item_dtype);
//            }
//        }
    }

    void set_value(Tango::Pipe& pipe, py::object& py_value)
    {
//        __set_value<Tango::Pipe>(pipe, py_value);
    }

    py::object  get_value(Tango::WPipe& pipe)
    {
//        py::object py_value;
//
//        Tango::DevicePipeBlob blob = pipe.get_blob();
//        py_value = PyTango::DevicePipe::extract(blob);
//        return py_value;
    }

}} // namespace PyTango::Pipe


namespace PyDevicePipe
{
  static void throw_wrong_python_data_type(const std::string& name, const char *method) {
      std::stringstream o;
      o << "Wrong Python type for pipe " << name << ends;
      Tango::Except::throw_exception("PyDs_WrongPythonDataTypeForPipe",
              o.str(), method);
  }

  template<typename T, long tangoTypeConst>
  void __append_scalar(T &obj, const std::string& name, py::object& py_value) {
//      typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
//      TangoScalarType value;
//      from_py<tangoTypeConst>::convert(py_value, value);
//      obj << value;
  }

  template<typename T, long tangoArrayTypeConst>
  void __append_array(T& obj, const std::string& name, py::object& py_value) {
//      typedef typename TANGO_const2type(tangoArrayTypeConst) TangoArrayType;
//
//      TangoArrayType* value = fast_convert2array<tangoArrayTypeConst>(py_value);
//      obj << value;
  }

  template<typename T>
  bool __check_type(const py::object& value) {
//      py::extract<T> item(value);
//      return item.check();
      return true;
  }

  template<typename T>
  bool __convert(const py::object& value, T& py_item_data) {
//      py::extract<T> item(value);
//      if (item.check()) {
//          py_item_data = item();
//          return true;
//      }
      return false;
  }

  void __append(Tango::DevicePipeBlob& dpb, const std::string& name, py::object& value) {
//      if (__check_type<string>(value)) {
//          __append_scalar<Tango::DevicePipeBlob, Tango::DEV_STRING>(dpb, name, value);
//      } else if (__check_type<int>(value)) {
//          __append_scalar<Tango::DevicePipeBlob, Tango::DEV_LONG64>(dpb, name, value);
//      } else if (__check_type<double>(value)) {
//          __append_scalar<Tango::DevicePipeBlob, Tango::DEV_DOUBLE>(dpb, name, value);
//      } else if (__check_type<bool>(value)) {
//          __append_scalar<Tango::DevicePipeBlob, Tango::DEV_BOOLEAN>(dpb, name, value);
//      } else if (__check_type<py::list>(value)) {
//          if (__check_type<string>(value[0])) {
//              __append_array<Tango::DevicePipeBlob, Tango::DEVVAR_STRINGARRAY>(dpb, name, value);
//          } else if (__check_type<int>(value[0])) {
//              __append_array<Tango::DevicePipeBlob, Tango::DEVVAR_LONG64ARRAY>(dpb, name, value);
//          } else if (__check_type<double>(value[0])) {
//              __append_array<Tango::DevicePipeBlob, Tango::DEVVAR_DOUBLEARRAY>(dpb, name, value);
//          } else {
//              throw_wrong_python_data_type(name, "__append");
//          }
//      } else {
//          throw_wrong_python_data_type(name, "__append");
//      }
  }

  void __set_value(Tango::DevicePipeBlob& dpb, py::dict& dict) {
//      int nitems = len(dict);
//      std::vector<std::string> elem_names;
//      for (auto i=0; i<nitems; i++) {
//          elem_names.push_back(py::extract<std::string>(dict.keys()[i]));
//      }
//      dpb.set_data_elt_names(elem_names);
//
//      py::list values = dict.values();
//      for (auto i=0; i <nitems; ++i) {
//          py::object item = values[i];
//          // Check if the value is an inner blob
//          py::tuple ptuple;
//          std::string blob_name;
//          py::dict pdict;
//          if (__convert(item, ptuple) && __convert(ptuple[0], blob_name)
//              && __convert(ptuple[1], pdict)) {
//              Tango::DevicePipeBlob inner_blob(blob_name);
//              __set_value(inner_blob, pdict);
//              dpb << inner_blob;
//          } else {
//              __append(dpb, elem_names[i], item);
//          }
//      }
  }

  void set_value(Tango::DevicePipeBlob& dpb, py::object& py_data) {
//      std::string name = py::extract<std::string>(py_data[0]);
//      dpb.set_name(name);
//
//      py::dict data = py::extract<py::dict>(py_data[1]);
//      __set_value(dpb, data);
  }
} // namespace PyDevicePipe

void export_pipe(py::module &m)
{
    py::class_<Tango::Pipe>(m, "Pipe")
//        .def(py::init<const std::string& , const Tango::DispLevel,
//                   py::optional<Tango::PipeWriteType> >())

        .def("get_name", [] (Tango::Pipe& self) -> std::string {
            return self.get_name();
        })
        .def("set_name", [] (Tango::Pipe& self, std::string name) -> void {
            self.set_name(name);
        })
        .def("set_default_properties", [] (Tango::Pipe& self, Tango::UserDefaultPipeProp& prop) -> void {
            self.set_default_properties(prop);
        })
        .def("get_root_blob_name", &Tango::Pipe::get_root_blob_name,
            py::return_value_policy::copy)
        .def("set_root_blob_name", &Tango::Pipe::set_root_blob_name)
        .def("get_desc", &Tango::Pipe::get_desc,
            py::return_value_policy::copy)
        .def("get_label", &Tango::Pipe::get_label,
            py::return_value_policy::copy)
        .def("get_disp_level", &Tango::Pipe::get_disp_level)
        .def("get_writable", &Tango::Pipe::get_writable)
        .def("get_pipe_serial_model", &Tango::Pipe::get_pipe_serial_model)
        .def("set_pipe_serial_model", &Tango::Pipe::set_pipe_serial_model)
        .def("has_failed", &Tango::Pipe::has_failed)

        .def("_set_value", (void (*) (Tango::Pipe &, py::object &))
             &PyTango::Pipe::set_value)

        .def("get_value", (py::object (*) (Tango::WPipe &))
             &PyTango::Pipe::get_value)
        ;

    py::class_<Tango::WPipe, Tango::Pipe>(m, "WPipe")
        .def(py::init<const std::string& , const Tango::DispLevel>())
    ;
}
