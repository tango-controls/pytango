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
#include "pipe.h"
#include "pyutils.h"
#include "pytgutils.h"
#include "device_impl.h"
#include "../exception.h"
#include "../tgutils.h"

namespace py = pybind11;

namespace PyTango {
namespace Pipe {

void _Pipe::read(Tango::DeviceImpl *dev, Tango::Pipe &pipe)
{
    std::cerr << "In pipe.cpp::_Pipe::Read" << std::endl;
    Device_5ImplWrap* dev_ptr = dynamic_cast<Device_5ImplWrap*>(dev);
    AutoPythonGIL __py_lock;
    if (!is_method_callable(dev_ptr->py_self, read_name))
    {
        std::stringstream o;
        o << read_name << " method " << " not found for " << pipe.get_name();
        Tango::Except::throw_exception("PyTango_ReadPipeMethodNotFound",
                                       o.str(), "PyTango::Pipe::read");
    }
    try {
        dev_ptr->py_self(read_name.c_str())(pipe);
    } catch (py::error_already_set &eas) {
        handle_python_exception(eas);
    }
}

void _Pipe::write(Tango::DeviceImpl* dev, Tango::WPipe &pipe)
{
    Device_5ImplWrap* dev_ptr = dynamic_cast<Device_5ImplWrap*>(dev);
    AutoPythonGIL __py_lock;
    if (!is_method_callable(dev_ptr->py_self, write_name))
    {
        std::stringstream o;
        o << write_name << " method not found for " << pipe.get_name();
        Tango::Except::throw_exception("PyTango_WritePipeMethodNotFound",
               o.str(), "PyTango::Pipe::write");
    }
    try {
        dev_ptr->py_self(write_name.c_str())(pipe);
    } catch(py::error_already_set &eas) {
            handle_python_exception(eas);
    }
}

bool _Pipe::is_allowed(Tango::DeviceImpl *dev, Tango::PipeReqType ty)
{
    Device_5ImplWrap* dev_ptr = dynamic_cast<Device_5ImplWrap*>(dev);
    AutoPythonGIL __py_lock;
    if (is_method_callable(dev_ptr->py_self, py_allowed_name))
    {
        try {
            py::object obj = dev_ptr->py_self.attr(py_allowed_name.c_str())(ty);
            bool ret = obj.cast<bool>();
            return ret;
        } catch (py::error_already_set &eas) {
            handle_python_exception(eas);
        }
    }
    // keep compiler quiet
    return true;
}

static void throw_wrong_python_data_type(const std::string& name, const char *method)
{
    std::stringstream o;
    o << "Wrong Python type for pipe " << name << ends;
    Tango::Except::throw_exception("PyDs_WrongPythonDataTypeForPipe", o.str(), method);
}

template<typename T>
void __append_scalar_encoded(T& obj, const std::string& name, py::object& py_value)
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

template<typename T, typename U>
void __append_scalar(T &obj, const std::string& name, py::object& py_value)
{
    U value = py_value.cast<U>();
    obj << value;
}

template<typename U>
void append_scalar(Tango::Pipe& pipe, const std::string& name, py::object& py_value)
{
    __append_scalar<Tango::Pipe, U>(pipe, name, py_value);
}

template<typename U>
void append_scalar(Tango::DevicePipeBlob& blob, const std::string& name, py::object& py_value)
{
    __append_scalar<Tango::DevicePipeBlob, U>(blob, name, py_value);
}

void append_scalar_encoded(Tango::Pipe& pipe, const std::string& name, py::object& py_value)
{
    __append_scalar_encoded<Tango::Pipe>(pipe, name, py_value);
}

void append_scalar_encoded(Tango::DevicePipeBlob& blob, const std::string& name, py::object& py_value)
{
    __append_scalar_encoded<Tango::DevicePipeBlob>(blob, name, py_value);
}

// -------------
// Array version
// -------------

template<typename T, typename U>
void __append_array(T& obj, const std::string& name, py::object& py_value)
{
    py::list py_list = py_value;
    std::vector<U> values;
    for (auto num : py_list) {
        U value = num.cast<U>();
        values.push_back(value);
    }
    for (auto val : values) {
        std::cout << val << std::endl;
    }
    obj << values;
}

template<typename U>
void append_array(Tango::Pipe& pipe, const std::string& name, py::object& py_value)
{
    __append_array<Tango::Pipe, U>(pipe, name, py_value);
}

template<typename U>
void append_array(Tango::DevicePipeBlob& blob, const std::string& name,
                  py::object& py_value)
{
    __append_array<Tango::DevicePipeBlob, U>(blob, name, py_value);
}

template<typename T>
void __append(T& obj, const std::string& name, py::object& py_value, const Tango::CmdArgType dtype)
{
    // This might be a bit verbose but at least WYSIWYG
    switch (dtype)
    {
    case Tango::DEV_BOOLEAN:
        append_scalar<Tango::DevBoolean>(obj, name, py_value);
        break;
    case Tango::DEV_SHORT:
        append_scalar<Tango::DevShort>(obj, name, py_value);
        break;
    case Tango::DEV_LONG:
        append_scalar<Tango::DevLong>(obj, name, py_value);
        break;
    case Tango::DEV_FLOAT:
        append_scalar<Tango::DevFloat>(obj, name, py_value);
        break;
    case Tango::DEV_DOUBLE:
        append_scalar<Tango::DevDouble>(obj, name, py_value);
        break;
    case Tango::DEV_USHORT:
        append_scalar<Tango::DevUShort>(obj, name, py_value);
        break;
    case Tango::DEV_ULONG:
        append_scalar<Tango::DevULong>(obj, name, py_value);
        break;
    case Tango::DEV_STRING:
        append_scalar<std::string>(obj, name, py_value);
        break;
    case Tango::DEV_STATE:
        append_scalar<Tango::DevState>(obj, name, py_value);
        break;
//    case Tango::CONST_DEV_STRING:
//        append_scalar<Tango::DevString>(obj, name, py_value);
//        break;
    case Tango::DEV_UCHAR:
        append_scalar<Tango::DevUChar>(obj, name, py_value);
        break;
    case Tango::DEV_LONG64:
        append_scalar<Tango::DevLong64>(obj, name, py_value);
        break;
    case Tango::DEV_ULONG64:
        append_scalar<Tango::DevULong64>(obj, name, py_value);
        break;
    case Tango::DEV_ENCODED:
        append_scalar_encoded(obj, name, py_value);
        break;
    case Tango::DEVVAR_CHARARRAY:
        append_array<Tango::DevUChar>(obj, name, py_value);
        break;
    case Tango::DEVVAR_SHORTARRAY:
        append_array<Tango::DevShort>(obj, name, py_value);
        break;
    case Tango::DEVVAR_LONGARRAY:
        append_array<Tango::DevLong>(obj, name, py_value);
        break;
    case Tango::DEVVAR_FLOATARRAY:
        append_array<Tango::DevFloat>(obj, name, py_value);
        break;
    case Tango::DEVVAR_DOUBLEARRAY:
        append_array<Tango::DevDouble>(obj, name, py_value);
        break;
    case Tango::DEVVAR_USHORTARRAY:
        append_array<Tango::DevUShort>(obj, name, py_value);
        break;
    case Tango::DEVVAR_ULONGARRAY:
        append_array<Tango::DevULong>(obj, name, py_value);
        break;
    case Tango::DEVVAR_BOOLEANARRAY:
        append_array<Tango::DevBoolean>(obj, name, py_value);
        break;
    case Tango::DEVVAR_LONG64ARRAY:
        append_array<Tango::DevLong64>(obj, name, py_value);
        break;
    case Tango::DEVVAR_ULONG64ARRAY:
        append_array<Tango::DevULong64>(obj, name, py_value);
        break;
    case Tango::DEVVAR_STATEARRAY:
        append_array<Tango::DevState>(obj, name, py_value);
        break;
    default:
        throw;
    }
}

template<typename T>
void __set_value(T& obj, py::object& py_value)
{
    std::cerr << "__set_value" << std::endl;
    // need to fill item names first because in case it is a sub-blob,
    // the Tango C++ API doesnt't provide a way to do it
    std::vector<std::string> elem_names;
    py::print(py_value);
    for (auto item : py_value) {
        py::print("start ===========");
        py::print(item);
        py::print("end =============");
        elem_names.push_back(item["name"].cast<std::string>());
    }
    obj.set_data_elt_names(elem_names);
    for (auto item : py_value) {
        py::print("start >>>>>>>>>>>");
        py::print(item);
        py::print("end >>>>>>>>>>>>>");
        std::string item_name = item["name"].cast<std::string>();
        py::object item_data = item["value"];
        Tango::CmdArgType item_dtype;
        item_dtype = item["dtype"].cast<Tango::CmdArgType>();
        std::cout << item_dtype << std::endl;
        if (item_dtype == Tango::DEV_PIPE_BLOB) // a sub-blob
        {
            py::print("start ************");
            py::print("item is a pipe blob");
            py::tuple inner_blob = item["value"];
            py::print(inner_blob);
            py::print("end **************");
            std::string inner_blob_name = (inner_blob[0]).cast<std::string>();
            py::print("blob name====", inner_blob_name);
            py::object inner_blob_data = inner_blob[1];
            py::print("blob value===", inner_blob_data);
            Tango::DevicePipeBlob blob(inner_blob_name);
            __set_value(blob, inner_blob_data);
            py::print("----------- blob into obj");
            obj << blob;
        } else {
            py::print("not a pipe blob");
            __append(obj, item_name, item_data, item_dtype);
        }
    }
}

void set_value(Tango::Pipe& pipe, py::object& py_value)
{
    std::cerr << "in pipe set_value" << std::endl;
    __set_value<Tango::Pipe>(pipe, py_value);
}

py::object  get_value(Tango::WPipe& pipe)
{
    std::cerr << "in pipe get_value !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    py::object py_value;

    Tango::DevicePipeBlob blob = pipe.get_blob();
//    py_value = PyTango::DevicePipe::extract(blob);
    return py_value;
}

}} // namespace PyTango::Pipe


namespace PyDevicePipe
{
static void throw_wrong_python_data_type(const std::string& name, const char *method) {
    std::stringstream o;
    o << "Wrong Python type for pipe " << name << ends;
    Tango::Except::throw_exception("PyDs_WrongPythonDataTypeForPipe", o.str(), method);
}

template<typename T, long tangoTypeConst>
void __append_scalar(T &obj, const std::string& name, py::object& py_value) {
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    TangoScalarType value;
//      from_py<tangoTypeConst>::convert(py_value, value);
//      obj << value;
    obj = py_value.cast<TangoScalarType>();
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

        .def("get_name", [](Tango::Pipe& self) -> std::string {
            return self.get_name();
        })
        .def("set_name", [](Tango::Pipe& self, std::string name) -> void {
            self.set_name(name);
        })
        .def("set_default_properties", [](Tango::Pipe& self, Tango::UserDefaultPipeProp& prop) -> void {
            self.set_default_properties(prop);
        })
        .def("get_root_blob_name", [](Tango::Pipe& self) -> std::string {
            return self.get_root_blob_name();
        })
        .def("set_root_blob_name", [](Tango::Pipe& self, std::string& name) -> void {
            self.set_root_blob_name(name);
        })
        .def("get_desc", [](Tango::Pipe& self) -> std::string {
            return self.get_desc();
        })
        .def("get_label", [](Tango::Pipe& self) -> std::string {
            return self.get_label();
        })
        .def("get_disp_level", [](Tango::Pipe& self) -> Tango::DispLevel {
            return self.get_disp_level();
        })
        .def("get_writable", [](Tango::Pipe& self) -> Tango::PipeWriteType {
            return self.get_writable();
        })
        .def("get_pipe_serial_model", [](Tango::Pipe& self) -> Tango::PipeSerialModel {
            return self.get_pipe_serial_model();
        })
        .def("set_pipe_serial_model", [](Tango::Pipe& self, Tango::PipeSerialModel ser_model) void {
            return self.set_pipe_serial_model(ser_model);
        })
        .def("has_failed", [](Tango::Pipe& self) -> bool {
            return self.has_failed();
        })
        .def("_set_value", [](Tango::Pipe& self, py::object&  blob) -> void {
            PyTango::Pipe::set_value(self, blob);
        })
        .def("get_value", [](Tango::WPipe& self) -> py::object {
             return PyTango::Pipe::get_value(self);
        })
        ;

    py::class_<Tango::WPipe, Tango::Pipe>(m, "WPipe")
        .def(py::init<const std::string& , const Tango::DispLevel>())
    ;
}
