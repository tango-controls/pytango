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
#include "../device_pipe.h"

namespace py = pybind11;

namespace PyTango {
namespace Pipe {

void _Pipe::read(Tango::DeviceImpl *dev, Tango::Pipe &pipe)
{
    DeviceImplWrap* dev_ptr = dynamic_cast<DeviceImplWrap*>(dev);
    AutoPythonGILEnsure __py_lock;
    if (!is_method_callable(dev_ptr->py_self, read_name))
    {
        std::stringstream o;
        o << read_name << " method " << " not found for " << pipe.get_name();
        Tango::Except::throw_exception("PyTango_ReadPipeMethodNotFound",
                                       o.str(), "PyTango::Pipe::read");
    }
    try {
        py::object obj = dev_ptr->py_self.attr(read_name.c_str())();
        py::tuple tup(obj);
        std::string root_blob_name = tup[0].cast<std::string>();
        pipe.set_root_blob_name(root_blob_name);
        py::object items(tup[1]);
        set_value(pipe, items);
    }
    catch (py::error_already_set &eas) {
        handle_python_exception(eas);
    }
}

void _Pipe::write(Tango::DeviceImpl* dev, Tango::WPipe &pipe)
{
    DeviceImplWrap* dev_ptr = dynamic_cast<DeviceImplWrap*>(dev);
    AutoPythonGILEnsure __py_lock;
    if (!is_method_callable(dev_ptr->py_self, write_name))
    {
        std::stringstream o;
        o << write_name << " method not found for " << pipe.get_name();
        Tango::Except::throw_exception("PyTango_WritePipeMethodNotFound",
               o.str(), "PyTango::Pipe::write");
    }
    try {
        py::print("before get_pipe_write_value");
        py::object value = get_pipe_write_value(pipe);
        py::print("after get_pipe_write_value");
        dev_ptr->py_self.attr(write_name.c_str())(value);
    }
    catch(py::error_already_set &eas) {
        handle_python_exception(eas);
    }
}

bool _Pipe::is_allowed(Tango::DeviceImpl *dev, Tango::PipeReqType ty)
{
    DeviceImplWrap* dev_ptr = dynamic_cast<DeviceImplWrap*>(dev);
    AutoPythonGILEnsure __py_lock;
    if (is_method_callable(dev_ptr->py_self, py_allowed_name))
    {
        try {
            py::object obj = dev_ptr->py_self.attr(py_allowed_name.c_str())(ty);
            return obj.cast<bool>();
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
void append_scalar_encoded(T& obj, const std::string& name, py::object& py_value)
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
    obj << value;
}

template<typename T, long tangoTypeConst>
void __append_scalar(T& obj, const std::string& name, py::object& py_value)
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    TangoScalarType value = py_value.cast<TangoScalarType>();
    obj << value;
}

template<typename T>
void __append_scalar_string(T& obj, const std::string& name, py::object& py_value)
{
    std::string value = py_value.cast<std::string>();
    obj << value;
}

template<long tangoTypeConst>
void append_scalar(Tango::Pipe& pipe, const std::string& name,
          py::object& py_value)
{
    __append_scalar<Tango::Pipe, tangoTypeConst>(pipe, name, py_value);
}

template<>
void append_scalar<Tango::DEV_STRING>(Tango::Pipe& pipe, const std::string& name,
          py::object& py_value)
{
    __append_scalar_string<Tango::Pipe>(pipe, name, py_value);
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
void append_scalar<Tango::DEV_STRING>(Tango::DevicePipeBlob& blob, const std::string &name,
          py::object& py_value)
{
    __append_scalar_string<Tango::DevicePipeBlob>(blob, name, py_value);
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
                   const std::string &name,
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
    TangoArrayType* array = new TangoArrayType(len, len, data_buffer, true);
    obj << array;
}

template<typename T>
void __append_array_string(T& obj, const std::string& name, py::object& py_value)
{
    py::list py_list = py_value;
    std::vector<std::string> values;
    for (auto& item : py_list) {
        std::string value = item.cast<std::string>();
        values.push_back(value);
    }
    obj << values;
}

template<long tangoArrayTypeConst>
void append_array(Tango::Pipe& pipe, const std::string& name, py::object& py_value)
{
    __append_array<Tango::Pipe, tangoArrayTypeConst>(pipe, name, py_value);
}

template<>
void append_array<Tango::DEVVAR_STRINGARRAY>(Tango::Pipe& pipe, const std::string& name, py::object& py_value)
{
    __append_array_string<Tango::Pipe>(pipe, name, py_value);
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
void append_array<Tango::DEVVAR_STRINGARRAY>(Tango::DevicePipeBlob& blob, const std::string& name, py::object& py_value)
{
    __append_array_string<Tango::DevicePipeBlob>(blob, name, py_value);
}

template<>
void append_array<Tango::DEV_VOID>(Tango::DevicePipeBlob& blob,
                   const std::string &name,
                   py::object& py_value)
{
    throw_wrong_python_data_type(blob.get_name(), "append_array");
}

template<>
void append_array<Tango::DEV_PIPE_BLOB>(Tango::DevicePipeBlob& blob,
                    const std::string &name,
                    py::object& py_value)
{
    throw_wrong_python_data_type(blob.get_name(), "append_array");
}

template<>
void append_array<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::DevicePipeBlob& blob,
                         const std::string &name,
                         py::object& py_value)
{
    throw_wrong_python_data_type(blob.get_name(), "append_array");
}

template<>
void append_array<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::DevicePipeBlob& blob,
                           const std::string &name,
                           py::object& py_value)
{
    throw_wrong_python_data_type(blob.get_name(), "append_array");
}

template<typename T>
void __append(T& obj, const std::string& name,
    py::object& py_value, const Tango::CmdArgType dtype)
{
    TANGO_DO_ON_DEVICE_DATA_TYPE_ID(dtype,
        append_scalar<tangoTypeConst>(obj, name, py_value);
    ,
        append_array<tangoTypeConst>(obj, name, py_value);
    );
}

template<typename T>
void __set_value(T& obj, py::object& py_value)
{
    // need to fill item names first because in case it is a sub-blob,
    // the Tango C++ API doesnt't provide a way to do it
    std::vector<std::string> elem_names;
    for (auto& item : py_value) {
        elem_names.push_back(item["name"].cast<std::string>());
    }
    obj.set_data_elt_names(elem_names);
    for (auto& item : py_value) {
        std::string item_name = item["name"].cast<std::string>();
        py::object item_data = item["value"];
        Tango::CmdArgType item_dtype;
        item_dtype = item["dtype"].cast<Tango::CmdArgType>();
        if (item_dtype == Tango::DEV_PIPE_BLOB) // a sub-blob
        {
            py::tuple inner_blob = item["value"];
            std::string inner_blob_name = (inner_blob[0]).cast<std::string>();
            py::object inner_blob_data = inner_blob[1];
            Tango::DevicePipeBlob blob(inner_blob_name);
            __set_value(blob, inner_blob_data);
            obj << blob;
        } else {
            __append(obj, item_name, item_data, item_dtype);
        }
    }
}

void set_value(Tango::Pipe& pipe, py::object& py_value)
{
    __set_value<Tango::Pipe>(pipe, py_value);
}

void set_value(Tango::DevicePipeBlob& dpb, py::object& py_data) {
    py::tuple tup(py_data);
    std::string name = tup[0].cast<std::string>();
    dpb.set_name(name);
    py::object items(tup[1]);
    __set_value<Tango::DevicePipeBlob>(dpb, items);
}

py::object get_pipe_write_value(Tango::WPipe& pipe)
{
    py::print("in get_pipe_write_value");
    Tango::DevicePipeBlob blob = pipe.get_blob();
    py::print("before extract blob");
    py::object ret = PyTango::DevicePipe::extract(blob);
    py::print("after extract blob");
    return ret;
}

}} // namespace PyTango::Pipe

void export_pipe(py::module &m)
{
    py::class_<Tango::Pipe>(m, "Pipe")
        .def(py::init())
        .def(py::init([](const std::string& name, const Tango::DispLevel level, Tango::PipeWriteType wtype) {
            return new Tango::Pipe(name, level, wtype);
        })) //, py::arg("name"), py::arg("level"), py::arg("wtype")=Tango::PIPE_READ))

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
    ;
    py::class_<Tango::WPipe, Tango::Pipe>(m, "WPipe")
        .def(py::init<const std::string& , const Tango::DispLevel>())
    ;
}
