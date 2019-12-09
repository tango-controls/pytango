/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <memory>
#include <tango.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <callback.h>
#include <device_attribute.h>
#include <from_py.h>

namespace py = pybind11;

namespace PyDeviceProxy
{
//    struct PickleSuite : py::pickle_suite
//    {
//        static py::tuple getinitargs(Tango::DeviceProxy& self)
//        {
//            std::string ret = self.get_db_host() + ":" + self.get_db_port() +
//                              "/" + self.dev_name();
//            return py::make_tuple(ret);
//        }
//    };

    static inline void pylist_to_devattrs(Tango::DeviceProxy& self,
        py::object& py_list, std::vector<Tango::DeviceAttribute> &dev_attrs)
    {
        for (auto item : py_list) {
            py::tuple tup = py::reinterpret_borrow<py::tuple>(item);
            std::string name = tup[0].cast<std::string>();
            py::object value = tup[1];
            Tango::DeviceAttribute w_dev_attr;
            PyDeviceAttribute::reset(w_dev_attr, name, self, value);
            dev_attrs.push_back(w_dev_attr);
        }
    }

    template<typename ED, typename EDList>
    static py::object
    get_events__aux(Tango::DeviceProxy& self, int event_id)
    {
        EDList event_list;
        // C++ signature
        self.get_events(event_id, event_list);
        py::list r;
        for (size_t i=0; i < event_list.size(); ++i) {
            ED* event_data = event_list[i];
            py::object py_ev = py::cast(event_data);
            CallBackPushEvent::fill_py_event(event_data, py_ev, self);
            r.append(py_ev);
        }
        return r;
    }

    // This code also appears in pipe.cpp
    static void throw_wrong_python_data_type(const std::string& name,
                 const char *method)
    {
        std::stringstream ss;
        ss << "Wrong Python type for pipe " << name << ends;
//gm        Tango::Except::throw_exception("PyDs_WrongPythonDataTypeForPipe", ss.str(), method);
    }


    template<typename T>
    void append_scalar_encoded(T& obj, const std::string& name,
                       py::object& py_value)
    {
        py::tuple tup = py_value;
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
    void __append_scalar(T &obj, const std::string& name, py::object& py_value)
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
    void append_scalar(Tango::DevicePipe& pipe, const std::string& name,
                   py::object& py_value)
    {
        __append_scalar<Tango::DevicePipe, tangoTypeConst>(pipe, name, py_value);
    }

    template<>
    void append_scalar<Tango::DEV_STRING>(Tango::DevicePipe& pipe, const std::string& name,
              py::object& py_value)
    {
        __append_scalar_string<Tango::DevicePipe>(pipe, name, py_value);
    }

    template<>
    void append_scalar<Tango::DEV_VOID>(Tango::DevicePipe& pipe,
                        const std::string& name, py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_scalar");
    }

    template<>
    void append_scalar<Tango::DEV_PIPE_BLOB>(Tango::DevicePipe& pipe,
                             const std::string& name,
                             py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_scalar");
    }

    template<>
    void append_scalar<Tango::DEV_ENCODED>(Tango::DevicePipe& pipe,
                           const std::string& name,
                           py::object& py_value)
    {
        append_scalar_encoded<Tango::DevicePipe>(pipe, name, py_value);
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
    void append_array(Tango::DevicePipe& pipe, const std::string& name,
                  py::object& py_value)
    {
        __append_array<Tango::DevicePipe, tangoArrayTypeConst>(pipe, name, py_value);
    }

    template<>
    void append_array<Tango::DEVVAR_STRINGARRAY>(Tango::DevicePipe& pipe,
                           const std::string& name,
                           py::object& py_value)
    {
        __append_array_string<Tango::DevicePipe>(pipe, name, py_value);
    }

    template<>
    void append_array<Tango::DEV_VOID>(Tango::DevicePipe& pipe,
                       const std::string& name,
                       py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_array");
    }

    template<>
    void append_array<Tango::DEV_PIPE_BLOB>(Tango::DevicePipe& pipe,
                            const std::string& name,
                            py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_array");
    }

    template<>
    void append_array<Tango::DEVVAR_LONGSTRINGARRAY>(Tango::DevicePipe& pipe,
                                 const std::string& name,
                                 py::object& py_value)
    {
        throw_wrong_python_data_type(pipe.get_name(), "append_array");
    }

    template<>
    void append_array<Tango::DEVVAR_DOUBLESTRINGARRAY>(Tango::DevicePipe& pipe,
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
    void append_array<Tango::DEVVAR_STRINGARRAY>(Tango::DevicePipeBlob& blob,
                           const std::string& name,
                           py::object& py_value)
    {
        __append_array_string<Tango::DevicePipeBlob>(blob, name, py_value);
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
        for (auto item : py_value) {
            elem_names.push_back(item["name"].cast<std::string>());
        }
        obj.set_data_elt_names(elem_names);
        for (auto item : py_value) {
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

    void set_value(Tango::DevicePipe& pipe, py::object& py_value)
    {
        __set_value<Tango::DevicePipe>(pipe, py_value);
    }

    void set_value(Tango::DevicePipeBlob& blob, py::object& value)
    {
        __set_value<Tango::DevicePipeBlob>(blob, value);
    }

    template<typename T>
    bool check_cast(py::handle obj)
    {
        try {
            obj.cast<T>();
            return true;
        } catch (py::cast_error &e) {
            return false;
        } catch (py::error_already_set &e) {
            return false;
        }
    }
};

void export_device_proxy(py::module &m) {

    py::class_<Tango::DeviceProxy>(m, "DeviceProxy", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<const Tango::DeviceProxy &>())
        .def(py::init([](std::string& name) {
            Tango::DeviceProxy* dp = nullptr;
            AutoPythonAllowThreads guard;
            // C++ signature
            return new Tango::DeviceProxy(name); // C++ signature
        }))

        .def(py::init([](std::string& name, bool ch_access) {
            Tango::DeviceProxy* dp = nullptr;
            AutoPythonAllowThreads guard;
            return new Tango::DeviceProxy(name, ch_access); // C++ signature
        }))
        //
        // Pickle
        //
//        .def_pickle(PyDeviceProxy::PickleSuite())

        //
        // general methods
        //
        .def("dev_name", [](Tango::DeviceProxy& self) -> std::string {
            return self.dev_name(); // Tango C++ signature
        })
        .def("info", [](Tango::DeviceProxy& self) -> Tango::DeviceInfo {
            return self.info(); // Tango C++ signature
        })
        .def("get_device_db", [](Tango::DeviceProxy& self) -> Tango::Database* {
            return self.get_device_db(); // Tango C++ signature
        })
        .def("_status", [](Tango::DeviceProxy& self) -> std::string {
            AutoPythonAllowThreads guard;
            return self.status(); // Tango C++ signature
        })
       .def("_state", [](Tango::DeviceProxy& self) -> Tango::DevState {
            AutoPythonAllowThreads guard;
            return self.state(); // Tango C++ signature
        })
        .def("adm_name", [](Tango::DeviceProxy& self) -> std::string {
            return self.adm_name(); // Tango C++ signature
        })
        .def("description", [](Tango::DeviceProxy& self) -> std::string {
            return self.description(); // Tango C++ signature
        })
        .def("name", [](Tango::DeviceProxy& self) -> std::string {
            return self.name(); // Tango C++ signature
        })
        .def("alias", [](Tango::DeviceProxy& self) -> std::string {
            py::print("in pybind c++ alias method");
            std::string alias;
            try {
                alias = self.alias(); // Tango C++ signature
            } catch (exception& e) {
                py::print("got the exception");
            }
            return alias;
        })
        .def("get_tango_lib_version", [](Tango::DeviceProxy& self) -> int {
            py::print("in pybind get_tango_lib_version");
            int version = self.get_tango_lib_version();
            std::cerr << version << std::endl;
            return version;
//            return self.get_tango_lib_version(); // Tango C++ signature
        })
        .def("_ping", [](Tango::DeviceProxy& self) -> int {
            AutoPythonAllowThreads guard;
            return self.ping(); // Tango C++ signature
        })
        .def("black_box", [](Tango::DeviceProxy& self, int nb) -> std::vector<std::string>* {
            return std::move(self.black_box(nb)); // Tango C++ signature
        })
        //
        // command methods
        //
        .def("get_command_list", [](Tango::DeviceProxy& self) -> std::vector<std::string>* {
            return std::move(self.get_command_list()); // Tango C++ signature
        })
        .def("_get_command_config", [](Tango::DeviceProxy& self, const std::string& cmd_name) -> Tango::CommandInfo {
            return self.get_command_config(cmd_name); // Tango C++ signature
        })
        .def("_get_command_config", [](Tango::DeviceProxy& self, std::vector<std::string>& cmd_names) -> Tango::CommandInfoList* {
            return self.get_command_config(cmd_names); // Tango C++ signature
        })
        .def("command_query", [](Tango::DeviceProxy& self, std::string cmd_name) -> Tango::CommandInfo {
            return self.command_query(cmd_name); // Tango C++ signature
        })
        .def("command_list_query", [](Tango::DeviceProxy& self) -> Tango::CommandInfoList* {
            return std::move(self.command_list_query()); // Tango C++ signature
        })
        .def("import_info", [](Tango::DeviceProxy& self) {
            py::print("import info");
//            Tango::DbDevImportInfo info = self.import_info(); // Tango C++ signature
            self.import_info(); // Tango C++ signature
            py::print("imported info");
            Tango::DbDevImportInfo info;
            return info;
        })
        //
        // property methods
        //
        .def("_get_property", [](Tango::DeviceProxy& self, std::string& prop_name) -> Tango::DbData {
            Tango::DbData dbData;
            self.get_property(prop_name, dbData); // Tango C++ signature
            return dbData;
        })
        .def("_get_property", [](Tango::DeviceProxy& self, std::vector<std::string>& prop_names) -> Tango::DbData {
            Tango::DbData dbData;
            self.get_property(prop_names, dbData); // Tango C++ signature
            return dbData;
        })
        .def("_get_property", [](Tango::DeviceProxy& self, Tango::DbData dbData) -> Tango::DbData {
            self.get_property(dbData); // Tango C++ signature
            return dbData;
        })
        .def("_put_property", [](Tango::DeviceProxy& self, Tango::DbData& dbData) -> void {
            self.put_property(dbData); // Tango C++ signature
        })
        .def("_delete_property", [](Tango::DeviceProxy& self, std::string& prop_name) -> void {
            self.delete_property(prop_name); // Tango C++ signature
        })
        .def("_delete_property", [](Tango::DeviceProxy& self, std::vector<std::string>& prop_names) -> void {
            self.delete_property(prop_names); // Tango C++ signature
        })
        .def("_delete_property", [](Tango::DeviceProxy& self, Tango::DbData& dbData) -> void {
            self.delete_property(dbData); // Tango C++ signature
        })
        .def("_get_property_list", [](Tango::DeviceProxy& self, const std::string& filter, std::vector<std::string>& prop_list) -> std::vector<std::string> {
            self.get_property_list(filter, prop_list); // Tango C++ signature
            return prop_list;
        })
        //
        // pipe methods
        //
        .def("get_pipe_list", [](Tango::DeviceProxy& self) -> std::vector<std::string>* {
            return self.get_pipe_list(); // Tango C++ signature
        })
        .def("_get_pipe_config", [](Tango::DeviceProxy& self, std::string& pipe_name) -> Tango::PipeInfo {
            return self.get_pipe_config(pipe_name); // Tango C++ signature
        })
        .def("_get_pipe_config", [](Tango::DeviceProxy& self, std::vector<std::string>& pipe_names) -> Tango::PipeInfoList* {
            return  self.get_pipe_config(pipe_names); // Tango C++ signature
        })
        .def("_set_pipe_config", [](Tango::DeviceProxy& self, Tango::PipeInfoList& pipes) -> void {
            self.set_pipe_config(pipes); // Tango C++ signature
        })
        .def("__read_pipe", [](Tango::DeviceProxy& self, const std::string& pipe_name) {
            AutoPythonAllowThreads guard;
            return self.read_pipe(pipe_name); // Tango C++ signature
        })
        .def("__write_pipe", [](Tango::DeviceProxy& self, const std::string& pipe_name,
                              const std::string& root_blob_name, py::object py_value) {
            Tango::DevicePipe device_pipe(pipe_name, root_blob_name);
            PyDeviceProxy::set_value(device_pipe, py_value);
            AutoPythonAllowThreads guard;
            self.write_pipe(device_pipe); // Tango C++ signature
        })
        //
        // attribute methods
        //
        .def("get_attribute_list", [](Tango::DeviceProxy& self) -> std::vector<std::string>* {
            return std::move(self.get_attribute_list()); // Tango C++ signature
        })
        .def("_get_attribute_config", [](Tango::DeviceProxy& self, std::string& attr_name) -> Tango::AttributeInfoEx {
            return self.get_attribute_config(attr_name); // Tango C++ signature
        })
        .def("_get_attribute_config", [](Tango::DeviceProxy& self, std::vector<std::string>& attr_names) -> Tango::AttributeInfoList* {
            return std::move(self.get_attribute_config(attr_names)); // Tango C++ signature
        })
        .def("_get_attribute_config_ex", [](Tango::DeviceProxy& self, std::vector<std::string>& attr_names) -> Tango::AttributeInfoListEx* {
            return std::move(self.get_attribute_config_ex(attr_names)); // Tango C++ signature
        })
        .def("attribute_query", [](Tango::DeviceProxy& self, std::string& attr_name) -> Tango::AttributeInfoEx {
            return self.attribute_query(attr_name); // Tango C++ signature
        })
        .def("attribute_list_query", [](Tango::DeviceProxy& self) -> Tango::AttributeInfoList* {
            return std::move(self.attribute_list_query()); // Tango C++ signature
        })
        .def("attribute_list_query_ex", [](Tango::DeviceProxy& self) -> Tango::AttributeInfoListEx* {
            return std::move(self.attribute_list_query_ex()); // Tango C++ signature
        })
        .def("_set_attribute_config", [](Tango::DeviceProxy& self, Tango::AttributeInfoList &attrs) -> void {
            self.set_attribute_config(attrs); // Tango C++ signature
        })

        .def("_set_attribute_config", [](Tango::DeviceProxy& self, Tango::AttributeInfoListEx& attrs) -> void {
            self.set_attribute_config(attrs); // Tango C++ signature
        })
        .def("_read_attribute", [](Tango::DeviceProxy& self, std::string& attr_name) -> py::object {
            Tango::DeviceAttribute* dev_attr = nullptr;
            {
                AutoPythonAllowThreads guard;
                dev_attr = new Tango::DeviceAttribute(self.read_attribute(attr_name)); // Tango C++ signature
            }
            return PyDeviceAttribute::convert_to_python(dev_attr, self);
         })
        .def("_read_attributes", [](Tango::DeviceProxy& self, std::vector<std::string> &attr_names) -> py::list {
            std::unique_ptr<std::vector<Tango::DeviceAttribute>> dev_attr_vec;
            {
                AutoPythonAllowThreads guard;
                dev_attr_vec.reset(self.read_attributes(attr_names)); // Tango C++ signature
            }
            return PyDeviceAttribute::convert_to_python(dev_attr_vec, self);
        })
        .def("_write_attribute",[](Tango::DeviceProxy& self, const std::string&  attr_name, py::object py_value) -> void {
            Tango::DeviceAttribute dev_attr;
            PyDeviceAttribute::reset(dev_attr, attr_name, self, py_value);
            AutoPythonAllowThreads guard;
            self.write_attribute(dev_attr); // Tango C++ signature
        })
        .def("_write_attribute", [](Tango::DeviceProxy& self, const Tango::AttributeInfo & attr_info, py::object py_value) -> void {
            Tango::DeviceAttribute da;
            PyDeviceAttribute::reset(da, attr_info, py_value);
            AutoPythonAllowThreads guard;
            self.write_attribute(da); // Tango C++ signature
        })
        .def("_write_attributes", [](Tango::DeviceProxy& self, py::object& py_list) -> void {
            std::vector<Tango::DeviceAttribute> dev_attrs;
            PyDeviceProxy::pylist_to_devattrs(self, py_list, dev_attrs);
            AutoPythonAllowThreads guard;
            self.write_attributes(dev_attrs); // Tango C++ signature
        })
        .def("_write_read_attribute", [](Tango::DeviceProxy& self, std::string& attr_name,
                py::object py_value) -> py::object {
            Tango::DeviceAttribute w_dev_attr;
            Tango::DeviceAttribute* r_dev_attr;
            PyDeviceAttribute::reset(w_dev_attr, attr_name, self, py_value);
            {
                AutoPythonAllowThreads guard;
                Tango::DeviceAttribute da = self.write_read_attribute(w_dev_attr); // Tango C++ signature
                r_dev_attr = new Tango::DeviceAttribute(da);
            }
            // Convert the result back to python
            return PyDeviceAttribute::convert_to_python(r_dev_attr, self);
        })
        .def("_write_read_attributes", [](Tango::DeviceProxy& self, py::object py_list)-> py::list {
            std::vector<Tango::DeviceAttribute> dev_attrs;
            std::vector<std::string> attr_names;
            for (auto item : py_list) {
                py::tuple tup = py::reinterpret_borrow<py::tuple>(item);
                std::string name = tup[0].cast<std::string>();
                attr_names.push_back(name);
            }
            PyDeviceProxy::pylist_to_devattrs(self, py_list, dev_attrs);
            std::unique_ptr<std::vector<Tango::DeviceAttribute>> dev_attr_vec;
            {
                AutoPythonAllowThreads guard;
                // C++ signature
                dev_attr_vec.reset(self.write_read_attributes(dev_attrs, attr_names));
            }
            // Convert the result back to python
            return PyDeviceAttribute::convert_to_python(dev_attr_vec, self);
        })
        //
        // history methods
        //
        .def("command_history", [](Tango::DeviceProxy& self, std::string& cmd_name, int depth) -> std::vector<Tango::DeviceDataHistory>* {
            AutoPythonAllowThreads guard;
            return self.command_history(cmd_name, depth);
        })
        .def("attribute_history", [](Tango::DeviceProxy& self, std::string& attr_name, int depth) -> py::list{
            std::unique_ptr<std::vector<Tango::DeviceAttributeHistory>> attr_hist;
            {
                AutoPythonAllowThreads guard;
                attr_hist.reset(self.attribute_history(attr_name, depth));
            }
            return PyDeviceAttribute::convert_to_python(attr_hist, self);
        })
        //
        // Polling administration methods
        //
        .def("polling_status", [](Tango::DeviceProxy& self) -> std::vector<std::string>* {
            return std::move(self.polling_status());
        })
        .def("poll_command", [](Tango::DeviceProxy& self, std::string& cmd_name, int polling_period) -> void {
            self.poll_command(cmd_name, polling_period);
        })
        .def("poll_attribute", [](Tango::DeviceProxy& self, std::string& att_name, int polling_period) -> void {
            self.poll_attribute(att_name, polling_period);
        })
        .def("get_command_poll_period", [](Tango::DeviceProxy& self, std::string& cmd_name) -> int {
            return self.get_command_poll_period(cmd_name);
        })
        .def("get_attribute_poll_period", [](Tango::DeviceProxy& self, std::string& attr_name) -> int {
            return self.get_attribute_poll_period(attr_name);
        })
        .def("is_command_polled", [](Tango::DeviceProxy& self, std::string& cmd_name) -> bool {
            return self.is_command_polled(cmd_name);
        })
        .def("is_attribute_polled", [](Tango::DeviceProxy& self, std::string& attr_name) -> bool {
            return self.is_attribute_polled(attr_name);
        })
        .def("stop_poll_command", [](Tango::DeviceProxy& self, std::string& cmd_name) -> void {
            self.stop_poll_command(cmd_name);
        })
        .def("stop_poll_attribute", [](Tango::DeviceProxy& self, std::string& attr_name) -> void {
            self.stop_poll_attribute(attr_name);
        })
        //
        // Asynchronous methods
        //
        .def("__read_attributes_asynch", [](Tango::DeviceProxy& self, std::vector<std::string> names) -> long {
            AutoPythonAllowThreads guard;

            return self.read_attributes_asynch(names); // C++ signature
        })
        .def("read_attributes_reply", [](Tango::DeviceProxy& self, long id) -> py::list {
            std::unique_ptr<std::vector<Tango::DeviceAttribute>> dev_attr_vec;
            {
                AutoPythonAllowThreads guard;
                // C++ signature
                dev_attr_vec.reset(self.read_attributes_reply(id));
            }
            return PyDeviceAttribute::convert_to_python(dev_attr_vec, self);
        })
        .def("read_attributes_reply", [](Tango::DeviceProxy& self, long id, long timeout) -> py::list {
            std::unique_ptr<std::vector<Tango::DeviceAttribute>> dev_attr_vec;
            {
                AutoPythonAllowThreads guard;
                dev_attr_vec.reset(self.read_attributes_reply(id, timeout)); // C++ signature
            }
            return PyDeviceAttribute::convert_to_python(dev_attr_vec, self);
        })
        .def("pending_asynch_call", [](Tango::DeviceProxy& self, Tango::asyn_req_type req_type)  -> long {

           return self.pending_asynch_call(req_type); // C++ signature
        })
        .def("__write_attributes_asynch", [](Tango::DeviceProxy& self, py::object py_list) -> long {
            std::vector<Tango::DeviceAttribute> dev_attrs;
            PyDeviceProxy::pylist_to_devattrs(self, py_list, dev_attrs);
            AutoPythonAllowThreads guard;
            // C++ signature
            return self.write_attributes_asynch(dev_attrs);
        })
        .def("write_attributes_reply", [](Tango::DeviceProxy& self, long id) {
            AutoPythonAllowThreads guard;
            self.write_attributes_reply(id); // C++ signature
        })
        .def("write_attributes_reply",[](Tango::DeviceProxy& self, long id, long timeout) {
            AutoPythonAllowThreads guard;

            self.write_attributes_reply(id, timeout); // C++ signature
        })
        .def("__read_attributes_asynch", [](Tango::DeviceProxy& self, std::vector<std::string>& names,
                py::object py_cb) {
//            CallBackAutoDie* cb = py::cast<CallBackAutoDie>(py_cb);
//            cb->set_autokill_references(py_cb, py_self);
            try {
                AutoPythonAllowThreads guard;
                // C++ signature
//                self.read_attributes_asynch(names, cb);
            } catch (...) {
//                cb->unset_autokill_references();
                throw;
            }
        })
        .def("__write_attributes_asynch",[](Tango::DeviceProxy& self, py::object py_list, py::object py_cb) {
            std::vector<Tango::DeviceAttribute> dev_attrs;
            PyDeviceProxy::pylist_to_devattrs(self, py_list, dev_attrs);
//            CallBackAutoDie* cb = py::cast<CallBackAutoDie*>(py_cb);
//            cb->set_autokill_references(py_cb, py_self);
//            cb.attr_written(nullptr);
            try {
                AutoPythonAllowThreads guard;
                // C++ signature
//                self.write_attributes_asynch(dev_attrs, *cb);
            } catch (...) {
//                cb->unset_autokill_references();
                throw;
            }
        })
        //
        // Logging administration methods
        //
        .def("add_logging_target", [](Tango::DeviceProxy& self, const std::string& target_type_name) -> void {
            self.add_logging_target(target_type_name);
        })
        .def("remove_logging_target", [](Tango::DeviceProxy& self, const std::string& target_type_name) -> void {
                self.remove_logging_target(target_type_name);
        })
        .def("get_logging_target", [](Tango::DeviceProxy& self) -> std::vector<std::string> {
            return self.get_logging_target();
        })
        .def("get_logging_level", [](Tango::DeviceProxy& self) -> int {
            return self.get_logging_level();
        })
        .def("set_logging_level", [](Tango::DeviceProxy& self, int level) -> void {
            self.set_logging_level(level);
        })
        //
        // Event methods
        //
        .def("__subscribe_event", [](Tango::DeviceProxy& self,
                                     std::string& attr_name,
                                     Tango::EventType event,
                                     py::object py_cb_or_queuesize,
                                     std::vector<std::string> filters,
                                     bool stateless) -> int {
            py::print("in dp subscribe event");
            if (PyDeviceProxy::check_cast<int>(py_cb_or_queuesize)) {
                int event_queue_size = py_cb_or_queuesize.cast<int>();
                AutoPythonAllowThreads guard;
                // C++ signature
                return self.subscribe_event(attr_name, event, event_queue_size, filters, stateless);
            } else {
                CallBackPushEvent* cb = py::cast<CallBackPushEvent*>(py_cb_or_queuesize);
                cb->set_device(self);
                cb->m_callback = py::getattr(py_cb_or_queuesize, "push_event");
                AutoPythonAllowThreads guard;
                // C++ signature
                return self.subscribe_event(attr_name, event, cb, filters, stateless);
            }
        }, py::arg("attr_name"),
           py::arg("event"),
           py::arg("cb_or_queuesize"),
           py::arg("filters")=py::list(),
           py::arg("stateless")=false)

       .def("__unsubscribe_event", [](Tango::DeviceProxy& self, int event_id) -> void {
            // If the callback is running, unsubscribe_event will lock
            // until it finishes. So we MUST release GIL to avoid a deadlock
            AutoPythonAllowThreads guard;
            // C++ signature
            self.unsubscribe_event(event_id);
        })
        .def("__get_callback_events", [](Tango::DeviceProxy& self, int event_id,
                 CallBackPushEvent *cb) -> void{
             cb->set_device(self);
             // C++ signature
             self.get_events(event_id, cb);
        })
        .def("__get_attr_conf_events", [](Tango::DeviceProxy& self, int event_id) {
            return PyDeviceProxy::get_events__aux<Tango::AttrConfEventData, Tango::AttrConfEventDataList>(
                                                        self, event_id);
        })
        .def("__get_data_events", [](Tango::DeviceProxy& self, int event_id) {
            return PyDeviceProxy::get_events__aux<Tango::EventData, Tango::EventDataList>(
                                                        self, event_id);
        })
        .def("__get_data_ready_events", [](Tango::DeviceProxy& self, int event_id) {
            return PyDeviceProxy::get_events__aux<Tango::DataReadyEventData, Tango::DataReadyEventDataList>(
                self, event_id);
        })
        .def("__get_devIntrChange_events", [](Tango::DeviceProxy& self, int event_id) {
            return PyDeviceProxy::get_events__aux<Tango::DevIntrChangeEventData, Tango::DevIntrChangeEventDataList>(
                    self, event_id);
        })
        .def("__get_pipe_events", [](Tango::DeviceProxy& self, int event_id) {
            return PyDeviceProxy::get_events__aux<Tango::PipeEventData, Tango::PipeEventDataList>(
                    self, event_id);
        })
        //
        // methods to access data in event queues
        //
        .def("event_queue_size", [](Tango::DeviceProxy& self, int event_id) -> int {
            // C++ signature
            return self.event_queue_size(event_id);
        })
        .def("get_last_event_date", [](Tango::DeviceProxy& self, int event_id) -> Tango::TimeVal {
            // C++ signature
            return self.get_last_event_date(event_id);
        })
        .def("is_event_queue_empty", [](Tango::DeviceProxy& self, int event_id) -> bool {
            // C++ signature
            return self.is_event_queue_empty(event_id);
        })
        //
        // Locking methods
        //
        .def("lock", [](Tango::DeviceProxy& self, int lock_validity) -> void {
            // C++ signature
            return self.lock(lock_validity);
        }, py::arg("lock_validity")=Tango::DEFAULT_LOCK_VALIDITY)
        .def("unlock", [](Tango::DeviceProxy& self, bool force) -> void {
            // C++ signature
            return self.unlock(force);
        }, py::arg("force")=false)

        .def("locking_status", [](Tango::DeviceProxy& self) -> std::string {
            // C++ signature
            return self.locking_status();
        })
        .def("is_locked", [](Tango::DeviceProxy& self) -> bool {
            // C++ signature
            return self.is_locked();
        })
        .def("is_locked_by_me", [](Tango::DeviceProxy& self) -> bool {
            // C++ signature
            return self.is_locked_by_me();
        })
        .def("get_locker", [](Tango::DeviceProxy& self, Tango::LockerInfo& li) -> bool {
            // C++ signature
            return self.get_locker(li);
        })
        //
        // Connection methods
        //
        .def("get_db_host", [](Tango::DeviceProxy& self) -> std::string {
            return self.get_db_host();
        })
        .def("get_db_port", [](Tango::DeviceProxy& self) -> std::string {
            return self.get_db_port();
        })
        .def("get_db_port_num", [](Tango::DeviceProxy& self) -> long {
            return self.get_db_port_num();
        })
        .def("get_from_env_var", [](Tango::DeviceProxy& self) -> bool {
            return self.get_from_env_var();
        })
        .def("get_fqdn", [](Tango::DeviceProxy& self) -> std::string {
            std::string fqdn_str;
            self.get_fqdn(fqdn_str);
            return std::move(fqdn_str);
        })
        .def("is_dbase_used", [](Tango::DeviceProxy& self) -> bool {
            return self.is_dbase_used();
        })
        .def("get_dev_host", [](Tango::DeviceProxy& self) -> std::string {
            return self.get_dev_host();
        })
        .def("get_dev_port", [](Tango::DeviceProxy& self) -> std::string {
            return self.get_dev_port();
        })
        .def("get_idl_version", [](Tango::DeviceProxy& self) -> int {
            return self.get_idl_version();
        })
        .def("set_timeout_millis", [](Tango::DeviceProxy& self, int timeout) -> void {
            self.set_timeout_millis(timeout);
        })
        .def("get_timeout_millis", [](Tango::DeviceProxy& self) -> int {
            return self.get_timeout_millis();
        })
        .def("get_source", [](Tango::DeviceProxy& self) -> Tango::DevSource {
            return self.get_source();
        })
        .def("set_source", [](Tango::DeviceProxy& self, Tango::DevSource src) -> void {
            self.set_source(src);
        })
        .def("get_transparency_reconnection", [](Tango::DeviceProxy& self) -> bool {
            return self.get_transparency_reconnection();
        })
        .def("set_transparency_reconnection", [](Tango::DeviceProxy& self, bool reconn) -> void {
            self.set_transparency_reconnection(reconn);
        })
        .def("__command_inout", [](Tango::DeviceProxy& self, std::string& cmd_name) -> Tango::DeviceData {
            AutoPythonAllowThreads guard;
            return self.command_inout(cmd_name);
        })
        .def("__command_inout", [](Tango::DeviceProxy& self, std::string& cmd_name,
                Tango::DeviceData &argin) -> Tango::DeviceData {
            AutoPythonAllowThreads guard;
            return self.command_inout(cmd_name, argin);
        })
        .def("__command_inout_asynch_id", [](Tango::DeviceProxy& self, std::string& cmd_name,
                Tango::DeviceData &argin, bool forget) -> long {
            AutoPythonAllowThreads guard;
            return self.command_inout_asynch(cmd_name, argin, forget);
        }, py::arg("cmd_name"), py::arg("argin"), py::arg("forget")=false)
        .def("__command_inout_asynch_id", [](Tango::DeviceProxy& self, std::string& cmd_name, bool forget) -> long {
            AutoPythonAllowThreads guard;
            return self.command_inout_asynch(cmd_name, forget);
        }, py::arg("cmd_name"), py::arg("forget")=false)

        .def("__command_inout_asynch_cb", [](Tango::DeviceProxy& self, std::string& cmd_name,
                py::object py_cb) -> void {
            std::shared_ptr<CallBackAutoDie> cb = std::shared_ptr<CallBackAutoDie>(py::cast<CallBackAutoDie*>(py_cb));
            cerr << "after cast" << endl;
//            cb->m_callback(py::getattr(py_cb, "m_callback"));
//            cb->set_weak_parent(py::getattr(py_cb, "m_weak_parent"));
//            cb->set_autokill_references(cb, self);
            try {
                cerr << "inout cmd no arg async going to issue" << endl;
                AutoPythonAllowThreads guard;
                // C++ signature
                self.command_inout_asynch(cmd_name, *cb);
                cerr << "inout cmd no arg async issued" << endl;
            } catch (...) {
//                cb->unset_autokill_references();
                throw;
            }
        })
        .def("__command_inout_asynch_cb", [](Tango::DeviceProxy& self, std::string& cmd_name,
                Tango::DeviceData &argin, py::object py_cb) -> void {
            CallBackAutoDie* cb = py::cast<CallBackAutoDie*>(py_cb);
//            cb->set_callback(py::getattr(py_cb, "m_callback"));
//            cb->set_autokill_references(cb, self);
            try {
                AutoPythonAllowThreads guard;
                // C++ signature
                self.command_inout_asynch(cmd_name, argin, *cb);
                cerr << "inout cmd async issued" << endl;
            } catch (...) {
//                cb->unset_autokill_references();
                throw;
            }
        })
        .def("command_inout_reply_raw", [](Tango::DeviceProxy& self, long id) -> Tango::DeviceData {
            AutoPythonAllowThreads guard;
            return self.command_inout_reply(id);
        })
        .def("command_inout_reply_raw", [](Tango::DeviceProxy& self, long id, long timeout) -> Tango::DeviceData {
            AutoPythonAllowThreads guard;
            return self.command_inout_reply(id, timeout);
        })
        //
        // Asynchronous methods
        //
        .def("get_asynch_replies", [](Tango::DeviceProxy& self) -> void {
            AutoPythonAllowThreads guard;
            cout << "start get async replies" << endl;
            self.get_asynch_replies();
            cout << "finish get async replies" << endl;
        })
        .def("get_asynch_replies", [](Tango::DeviceProxy& self, long timeout) -> void {
            AutoPythonAllowThreads guard;
            self.get_asynch_replies(timeout);
        })
        .def("cancel_asynch_request", [](Tango::DeviceProxy& self, long id) -> void {
            self.cancel_asynch_request(id);
        })
        .def("cancel_all_polling_asynch_request",[](Tango::DeviceProxy& self) -> void {
            self.cancel_all_polling_asynch_request();
        })
        //
        // Control access related methods
        //
        .def("get_access_control", [](Tango::DeviceProxy& self) -> Tango::AccessControlType {
            return self.get_access_control();
        })
        .def("set_access_control", [](Tango::DeviceProxy& self, Tango::AccessControlType& act) -> void {
            self.set_access_control(act);
        })
        .def("get_access_right", [](Tango::DeviceProxy& self) -> Tango::AccessControlType {
            return self.get_access_right();
        })

    ;
}
