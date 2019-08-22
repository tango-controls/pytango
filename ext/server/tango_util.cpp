/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <list>
#include <tango.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pyutils.h>
#include <pytgutils.h>
#include "server/device_class.h"
#include <exception>

namespace py = pybind11;

namespace PyUtil
{
    void _class_factory(Tango::DServer* dserver)
    {
        AutoPythonGIL guard;
        py::print("need class_factory");
        py::object tango = py::cast<py::object>(PyImport_AddModule("tango"));
        py::list cpp_class_list = tango.attr("get_cpp_classes")();
        py::print(len(cpp_class_list));
        // First, create CPP class if any.
        // Their names are defined in a Python list
        auto cl_len =len(cpp_class_list);
        for(auto i = 0; i < cl_len; ++i)
        {
            py::print("tango_util:_class_factory code needs checking");
//            std::tuple class_info = cpp_class_list[i].cast<std::tuple>();
//            char *class_name = extract<char *>(class_info[0]);
//            char *par_name   = extract<char *>(class_info[1]);
//            dserver->_create_cpp_class(class_name, par_name);
        }
        //
        // Create Python classes with a call to the class_factory Python function
        //
        py::print("tango_util.cpp: Entering python class factory");
        tango.attr("class_factory")();
        py::print("Leaving python class factory");
        //
        // Make all Python tango class(es) known to C++ and set the PyInterpreter state
        //
        py::list constructed_classes = tango.attr("get_constructed_classes")();
        auto cc_len = len(constructed_classes);
        py::print("length of constructed classes", cc_len);
        for (auto i = 0; i < cc_len; i++) {
            DeviceClass* device_class_ptr = constructed_classes[i].cast<DeviceClass*>();
            py::print("5");
            device_class_ptr->m_self.attr("command_factory")();
            dserver->_add_class(device_class_ptr);
            py::print("here4");
        }
    }

    inline Tango::Util* init(std::list<std::string> &arglist)
    {
        int argc = (int) arglist.size();
        char** argv = new char*[argc];
        static Tango::Util* res = 0;
        int i = 0;
        for (auto item : arglist) {
            argv[i] = (char*)item.c_str();
            py::print(argv[i]);
            i++;
        }
        res = Tango::Util::init(argc, argv);
        delete [] argv;
// Is this necessary here, its done in pytango.cpp?
        if (PyEval_ThreadsInitialized() == 0)
        {
            PyEval_InitThreads();
        }
        py::print(res);
        return res;
    }

    inline bool event_loop()
    {
        py::print("need event_loop");
        AutoPythonGIL guard;
        py::object tango = py::cast<py::object>(PyImport_AddModule("tango"));
        py::object py_event_loop = tango.attr("_server_event_loop");
        return py_event_loop().cast<bool>();
    }

    inline void server_set_event_loop(Tango::Util& self, py::object& py_event_loop)
    {
        py::print("need server_set_event_loop");
        py::object tango = py::cast<py::object>(PyImport_AddModule("tango"));
        if (py_event_loop.is_none()) {
            self.server_set_event_loop(NULL);
            tango.attr("_server_event_loop") = py_event_loop;
        } else {
            tango.attr("_server_event_loop") = py_event_loop;
            self.server_set_event_loop(event_loop);
        }
    }
}

void export_util(py::module &m) {
    py::class_<Tango::Interceptors>(m, "Interceptors")
        .def("create_thread", &Tango::Interceptors::create_thread)
        .def("delete_thread", &Tango::Interceptors::delete_thread)
    ;
    py::class_<Tango::Util, std::unique_ptr<Tango::Util, py::nodelete>>(m, "Util")
//        .def("__init__", [](Tango::Util& self, std::list<std::string>& args) {
        .def(py::init([](std::list<std::string>& args) {
            py::print(args);
            Tango::Util* instance = PyUtil::init(args);
            py::print(instance);
            return std::unique_ptr<Tango::Util, py::nodelete>(instance);
        }))
        .def_static("instance", [](bool b) {
            py::print("bool instance");
            return Tango::Util::instance(b);
        }, py::arg("bool") = true, py::return_value_policy::reference)

        .def("set_trace_level", &Tango::Util::set_trace_level)
        .def("get_trace_level", &Tango::Util::get_trace_level)
        .def("get_ds_inst_name", &Tango::Util::get_ds_inst_name,
            py::return_value_policy::copy)
        .def("get_ds_exec_name", &Tango::Util::get_ds_exec_name,
            py::return_value_policy::copy)
        .def("get_ds_name", &Tango::Util::get_ds_name,
            py::return_value_policy::copy)
        .def("get_host_name", &Tango::Util::get_host_name,
            py::return_value_policy::copy)
        .def("get_pid_str", [](Tango::Util& self) {
            return self.get_pid_str();
        })
        .def("get_pid", &Tango::Util::get_pid)
        .def("get_tango_lib_release", &Tango::Util::get_tango_lib_release)
        .def("get_version_str", [](Tango::Util& self){
            return self.get_version_str();
        })
        .def("get_server_version", &Tango::Util::get_server_version,
            py::return_value_policy::copy)
        .def("set_server_version", &Tango::Util::set_server_version)
        .def("set_serial_model", &Tango::Util::set_serial_model)
        .def("get_serial_model", &Tango::Util::get_serial_model)
        .def("reset_filedatabase", &Tango::Util::reset_filedatabase)
        .def("unregister_server", &Tango::Util::unregister_server)
        .def("get_dserver_device", &Tango::Util::get_dserver_device,
            py::return_value_policy::reference_internal)
        .def("server_init", [](Tango::Util& self, bool with_window) -> void {
            py::print("tango_util.py: server_init() need this");
//            AutoPythonAllowThreads guard;
            Tango::DServer::register_class_factory(PyUtil::_class_factory);
            Tango::Util* instance = Tango::Util::instance();
            py::print(instance);
            instance->server_init();
        }, py::arg("with_window")=false)
        .def("server_run", [](Tango::Util& self) -> void {
            AutoPythonAllowThreads guard;
            Tango::Util::instance()->server_run();
        })
        .def("server_cleanup", &Tango::Util::server_cleanup)
        .def("trigger_cmd_polling", &Tango::Util::trigger_cmd_polling)
        .def("trigger_attr_polling", &Tango::Util::trigger_attr_polling)
        .def("set_polling_threads_pool_size", &Tango::Util::set_polling_threads_pool_size)
        .def("get_polling_threads_pool_size", &Tango::Util::get_polling_threads_pool_size)
        .def("is_svr_starting", &Tango::Util::is_svr_starting)
        .def("is_svr_shutting_down", &Tango::Util::is_svr_shutting_down)
        .def("is_device_restarting", &Tango::Util::is_device_restarting)
        .def("get_sub_dev_diag", &Tango::Util::get_sub_dev_diag,
            py::return_value_policy::reference_internal)
        .def("connect_db", &Tango::Util::connect_db)
        .def("reset_filedatabase", &Tango::Util::reset_filedatabase)
        .def("get_database", [](Tango::Util& self) {
            py::print("this is the get_database call");
            return self.get_database();
        }, py::return_value_policy::reference_internal)
        .def("unregister_server", &Tango::Util::unregister_server)
        .def("get_device_list_by_class", [](Tango::Util &self, const std::string& name) {
            py::print("need to check the return code here");
            // Does the vector returned by this call get automatically
            // translated into a py::list, otherwise will have to iterate
            // over the vector and append to the list ??????????????????
            // see get_device_list() below.
            return self.get_device_list_by_class(name);
        })
        .def("get_device_by_name", [](Tango::Util &self, const std::string& dev_name) {
            Tango::DeviceImpl *value = self.get_device_by_name(dev_name);
            py::object py_value = py::cast(value);
            return py_value;
        })
        .def("get_device_list", [](Tango::Util &self, const std::string& name) {
            py::list py_dev_list;
            std::vector<Tango::DeviceImpl *> dev_list = self.get_device_list(name);
            for(std::vector<Tango::DeviceImpl *>::iterator it = dev_list.begin(); it != dev_list.end(); ++it)
            {
                py::object py_value = py::cast(*it);
                py_dev_list.append(py_value);
            }
            return py_dev_list;
        })
        .def("server_set_event_loop", [](Tango::Util &self, py::object& py_event_loop) -> void {
            PyUtil::server_set_event_loop(self, py_event_loop);
        })
        .def("set_interceptors", &Tango::Util::set_interceptors)

        .def_property_readonly_static("_UseDb", [](py::object /* self */) -> bool {
            return Tango::Util::_UseDb;
        })
        .def_property_readonly_static("_FileDb", [](py::object /* self */) -> bool {
            return Tango::Util::_FileDb;
        })
        .def("get_dserver_ior", [](Tango::Util& self, Tango::DServer* dserver) {
            Tango::Device_var d = dserver->_this();
            dserver->set_d_var(Tango::Device::_duplicate(d));
            return self.get_orb()->object_to_string(d);
        })
        .def("get_device_ior", [](Tango::Util& self, Tango::DeviceImpl* device) {
            return self.get_orb()->object_to_string(device->get_d_var());
        })
        .def("orb_run", [](Tango::Util& self) -> void {
            AutoPythonAllowThreads guard;
            self.get_orb()->run();
        })
    ;
}
