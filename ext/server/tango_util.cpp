/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

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
#include <thread>
namespace py = pybind11;

namespace PyUtil
{
    void _class_factory(Tango::DServer* dserver)
    {
        std::thread::id id3 = std::this_thread::get_id();
        std::cout << "_class_factory thread id" << id3 << std::endl;

        AutoPythonGIL guard;
        py::object tango = py::cast<py::object>(PyImport_AddModule("tango"));
        py::list cpp_class_list = tango.attr("get_cpp_classes")();
        // First, create CPP class if any, their names are defined in a Python list
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
        for (auto i = 0; i < cc_len; i++) {
            DeviceClass* device_class_ptr = constructed_classes[i].cast<DeviceClass*>();
            dserver->_add_class(device_class_ptr);
        }
    }

    inline Tango::Util* init(std::list<std::string>& arglist)
    {
        std::cout << "tango_util::init called" << std::endl;
        int argc = (int) arglist.size();
        char** argv = new char*[argc];
        static Tango::Util* res = 0;
        int i = 0;
        for (auto item : arglist) {
            argv[i++] = (char*)item.c_str();
        }
        res = Tango::Util::init(argc, argv);
        delete [] argv;
        // Is this necessary here, its done in pytango.cpp?
        if (PyEval_ThreadsInitialized() == 0)
        {
            PyEval_InitThreads();
            py::gil_scoped_release release;
        }
        return res;
    }

    inline bool event_loop()
    {
        AutoPythonGIL guard;
        py::object tango = py::cast<py::object>(PyImport_AddModule("tango"));
        py::object py_event_loop = tango.attr("_server_event_loop");
        return py_event_loop().cast<bool>();
    }

    inline void server_set_event_loop(Tango::Util& self, py::object& py_event_loop)
    {
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
        .def(py::init([](std::list<std::string>& args) {
            Tango::Util* instance = PyUtil::init(args);
            return std::unique_ptr<Tango::Util, py::nodelete>(instance);
        }))
        .def_static("instance", [](bool b) {
            return Tango::Util::instance(b);
        }, py::arg("bool") = true)
        .def("set_trace_level", [](Tango::Util& self, int level) -> void {
            self.set_trace_level(level);
        })
        .def("get_trace_level", [](Tango::Util& self) -> int {
            return self.get_trace_level();
        })
        .def("get_ds_inst_name", [](Tango::Util& self) -> std::string {
            return self.get_ds_inst_name();
        })
        .def("get_ds_exec_name", [](Tango::Util& self) -> std::string {
            return self.get_ds_exec_name();
        })
        .def("get_ds_name", [](Tango::Util& self) -> std::string {
            return self.get_ds_name();
        })
        .def("get_host_name", [](Tango::Util& self) -> std::string {
            return self.get_host_name();
        })
        .def("get_pid_str", [](Tango::Util& self) -> std::string {
            return self.get_pid_str();
        })
        .def("get_pid", [](Tango::Util& self) -> int {
            return self.get_pid();
        })
        .def("get_tango_lib_release", [](Tango::Util& self) -> long {
            return self.get_tango_lib_release();
        })
        .def("get_version_str", [](Tango::Util& self) -> std::string {
            return self.get_version_str();
        })
        .def("get_server_version", [](Tango::Util& self) -> std::string {
            return self.get_server_version();
        })
        .def("set_server_version", [](Tango::Util& self, std::string& version) -> void {
            self.set_server_version(version.c_str());
        })
        .def("set_serial_model", [](Tango::Util& self, Tango::SerialModel model) -> void {
            self.set_serial_model(model);
        })
        .def("get_serial_model", [](Tango::Util& self) -> Tango::SerialModel {
            return self.get_serial_model();
        })
        .def("reset_filedatabase", [](Tango::Util& self) -> void {
            self.reset_filedatabase();
        })
        .def("unregister_server", [](Tango::Util& self) -> void {
            self.unregister_server();
        })
        .def("get_dserver_device", [](Tango::Util& self) -> Tango::DServer* {
            return self.get_dserver_device();
        })
        .def("server_init", [](Tango::Util& self, bool with_window) -> void {
            std::thread::id id1 = std::this_thread::get_id();
            std::cout << "tango_util.py: server_init thread id " << id1 << std::endl;
            std::cerr << "tango_util.py: server_init() thread init'd " << PyEval_ThreadsInitialized() << std::endl;
//            AutoPythonAllowThreads guard;
            Tango::DServer::register_class_factory(PyUtil::_class_factory);
            std::thread::id id2 = std::this_thread::get_id();
            std::cout << "tango_util.py: server_init thread id2 " << id2 << std::endl;
            self.server_init();
            std::cerr << "tango_util.py: init done" << std::endl;
        }, py::arg("with_window")=false)
        .def("server_run", [](Tango::Util& self) -> void {
            AutoPythonAllowThreads guard;
            self.server_run();
        })
        .def("server_cleanup",  [](Tango::Util& self) -> void {
            self.server_cleanup();
        })
        .def("trigger_cmd_polling",  [](Tango::Util& self, Tango::DeviceImpl *dev, const std::string &name) -> void {
            self.trigger_cmd_polling(dev, name);
        })
        .def("trigger_attr_polling",  [](Tango::Util& self, Tango::DeviceImpl *dev, const std::string &name) -> void {
            self.trigger_attr_polling(dev, name);
        })
        .def("set_polling_threads_pool_size",  [](Tango::Util& self, long nb_threads) -> void {
            self.set_polling_threads_pool_size(nb_threads);
        })
        .def("get_polling_threads_pool_size", [](Tango::Util& self) -> long {
            return self.get_polling_threads_pool_size();
        })
        .def("is_svr_starting", [](Tango::Util& self) -> bool {
            return self.is_svr_starting();
        })
        .def("is_svr_shutting_down", [](Tango::Util& self) -> bool {
            return self.is_svr_shutting_down();
        })
        .def("is_device_restarting", [](Tango::Util& self, std::string &dev_name) -> bool {
            return self.is_device_restarting(dev_name);
        })
        .def("get_sub_dev_diag", [](Tango::Util& self) -> py::object {
            return py::cast(self.get_sub_dev_diag());
        })
        .def("connect_db", [](Tango::Util& self) -> void {
            self.connect_db();
        })
        .def("reset_filedatabase", [](Tango::Util& self) -> void {
            self.reset_filedatabase();
        })
        .def("get_database", [](Tango::Util& self) {
            return self.get_database();
        })
        .def("get_device_list_by_class", [](Tango::Util& self, const std::string& name) {
            py::print("need to check the return code here");
            // Does the vector returned by this call get automatically
            // translated into a py::list, otherwise will have to iterate
            // over the vector and append to the list ??????????????????
            // see get_device_list() below.
            return self.get_device_list_by_class(name);
        })
        .def("get_device_by_name", [](Tango::Util& self, const std::string& dev_name) {
            Tango::DeviceImpl* value = self.get_device_by_name(dev_name);
            return py::cast(value);
        })
        .def("get_device_list", [](Tango::Util& self, const std::string& name) {
            py::list py_dev_list;
            std::vector<Tango::DeviceImpl*> dev_list = self.get_device_list(name);
            for(std::vector<Tango::DeviceImpl*>::iterator it = dev_list.begin(); it != dev_list.end(); ++it)
            {
                py::object py_value = py::cast(*it);
                py_dev_list.append(py_value);
            }
            return py_dev_list;
        })
        .def("server_set_event_loop", [](Tango::Util& self, py::object& py_event_loop) -> void {
            PyUtil::server_set_event_loop(self, py_event_loop);
        })
        .def("set_interceptors", [](Tango::Util& self, Tango::Interceptors *in) -> void {
                self.set_interceptors(in);
        })
        .def("_UseDb", [](Tango::Util& self) -> bool {
            return self._UseDb;
        })
       .def("_FileDb", [](Tango::Util& self) -> bool {
            return self._FileDb;
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
