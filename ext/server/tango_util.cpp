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
#include <pyutils.h>

namespace py = pybind11;

//namespace PyUtil
//{
//    void _class_factory(Tango::DServer* dserver)
//    {
//        AutoPythonGIL guard;
//        PYTANGO_MOD
//
//    //
//    // First, create CPP class if any. Their names are defined in a Python list
//    //
//        boost::python::list cpp_class_list =
//            extract<boost::python::list>(pytango.attr("get_cpp_classes")());
//        Py_ssize_t cl_len = boost::python::len(cpp_class_list);
//        for(Py_ssize_t i = 0; i < cl_len; ++i)
//        {
//            bopy::tuple class_info = extract<bopy::tuple>(cpp_class_list[i]);
//            char *class_name = extract<char *>(class_info[0]);
//            char *par_name   = extract<char *>(class_info[1]);
//            dserver->_create_cpp_class(class_name, par_name);
//        }
//
//    //
//    // Create Python classes with a call to the class_factory Python function
//    //
//        pytango.attr("class_factory")();
//
//    //
//    // Make all Python tango class(es) known to C++ and set the PyInterpreter state
//    //
//        boost::python::list constructed_classes(pytango.attr("get_constructed_classes")());
//        Py_ssize_t cc_len = boost::python::len(constructed_classes);
//        for(Py_ssize_t i = 0; i < cc_len; ++i)
//        {
//            CppDeviceClass *cpp_dc = extract<CppDeviceClass *> (constructed_classes[i])();
//            dserver->_add_class(cpp_dc);
//        }
//    }
//
//
//    inline Tango::Util* init(boost::python::object &obj)
//    {
//        PyObject *obj_ptr = obj.ptr();
//        if(PySequence_Check(obj_ptr) == 0)
//        {
//            raise_(PyExc_TypeError, param_must_be_seq);
//        }
//
//        int argc = (int) PySequence_Length(obj_ptr);
//        char** argv = new char*[argc];
//        Tango::Util* res = 0;
//
//        try {
//            for(int i = 0; i < argc; ++i)
//            {
//                PyObject* item_ptr = PySequence_GetItem(obj_ptr, i);
//                str item = str(handle<>(item_ptr));
//                argv[i] = extract<char *>(item);
//            }
//	    res = Tango::Util::init(argc, argv);
//        } catch (...) {
//            delete [] argv;
//            throw;
//        }
//        delete [] argv;
//
//	if (PyEval_ThreadsInitialized() == 0)
//	{
//	    PyEval_InitThreads();
//	}
//
//        return res;
//    }

//    inline bool event_loop()
//    {
//        AutoPythonGIL guard;
//        PYTANGO_MOD
//        boost::python::object py_event_loop = pytango.attr("_server_event_loop");
//        boost::python::object py_ret = py_event_loop();
//        bool ret = boost::python::extract<bool>(py_ret);
//        return ret;
//    }
//
//    inline void server_set_event_loop(Tango::Util& self,
//                                      boost::python::object& py_event_loop)
//    {
//        PYTANGO_MOD
//        if (py_event_loop.ptr() == Py_None)
//        {
//            self.server_set_event_loop(NULL);
//            pytango.attr("_server_event_loop") = py_event_loop;
//        }
//        else
//        {
//            pytango.attr("_server_event_loop") = py_event_loop;
//            self.server_set_event_loop(event_loop);
//        }
//    }

//    static boost::shared_ptr<Tango::Util>
//    makeUtil(boost::python::object& args)
//    {
//        Tango::Util* util = PyUtil::init(args);
//        return boost::shared_ptr<Tango::Util>(util);
//    }
//}
//
//
//BOOST_PYTHON_FUNCTION_OVERLOADS (server_init_overload, PyUtil::server_init, 1, 2)

void export_util(py::module &m) {
    py::class_<Tango::Interceptors>(m, "Interceptors")
        .def("create_thread", &Tango::Interceptors::create_thread)
        .def("delete_thread", &Tango::Interceptors::delete_thread)
    ;

// noncopyable
    py::class_<Tango::Util>(m, "Util")
//        .def("__init__", py::make_constructor(PyUtil::makeUtil))
//        .def("init", PyUtil::init,
//            py::return_value_policy<reference_existing_object>())
//        .staticmethod("init")
//
        .def_static("instance", []() -> Tango::Util* {
            return Tango::Util::instance();
        }, py::return_value_policy::reference)
        .def_static("instance", [](bool b) -> Tango::Util* {
            return Tango::Util::instance(b);
        }, py::return_value_policy::reference)

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
        .def("server_init", [](Tango::Util& instance, bool with_window=false){
            AutoPythonAllowThreads guard;
//            Tango::DServer::register_class_factory(_class_factory);
            instance.server_init(with_window);
        })
        .def("server_run", [](Tango::Util& instance) {
            AutoPythonAllowThreads guard;
            instance.server_run();
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
        .def("get_database", &Tango::Util::get_database,
            py::return_value_policy::reference_internal)
        .def("unregister_server", &Tango::Util::unregister_server)
        .def("get_device_list_by_class", [](Tango::Util &self, const string &name){
            // Does the vector returned by this call get automatically
            // translated into a py::list, otherwise will have to interate
            // over the vector and append to the list ??????????????????
            // see get_device_list() below.
            return self.get_device_list_by_class(name);
        })
        .def("get_device_by_name", [](Tango::Util &self, const string &dev_name){
            Tango::DeviceImpl *value = self.get_device_by_name(dev_name);
            py::object py_value = py::cast(value);
            return py_value;
        })
        .def("get_device_list", [](Tango::Util &self, const string &name){
            py::list py_dev_list;
            std::vector<Tango::DeviceImpl *> dev_list = self.get_device_list(name);
            for(std::vector<Tango::DeviceImpl *>::iterator it = dev_list.begin(); it != dev_list.end(); ++it)
            {
                py::object py_value = py::cast(*it);
                py_dev_list.append(py_value);
            }
            return py_dev_list;
        })
//        .def("server_set_event_loop", &PyUtil::server_set_event_loop)
        .def("set_interceptors", &Tango::Util::set_interceptors)
        .def_property_static("_UseDb", []() -> bool {
            return Tango::Util::_UseDb;
            }, [](bool use_db) -> void {
            Tango::Util::_UseDb = use_db;
        })
        .def_property_readonly_static("_FileDb", []() -> bool {
            return Tango::Util::_FileDb;
        })
        .def("get_dserver_ior", [](Tango::Util& self, Tango::DServer* dserver){
            Tango::Device_var d = dserver->_this();
            dserver->set_d_var(Tango::Device::_duplicate(d));
            return self.get_orb()->object_to_string(d);
        })
        .def("get_device_ior", [](Tango::Util& self, Tango::DeviceImpl* device){
            return self.get_orb()->object_to_string(device->get_d_var());
        })
        .def("orb_run", [](Tango::Util& self){
            AutoPythonAllowThreads guard;
            self.get_orb()->run();
        })
    ;
}
