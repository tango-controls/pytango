/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "defs.h"
#include "pytgutils.h"
#include "exception.h"
#include "server/device_class.h"

using namespace boost::python;

namespace PyUtil
{
    void _class_factory(Tango::DServer* dserver)
    {
        AutoPythonGIL guard;
        PYTANGO_MOD

    //
    // First, create CPP class if any. Their names are defined in a Python list
    //
        boost::python::list cpp_class_list =
            extract<boost::python::list>(pytango.attr("get_cpp_classes")());
        Py_ssize_t cl_len = boost::python::len(cpp_class_list);
        for(Py_ssize_t i = 0; i < cl_len; ++i)
        {
            bopy::tuple class_info = extract<bopy::tuple>(cpp_class_list[i]);
            char *class_name = extract<char *>(class_info[0]);
            char *par_name   = extract<char *>(class_info[1]);
            dserver->_create_cpp_class(class_name, par_name);
        }

    //
    // Create Python classes with a call to the class_factory Python function
    //
        pytango.attr("class_factory")();

    //
    // Make all Python tango class(es) known to C++ and set the PyInterpreter state
    //
        boost::python::list constructed_classes(pytango.attr("get_constructed_classes")());
        Py_ssize_t cc_len = boost::python::len(constructed_classes);
        for(Py_ssize_t i = 0; i < cc_len; ++i)
        {
            CppDeviceClass *cpp_dc = extract<CppDeviceClass *> (constructed_classes[i])();
            dserver->_add_class(cpp_dc);
        }
    }

    void server_init(Tango::Util & instance, bool with_window = false)
    {
        AutoPythonAllowThreads guard;
        Tango::DServer::register_class_factory(_class_factory);
        instance.server_init(with_window);
    }

    void server_run(Tango::Util & instance)
    {
        AutoPythonAllowThreads guard;
        instance.server_run();
    }

    inline Tango::Util* init(boost::python::object &obj)
    {
        PyObject *obj_ptr = obj.ptr();
        if(PySequence_Check(obj_ptr) == 0)
        {
            raise_(PyExc_TypeError, param_must_be_seq);
        }

        int argc = (int) PySequence_Length(obj_ptr);
        char** argv = new char*[argc];
        Tango::Util* res = 0;

        try {
            for(int i = 0; i < argc; ++i)
            {
                PyObject* item_ptr = PySequence_GetItem(obj_ptr, i);
                str item = str(handle<>(item_ptr));
                argv[i] = extract<char *>(item);
            }
	    res = Tango::Util::init(argc, argv);
        } catch (...) {
            delete [] argv;
            throw;
        }
        delete [] argv;

	if (PyEval_ThreadsInitialized() == 0)
	{
	    PyEval_InitThreads();
	}

        return res;
    }

    inline Tango::Util* instance1()
    {
        return Tango::Util::instance();
    }

    inline Tango::Util* instance2(bool b)
    {
        return Tango::Util::instance(b);
    }

    inline object get_device_list_by_class(Tango::Util &self, const string &class_name)
    {
        boost::python::list py_dev_list;
        vector<Tango::DeviceImpl *> &dev_list = self.get_device_list_by_class(class_name);
        for(vector<Tango::DeviceImpl *>::iterator it = dev_list.begin(); it != dev_list.end(); ++it)
        {
            object py_value = object(
                        handle<>(
                            to_python_indirect<
                                Tango::DeviceImpl*,
                                detail::make_reference_holder>()(*it)));

            py_dev_list.append(py_value);
        }
        return py_dev_list;
    }

    inline object get_device_by_name(Tango::Util &self, const string &dev_name)
    {
        Tango::DeviceImpl *value = self.get_device_by_name(dev_name);
        object py_value = object(
                    handle<>(
                        to_python_indirect<
                            Tango::DeviceImpl*,
                            detail::make_reference_holder>()(value)));

        return py_value;
    }

    inline object get_device_list(Tango::Util &self, const string &name)
    {
        boost::python::list py_dev_list;
        vector<Tango::DeviceImpl *> dev_list = self.get_device_list(name);
        for(vector<Tango::DeviceImpl *>::iterator it = dev_list.begin(); it != dev_list.end(); ++it)
        {
            object py_value = object(
                        handle<>(
                            to_python_indirect<
                                Tango::DeviceImpl*,
                                detail::make_reference_holder>()(*it)));
            py_dev_list.append(py_value);
        }
        return py_dev_list;
    }

    inline bool event_loop()
    {
        AutoPythonGIL guard;
        PYTANGO_MOD
        boost::python::object py_event_loop = pytango.attr("_server_event_loop");
        boost::python::object py_ret = py_event_loop();
        bool ret = boost::python::extract<bool>(py_ret);
        return ret;
    }

    inline void server_set_event_loop(Tango::Util& self,
                                      boost::python::object& py_event_loop)
    {
        PYTANGO_MOD
        if (py_event_loop.ptr() == Py_None)
        {
            self.server_set_event_loop(NULL);
            pytango.attr("_server_event_loop") = py_event_loop;
        }
        else
        {
            pytango.attr("_server_event_loop") = py_event_loop;
            self.server_set_event_loop(event_loop);
        }
    }

    void set_use_db(bool use_db)
    {
        Tango::Util::_UseDb = use_db;
    }

    boost::python::str get_dserver_ior(Tango::Util& self, Tango::DServer* dserver)
    {
        Tango::Device_var d = dserver->_this();
	dserver->set_d_var(Tango::Device::_duplicate(d));
        const char *dserver_ior = self.get_orb()->object_to_string(d);
        boost::python::str ret = dserver_ior;
        delete [] dserver_ior;
        return ret;
    }

    boost::python::str get_device_ior(Tango::Util& self, Tango::DeviceImpl* device)
    {
        const char *ior = self.get_orb()->object_to_string(device->get_d_var());
        boost::python::str ret = ior;
        delete [] ior;
        return ret;
    }

    void orb_run(Tango::Util& self)
    {
        AutoPythonAllowThreads guard;
        self.get_orb()->run();
    }

    boost::python::str get_pid_str(Tango::Util& self)
    {
        boost::python::str ret = self.get_pid_str().c_str();
        return ret;
    }

    boost::python::str get_version_str(Tango::Util& self)
    {
        boost::python::str ret = self.get_version_str().c_str();
        return ret;
    }

    static boost::shared_ptr<Tango::Util>
    makeUtil(boost::python::object& args)
    {
        Tango::Util* util = PyUtil::init(args);
        return boost::shared_ptr<Tango::Util>(util);
    }
}


BOOST_PYTHON_FUNCTION_OVERLOADS (server_init_overload, PyUtil::server_init, 1, 2)

void export_util()
{
    class_<Tango::Interceptors>("Interceptors")
        .def("create_thread", &Tango::Interceptors::create_thread)
        .def("delete_thread", &Tango::Interceptors::delete_thread)
    ;

    class_<Tango::Util, boost::noncopyable>("Util", no_init)
        .def("__init__", boost::python::make_constructor(PyUtil::makeUtil))
        .def("init", PyUtil::init,
            return_value_policy<reference_existing_object>())
        .staticmethod("init")

        .def("instance", &PyUtil::instance1,
            return_value_policy<reference_existing_object>())
        .def("instance", &PyUtil::instance2,
            return_value_policy<reference_existing_object>())
        .staticmethod("instance")

        .def("set_trace_level", &Tango::Util::set_trace_level)
        .def("get_trace_level", &Tango::Util::get_trace_level)
        .def("get_ds_inst_name", &Tango::Util::get_ds_inst_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_ds_exec_name", &Tango::Util::get_ds_exec_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_ds_name", &Tango::Util::get_ds_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_host_name", &Tango::Util::get_host_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_pid_str", &PyUtil::get_pid_str)
        .def("get_pid", &Tango::Util::get_pid)
        .def("get_tango_lib_release", &Tango::Util::get_tango_lib_release)
        .def("get_version_str", &PyUtil::get_version_str)
        .def("get_server_version", &Tango::Util::get_server_version,
            return_value_policy<copy_non_const_reference>())
        .def("set_server_version", &Tango::Util::set_server_version)
        .def("set_serial_model", &Tango::Util::set_serial_model)
        .def("get_serial_model", &Tango::Util::get_serial_model)
        .def("reset_filedatabase", &Tango::Util::reset_filedatabase)
        .def("unregister_server", &Tango::Util::unregister_server)
        .def("get_dserver_device", &Tango::Util::get_dserver_device,
            return_value_policy<reference_existing_object>())
        .def("server_init", &PyUtil::server_init, server_init_overload())
        .def("server_run", &PyUtil::server_run)
        .def("server_cleanup", &Tango::Util::server_cleanup)
        .def("trigger_cmd_polling", &Tango::Util::trigger_cmd_polling)
        .def("trigger_attr_polling", &Tango::Util::trigger_attr_polling)
        .def("set_polling_threads_pool_size", &Tango::Util::set_polling_threads_pool_size)
        .def("get_polling_threads_pool_size", &Tango::Util::get_polling_threads_pool_size)
        .def("is_svr_starting", &Tango::Util::is_svr_starting)
        .def("is_svr_shutting_down", &Tango::Util::is_svr_shutting_down)
        .def("is_device_restarting", &Tango::Util::is_device_restarting)
        .def("get_sub_dev_diag", &Tango::Util::get_sub_dev_diag,
            return_internal_reference<>())
        .def("connect_db", &Tango::Util::connect_db)
        .def("reset_filedatabase", &Tango::Util::reset_filedatabase)
        .def("get_database", &Tango::Util::get_database,
            return_internal_reference<>())
        .def("unregister_server", &Tango::Util::unregister_server)
        .def("get_device_list_by_class", &PyUtil::get_device_list_by_class)
        .def("get_device_by_name", &PyUtil::get_device_by_name)
        .def("get_device_list", &PyUtil::get_device_list)
        .def("server_set_event_loop", &PyUtil::server_set_event_loop)
        .def("set_interceptors", &Tango::Util::set_interceptors)
        .def_readonly("_UseDb", &Tango::Util::_UseDb)
        .def_readonly("_FileDb", &Tango::Util::_FileDb)
        .def("set_use_db", &PyUtil::set_use_db)
        .staticmethod("set_use_db")
        .def("get_dserver_ior", &PyUtil::get_dserver_ior)
        .def("get_device_ior", &PyUtil::get_device_ior)
        .def("orb_run", &PyUtil::orb_run)
    ;
}
