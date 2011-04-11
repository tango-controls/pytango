#include <boost/python.hpp>
#include <tango.h>

#include "to_py.h"
#include "from_py.h"

using namespace boost::python;

namespace PyDServer
{
    PyObject* query_class(Tango::DServer &self)
    {
        Tango::DevVarStringArray *res = self.query_class();
        PyObject *py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
        delete res;
        return py_res;
    }
    
    PyObject* query_device(Tango::DServer &self)
    {
        Tango::DevVarStringArray *res = self.query_device();
        PyObject *py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
        delete res;
        return py_res;
    }
    
    PyObject* query_sub_device(Tango::DServer &self)
    {
        Tango::DevVarStringArray *res = self.query_sub_device();
        PyObject *py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
        delete res;
        return py_res;
    }
    
    PyObject* query_class_prop(Tango::DServer &self, const std::string &class_name)
    {
        std::string class_name2 = class_name;
        Tango::DevVarStringArray *res = self.query_class_prop(class_name2);
        PyObject *py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
        delete res;
        return py_res;
    }

    PyObject* query_dev_prop(Tango::DServer &self, const std::string &dev_name)
    {
        std::string dev_name2 = dev_name;
        Tango::DevVarStringArray *res = self.query_dev_prop(dev_name2);
        PyObject *py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
        delete res;
        return py_res;
    }
    
    PyObject* polled_device(Tango::DServer &self)
    {
        Tango::DevVarStringArray *res = self.polled_device();
        PyObject *py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
        delete res;
        return py_res;
    }
    
    PyObject* dev_poll_status(Tango::DServer &self, const std::string &dev_name)
    {
        std::string dev_name2 = dev_name;
        Tango::DevVarStringArray *res = self.dev_poll_status(dev_name2);
        PyObject *py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
        delete res;
        return py_res;
    }
    
    void add_obj_polling(Tango::DServer &self, object &py_long_str_array, bool with_db_upd = true, int delta_ms = 0)
    {
        Tango::DevVarLongStringArray long_str_array;
        convert2array(py_long_str_array, long_str_array);
        self.add_obj_polling(&long_str_array, with_db_upd, delta_ms);
    }
    
    void upd_obj_polling_period(Tango::DServer &self, object &py_long_str_array, bool with_db_upd = true)
    {
        Tango::DevVarLongStringArray long_str_array;
        convert2array(py_long_str_array, long_str_array);
        self.upd_obj_polling_period(&long_str_array, with_db_upd);
    }
    
    void rem_obj_polling(Tango::DServer &self, object &py_str_array, bool with_db_upd = true)
    {
        Tango::DevVarStringArray str_array;
        convert2array(py_str_array, str_array);
        self.rem_obj_polling(&str_array, with_db_upd);
    }
    
    void lock_device(Tango::DServer &self, object &py_long_str_array)
    {
        Tango::DevVarLongStringArray long_str_array;
        convert2array(py_long_str_array, long_str_array);
        self.lock_device(&long_str_array);
    }
    
    Tango::DevLong un_lock_device(Tango::DServer &self, object &py_long_str_array)
    {
        Tango::DevVarLongStringArray long_str_array;
        convert2array(py_long_str_array, long_str_array);
        return self.un_lock_device(&long_str_array);
    }
    
    void re_lock_devices(Tango::DServer &self, object &py_str_array)
    {
        Tango::DevVarStringArray str_array;
        convert2array(py_str_array, str_array);
        self.re_lock_devices(&str_array);
    }
    
    PyObject* dev_lock_status(Tango::DServer &self, Tango::ConstDevString dev_name)
    {
        Tango::DevVarLongStringArray* ret = self.dev_lock_status(dev_name);
        PyObject* py_ret = 
            CORBA_sequence_to_list<Tango::DevVarLongStringArray>::convert(*ret);
        delete ret;
        return py_ret;
    }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(add_obj_polling_overload, PyDServer::add_obj_polling, 2, 4)
BOOST_PYTHON_FUNCTION_OVERLOADS(upd_obj_polling_period_overload, PyDServer::upd_obj_polling_period, 2, 3)
BOOST_PYTHON_FUNCTION_OVERLOADS(rem_obj_polling_overload, PyDServer::rem_obj_polling, 2, 3)

void export_dserver()
{
    // The following function declarations are necessary to be able to cast
    // the function parameters from string& to const string&, otherwise python
    // will not recognize the method calls

    void (Tango::DServer::*restart_)(std::string &) = &Tango::DServer::restart;

    class_<Tango::DServer,
        bases<Tango::Device_4Impl>, boost::noncopyable>
        ("DServer", no_init)
        .def("query_class",  &PyDServer::query_class)
        .def("query_device",  &PyDServer::query_device)
        .def("query_sub_device",  &PyDServer::query_sub_device)
        .def("kill", &Tango::DServer::kill)
        .def("restart", 
            (void (Tango::DServer::*) (const std::string &)) restart_)
        .def("restart_server", &Tango::DServer::restart_server)
        .def("query_class_prop", &PyDServer::query_class_prop)
        .def("query_dev_prop", &PyDServer::query_dev_prop)
        .def("polled_device", &PyDServer::polled_device)
        .def("dev_poll_status", &PyDServer::polled_device)
        .def("add_obj_polling", &PyDServer::add_obj_polling, 
            add_obj_polling_overload())
        .def("upd_obj_polling_period", &PyDServer::upd_obj_polling_period, 
            upd_obj_polling_period_overload())
        .def("rem_obj_polling", &PyDServer::rem_obj_polling, 
            rem_obj_polling_overload())
        .def("stop_polling", &Tango::DServer::stop_polling)
        .def("start_polling", 
            (void (Tango::DServer::*)() ) &Tango::DServer::start_polling)
        .def("add_event_heartbeat", &Tango::DServer::add_event_heartbeat)
        .def("rem_event_heartbeat", &Tango::DServer::rem_event_heartbeat)
        .def("lock_device", &PyDServer::lock_device)
        .def("un_lock_device", &PyDServer::un_lock_device)
        .def("re_lock_devices", &PyDServer::re_lock_devices)
        .def("dev_lock_status", &PyDServer::dev_lock_status)
        .def("delete_devices", &Tango::DServer::delete_devices)
        .def("start_logging", &Tango::DServer::start_logging)
        .def("stop_logging", &Tango::DServer::stop_logging)
        .def("get_process_name", &Tango::DServer::get_process_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_personal_name", &Tango::DServer::get_personal_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_instance_name", &Tango::DServer::get_instance_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_full_name", &Tango::DServer::get_full_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_fqdn", &Tango::DServer::get_fqdn,
            return_value_policy<copy_non_const_reference>())
        .def("get_poll_th_pool_size", &Tango::DServer::get_poll_th_pool_size)
        .def("get_opt_pool_usage", &Tango::DServer::get_opt_pool_usage)
        .def("get_poll_th_conf", &Tango::DServer::get_poll_th_conf)
    ;
    
}
