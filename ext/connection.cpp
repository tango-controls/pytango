/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "pytgutils.h"
#include "callback.h"

namespace PyConnection
{

    using namespace boost::python;

    static
    Tango::DeviceData command_inout(Tango::Connection& self, const std::string & cmd_name, const Tango::DeviceData &argin)
    {
        AutoPythonAllowThreads guard;
        return self.command_inout(const_cast<std::string&>(cmd_name), const_cast<Tango::DeviceData&>(argin));
    }


    static
    long command_inout_asynch_id(Tango::Connection& self, const std::string &cmd_name, const Tango::DeviceData &argin, bool forget)
    {
        AutoPythonAllowThreads guard;
        return self.command_inout_asynch(const_cast<std::string&>(cmd_name), const_cast<Tango::DeviceData&>(argin), forget);
    }


    static
    Tango::DeviceData command_inout_reply(Tango::Connection& self, long id)
    {
        AutoPythonAllowThreads guard;
        return self.command_inout_reply(id);
    }

    static
    Tango::DeviceData command_inout_reply(Tango::Connection& self, long id, long timeout)
    {
        AutoPythonAllowThreads guard;
        return self.command_inout_reply(id, timeout);
    }

    static
    void command_inout_asynch_cb(object py_self, const std::string & cmd_name, const Tango::DeviceData &argin, object py_cb)
    {
        Tango::Connection* self = extract<Tango::Connection*>(py_self);
        PyCallBackAutoDie* cb = extract<PyCallBackAutoDie*>(py_cb);
        cb->set_autokill_references(py_cb, py_self);

        try {
            AutoPythonAllowThreads guard;
            self->command_inout_asynch(const_cast<std::string&>(cmd_name), const_cast<Tango::DeviceData&>(argin), *cb);
        } catch (...) {
            cb->unset_autokill_references();
            throw;
        }
    }

    static
    void get_asynch_replies(Tango::Connection& self)
    {
        AutoPythonAllowThreads guard;
        self.get_asynch_replies();
    }

    static
    void get_asynch_replies(Tango::Connection& self, long call_timeout)
    {
        AutoPythonAllowThreads guard;
        self.get_asynch_replies(call_timeout);
    }
    
    str get_fqdn()
    {
        std::string fqdn;
        Tango::Connection::get_fqdn(fqdn);
        return str(fqdn.c_str());
    }
}

void export_connection()
{
    using namespace boost::python;

    class_<Tango::Connection, boost::noncopyable> Connection("Connection", no_init)
    ;

    Connection
        .def("dev_name", pure_virtual(&Tango::Connection::dev_name))
        .def("get_db_host", &Tango::Connection::get_db_host,
            return_value_policy<copy_non_const_reference>())
        .def("get_db_port", &Tango::Connection::get_db_port,
            return_value_policy<copy_non_const_reference>())
        .def("get_db_port_num", &Tango::Connection::get_db_port_num)
        .def("get_from_env_var", &Tango::Connection::get_from_env_var)
        .def("get_fqdn", &PyConnection::get_fqdn)
        .staticmethod("get_fqdn")
        .def("is_dbase_used", &Tango::Connection::is_dbase_used)
        .def("get_dev_host", &Tango::Connection::get_dev_host,
            return_value_policy<copy_non_const_reference>())
        .def("get_dev_port", &Tango::Connection::get_dev_port,
            return_value_policy<copy_non_const_reference>())
        .def("connect", &Tango::Connection::connect)
        .def("reconnect", &Tango::Connection::reconnect)
        .def("get_idl_version", &Tango::Connection::get_idl_version)
        .def("set_timeout_millis", &Tango::Connection::set_timeout_millis)
        .def("get_timeout_millis", &Tango::Connection::get_timeout_millis)
        .def("get_source", &Tango::Connection::get_source)
        .def("set_source", &Tango::Connection::set_source)
        .def("get_transparency_reconnection",
            &Tango::Connection::get_transparency_reconnection)
        .def("set_transparency_reconnection",
            &Tango::Connection::set_transparency_reconnection)
        .def("__command_inout", &PyConnection::command_inout)
        .def("__command_inout_asynch_id", &PyConnection::command_inout_asynch_id)
        .def("__command_inout_asynch_cb", &PyConnection::command_inout_asynch_cb)
        .def("command_inout_reply_raw",
            (Tango::DeviceData (*)(Tango::Connection&, long))
            &PyConnection::command_inout_reply)
        .def("command_inout_reply_raw",
            (Tango::DeviceData (*)(Tango::Connection&, long, long))
            &PyConnection::command_inout_reply)

        //
        // Asynchronous methods
        //

        .def("get_asynch_replies",
            (void (*) (Tango::Connection&)) &PyConnection::get_asynch_replies)
        .def("get_asynch_replies",
            (void (*) (Tango::Connection&,long)) &PyConnection::get_asynch_replies)
        .def("cancel_asynch_request", &Tango::Connection::cancel_asynch_request)
        .def("cancel_all_polling_asynch_request",
            &Tango::Connection::cancel_all_polling_asynch_request)

        //
        // Control access related methods
        //
        
        .def("get_access_control", &Tango::Connection::get_access_control)
        .def("set_access_control", &Tango::Connection::set_access_control)
        .def("get_access_right", &Tango::Connection::get_access_right)
    ;
}
