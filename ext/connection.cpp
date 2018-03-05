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
#include "pytgutils.h"
#include "pyutils.h"
#include "callback.h"

namespace py = pybind11;


void export_connection(py::module &m) {
    py::class_<Tango::Connection>(m, "Connection", py::dynamic_attr())
        .def("get_db_host", &Tango::Connection::get_db_host, py::return_value_policy::copy)
        .def("get_db_port", &Tango::Connection::get_db_port, py::return_value_policy::copy)
        .def("get_db_port_num", &Tango::Connection::get_db_port_num)
        .def("get_from_env_var", &Tango::Connection::get_from_env_var)
        .def_static("get_fqdn", &Tango::Connection::get_fqdn)
        .def("is_dbase_used", &Tango::Connection::is_dbase_used)
        .def("get_dev_host", &Tango::Connection::get_dev_host, py::return_value_policy::copy)
        .def("get_dev_port", &Tango::Connection::get_dev_port, py::return_value_policy::copy)
        .def("connect", &Tango::Connection::connect)
        .def("reconnect", &Tango::Connection::reconnect)
        .def("get_idl_version", &Tango::Connection::get_idl_version)
        .def("set_timeout_millis", &Tango::Connection::set_timeout_millis)
        .def("get_timeout_millis", &Tango::Connection::get_timeout_millis)
        .def("get_source", &Tango::Connection::get_source)
        .def("set_source", &Tango::Connection::set_source)
        .def("get_transparency_reconnection", &Tango::Connection::get_transparency_reconnection)
        .def("set_transparency_reconnection", &Tango::Connection::set_transparency_reconnection)

        .def("__command_inout", [](Tango::Connection& self, std::string& cmd_name) -> Tango::DeviceData {
            AutoPythonAllowThreads guard;
            return self.command_inout(cmd_name);
        })

        .def("__command_inout", [](Tango::Connection& self, std::string& cmd_name,
                Tango::DeviceData &argin) -> Tango::DeviceData {
            AutoPythonAllowThreads guard;
            return self.command_inout(cmd_name, argin);
        })

        .def("__command_inout_asynch_id", [](Tango::Connection& self, string& cmd_name,
                Tango::DeviceData &argin, bool forget) -> long {
            AutoPythonAllowThreads guard;
            return self.command_inout_asynch(cmd_name, argin, forget);
        }, py::arg("cmd_name"), py::arg("argin"), py::arg("forget")=false)

        .def("__command_inout_asynch_id", [](Tango::Connection& self, string& cmd_name, bool forget) -> long {
            AutoPythonAllowThreads guard;
            return self.command_inout_asynch(cmd_name, forget);
        }, py::arg("cmd_name"), py::arg("forget")=false)

        .def("__command_inout_asynch_cb", [](Tango::Connection& self, std::string& cmd_name,
                py::object py_cb) -> void {
//            PyCallBackAutoDie* cb = py_cb.cast<PyCallBackAutoDie*>();
//            cb->set_autokill_references(py_cb, py_self);
//            try {
                AutoPythonAllowThreads guard;
//                self->command_inout_asynch(cmd_name, *cb);
//            } catch (...) {
//                cb->unset_autokill_references();
//                throw;
//            }
        })

        .def("__command_inout_asynch_cb", [](Tango::Connection& self, std::string& cmd_name,
                Tango::DeviceData &argin, py::object py_cb) -> void {
//            PyCallBackAutoDie* cb = extract<PyCallBackAutoDie*>(py_cb);
//            cb->set_autokill_references(py_cb, py_self);
//            try {
                AutoPythonAllowThreads guard;
//                self->command_inout_asynch(cmd_name, argin, *cb);
//            } catch (...) {
//                cb->unset_autokill_references();
//                throw;
//            }
        })

        .def("command_inout_reply_raw", [](Tango::Connection& self, long id) -> Tango::DeviceData {
            AutoPythonAllowThreads guard;
            return self.command_inout_reply(id);
        })

        .def("command_inout_reply_raw", [](Tango::Connection& self, long id, long timeout) -> Tango::DeviceData {
            AutoPythonAllowThreads guard;
            return self.command_inout_reply(id, timeout);
        })

        //
        // Asynchronous methods
        //
        .def("get_asynch_replies", [](Tango::Connection& self) -> void {
            AutoPythonAllowThreads guard;
            self.get_asynch_replies();
        })

        .def("get_asynch_replies", [](Tango::Connection& self, long timeout) -> void{
            AutoPythonAllowThreads guard;
            self.get_asynch_replies(timeout);
        })

        .def("cancel_asynch_request", &Tango::Connection::cancel_asynch_request)
        .def("cancel_all_polling_asynch_request",
            &Tango::Connection::cancel_all_polling_asynch_request)
        //
        // Control access related methods
        //
        .def("get_access_control", &Tango::Connection::get_access_control)
        .def("set_access_control", &Tango::Connection::set_access_control)
        .def("get_access_right", &Tango::Connection::get_access_right);
}
