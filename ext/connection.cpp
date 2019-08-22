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


//class MyCallBack: public Tango::CallBack
//{
//public:
//    MyCallBack():cb_executed(0),to(false),cmd_failed(false) {};
//
//    virtual void cmd_ended(Tango::CmdDoneEvent *);
//
//    long cb_executed;
//    short l;
//    bool to;
//    bool cmd_failed;
//};
//
//void MyCallBack::cmd_ended(Tango::CmdDoneEvent *cmd)
//{
//    cout << "In cmd_ended method for device " << cmd->device->dev_name() << endl;
//    cout << "Command = " << cmd->cmd_name << endl;
//
//    if (cmd->errors.length() == 0)
//    {
//        cmd->argout >> l;
//        cout << "Command result = " << l << endl;
//    }
//    else
//    {
//        long nb_err = cmd->errors.length();
//        cout << "Command returns error" << endl;
//        cout << "error length = " << nb_err << endl;
//        for (int i = 0;i < nb_err;i++)
//            cout << "error[" << i << "].reason = " << cmd->errors[i].reason << endl;
//        if (strcmp(cmd->errors[nb_err - 1].reason,"API_DeviceTimedOut") == 0)
//        {
//            to = true;
//            cout << "Timeout error" << endl;
//        }
//        else if (strcmp(cmd->errors[nb_err - 1].reason,"API_CommandFailed") == 0)
//        {
//            cmd_failed = true;
//            cout << "Command failed error" << endl;
//        }
//        else
//            cout << "Unknown error" << endl;
//    }
//
//    cb_executed++;
//}

void export_connection(py::module &m) {
    py::class_<Tango::Connection>(m, "Connection", py::dynamic_attr())
        .def("get_db_host", [](Tango::Connection& self) -> std::string {
            return self.get_db_host();
        })

        .def("get_db_port", [](Tango::Connection& self) -> std::string {
            return self.get_db_port();
        })

        .def("get_db_port_num", [](Tango::Connection& self) -> long {
            return self.get_db_port_num();
        })

        .def("get_from_env_var", [](Tango::Connection& self) -> bool {
            return self.get_from_env_var();
        })

        .def("get_fqdn", [](Tango::Connection& self) -> std::string {
            std::string fqdn_str;
            self.get_fqdn(fqdn_str);
            return fqdn_str;
        })

        .def("is_dbase_used", [](Tango::Connection& self) -> bool {
            return self.is_dbase_used();
        })
        .def("get_dev_host", [](Tango::Connection& self) -> std::string {
            return self.get_dev_host();
        })

        .def("get_dev_port", [](Tango::Connection& self) -> std::string {
            return self.get_dev_port();
        })

        .def("connect", [](Tango::Connection& self, std::string name) -> void {
            py::print("connection.cpp connect");
            self.connect(name);
        })

        .def("reconnect", [](Tango::Connection& self, bool reconn) -> void {
            py::print("connection.cpp reconnect");
            self.reconnect(reconn);
        })

        .def("get_idl_version", [](Tango::Connection& self) -> int {
            return self.get_idl_version();
        })

        .def("set_timeout_millis", [](Tango::Connection& self, int timeout) -> void {
            self.set_timeout_millis(timeout);
        })

        .def("get_timeout_millis", [](Tango::Connection& self) -> int {
            return self.get_timeout_millis();
        })

        .def("get_source", [](Tango::Connection& self) -> Tango::DevSource {
            return self.get_source();
        })

        .def("set_source", [](Tango::Connection& self, Tango::DevSource src) -> void {
            self.set_source(src);
        })

        .def("get_transparency_reconnection", [](Tango::Connection& self) -> bool {
            return self.get_transparency_reconnection();
        })

        .def("set_transparency_reconnection", [](Tango::Connection& self, bool reconn) -> void {
            self.set_transparency_reconnection(reconn);
        })

        .def("__command_inout", [](Tango::Connection& self, std::string& cmd_name) -> Tango::DeviceData {
            py::print(">>>>>>>>>>>>>>>>>",cmd_name);
            AutoPythonAllowThreads guard;
            return self.command_inout(cmd_name);
        })

        .def("__command_inout", [](Tango::Connection& self, std::string& cmd_name,
                Tango::DeviceData &argin) -> Tango::DeviceData {
            py::print(">>>>>>>>>>>>>>>>>",cmd_name);
            AutoPythonAllowThreads guard;
            return self.command_inout(cmd_name, argin);
        })

        .def("__command_inout_asynch_id", [](Tango::Connection& self, std::string& cmd_name,
                Tango::DeviceData &argin, bool forget) -> long {
            AutoPythonAllowThreads guard;
            return self.command_inout_asynch(cmd_name, argin, forget);
        }, py::arg("cmd_name"), py::arg("argin"), py::arg("forget")=false)

        .def("__command_inout_asynch_id", [](Tango::Connection& self, std::string& cmd_name, bool forget) -> long {
            AutoPythonAllowThreads guard;
            return self.command_inout_asynch(cmd_name, forget);
        }, py::arg("cmd_name"), py::arg("forget")=false)

        .def("__command_inout_asynch_cb", [](Tango::Connection& self, std::string& cmd_name,
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

        .def("__command_inout_asynch_cb", [](Tango::Connection& self, std::string& cmd_name,
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
            cout << "start get async replies" << endl;
            self.get_asynch_replies();
            cout << "finish get async replies" << endl;
        })

        .def("get_asynch_replies", [](Tango::Connection& self, long timeout) -> void {
            AutoPythonAllowThreads guard;
            self.get_asynch_replies(timeout);
        })

        .def("cancel_asynch_request", [](Tango::Connection& self, long id) -> void {
            self.cancel_asynch_request(id);
        })

        .def("cancel_all_polling_asynch_request",[](Tango::Connection& self) -> void {
            self.cancel_all_polling_asynch_request();
        })
        //
        // Control access related methods
        //
        .def("get_access_control", [](Tango::Connection& self) -> Tango::AccessControlType {
            self.get_access_control();
        })
        .def("set_access_control", [](Tango::Connection& self, Tango::AccessControlType& act) -> void {
            self.set_access_control(act);
        })
        .def("get_access_right", [](Tango::Connection& self) -> Tango::AccessControlType {
            self.get_access_control();
        })
    ;
}
