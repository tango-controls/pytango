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
#include "to_py.h"
#include "from_py.h"

namespace py = pybind11;

void export_dserver(py::module &m) {
    py::class_<Tango::DServer>(m, "DServer")
        .def("query_class", [](Tango::DServer& self) -> py::list {
            Tango::DevVarStringArray *res = self.query_class();
            py::list py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
            delete res;
            return py_res;
        })
        .def("query_device", [](Tango::DServer& self) -> py::list {
            Tango::DevVarStringArray *res = self.query_device();
            py::list py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
            delete res;
            return py_res;
        })
        .def("query_sub_device", [](Tango::DServer& self) -> py::list {
            Tango::DevVarStringArray *res = self.query_sub_device();
            py::list py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
            delete res;
            return py_res;
        })
        .def("kill", [](Tango::DServer& self) -> void {
            self.kill();
        })
        .def("restart", [](Tango::DServer& self, std::string& serv) -> void {
            self.restart(serv);
        })
        .def("restart_server", [](Tango::DServer& self) -> void {
            self.restart_server();
        })
        .def("query_class_prop", [](Tango::DServer& self, const std::string& class_name) -> py::list {
            std::string class_name2 = class_name;
            Tango::DevVarStringArray *res = self.query_class_prop(class_name2);
            py::list py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
            delete res;
            return py_res;
        })
        .def("query_dev_prop", [](Tango::DServer& self, const std::string& dev_name) -> py::list {
            std::string dev_name2 = dev_name;
            Tango::DevVarStringArray *res = self.query_dev_prop(dev_name2);
            py::list py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
            delete res;
            return py_res;
        })
        .def("polled_device", [](Tango::DServer& self) -> py::list {
            Tango::DevVarStringArray *res = self.polled_device();
            py::list py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
            delete res;
            return py_res;
        })
        .def("dev_poll_status", [](Tango::DServer& self, const std::string& dev_name) -> py::list {
            std::string dev_name2 = dev_name;
            Tango::DevVarStringArray *res = self.dev_poll_status(dev_name2);
            py::list py_res = CORBA_sequence_to_list<Tango::DevVarStringArray>::convert(*res);
            delete res;
            return py_res;
        })
        .def("add_obj_polling", [](Tango::DServer& self, py::object &py_long_str_array, bool with_db_upd, int delta_ms) -> void {
            Tango::DevVarLongStringArray long_str_array;
            convert2array(py_long_str_array, long_str_array);
            self.add_obj_polling(&long_str_array, with_db_upd, delta_ms);
        }, py::arg("py_long_str_array"), py::arg("with_db_upd")=true, py::arg("delta_ms")=0)

        .def("upd_obj_polling_period", [](Tango::DServer& self, py::object &py_long_str_array, bool with_db_upd = true) -> void {
            Tango::DevVarLongStringArray long_str_array;
            convert2array(py_long_str_array, long_str_array);
            self.upd_obj_polling_period(&long_str_array, with_db_upd);
        }, py::arg("py_long_str_array"), py::arg("with_db_upd")=true)

        .def("rem_obj_polling", [](Tango::DServer& self, py::object &py_str_array, bool with_db_upd) -> void {
            Tango::DevVarStringArray str_array;
            convert2array(py_str_array, str_array);
            self.rem_obj_polling(&str_array, with_db_upd);
        }, py::arg("py_str_array"), py::arg("with_db_upd")=true)

        .def("stop_polling", [](Tango::DServer& self) -> void {
            self.stop_polling();
        })
        .def("start_polling", [](Tango::DServer& self) -> void {
            self.start_polling();
        })
        .def("start_polling", [](Tango::DServer& self, Tango::PollingThreadInfo* info) -> void {
            self.start_polling(info);
        })
        .def("add_event_heartbeat", [](Tango::DServer& self) -> void {
            self.add_event_heartbeat();
        })
        .def("rem_event_heartbeat", [](Tango::DServer& self) -> void {
            self.rem_event_heartbeat();
        })
        .def("lock_device", [](Tango::DServer& self, py::object &py_long_str_array) -> void {
            Tango::DevVarLongStringArray long_str_array;
            convert2array(py_long_str_array, long_str_array);
            self.lock_device(&long_str_array);
        })
        .def("un_lock_device", [](Tango::DServer& self, py::object &py_long_str_array) -> void {
            Tango::DevVarLongStringArray long_str_array;
            convert2array(py_long_str_array, long_str_array);
            self.un_lock_device(&long_str_array);
        })
        .def("re_lock_devices", [](Tango::DServer& self, py::object &py_str_array) -> void {
            Tango::DevVarStringArray str_array;
            convert2array(py_str_array, str_array);
            self.re_lock_devices(&str_array);
        })
        .def("dev_lock_status", [](Tango::DServer& self, Tango::ConstDevString dev_name) -> py::list {
            Tango::DevVarLongStringArray* ret = self.dev_lock_status(dev_name);
            py::list py_ret = CORBA_sequence_to_list<Tango::DevVarLongStringArray>::convert(*ret);
            delete ret;
            return py_ret;
        })
        .def("delete_devices", [](Tango::DServer& self) -> void {
            self.delete_devices();
        })
        .def("start_logging", [](Tango::DServer& self) -> void {
            self.start_logging();
        })
        .def("stop_logging", [](Tango::DServer& self) -> void {
            self.stop_logging();
        })
        .def("get_process_name", [](Tango::DServer& self) -> std::string {
            return self.get_process_name();
        })
        .def("get_personal_name", [](Tango::DServer& self) -> std::string {
            return self.get_personal_name();
        })
        .def("get_instance_name", [](Tango::DServer& self) -> std::string {
            return self.get_instance_name();
        })
        .def("get_full_name", [](Tango::DServer& self) -> std::string {
            return self.get_full_name();
        })
        .def("get_fqdn", [](Tango::DServer& self) -> std::string {
            return self.get_fqdn();
        })
        .def("get_poll_th_pool_size", [](Tango::DServer& self) -> long {
            return self.get_poll_th_pool_size();
        })
        .def("get_opt_pool_usage", [](Tango::DServer& self) -> bool {
            return self.get_opt_pool_usage();
        })
        .def("get_poll_th_conf", [](Tango::DServer& self) -> std::vector<std::string> {
            return self.get_poll_th_conf();
        })
    ;
    
}
