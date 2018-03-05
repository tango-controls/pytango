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

void export_api_util(py::module &m) {
    py::class_<Tango::ApiUtil, std::unique_ptr<Tango::ApiUtil, py::nodelete>>(m, "ApiUtil")

        .def_static("instance", &Tango::ApiUtil::instance, "Return an instance")

        .def("pending_asynch_call", &Tango::ApiUtil::pending_asynch_call)

        .def("get_asynch_replies", [](Tango::ApiUtil& self) -> void {
            AutoPythonAllowThreads guard;
            self.get_asynch_replies();
        })

        .def("get_asynch_replies", [](Tango::ApiUtil& self, long timeout) -> void {
            AutoPythonAllowThreads guard;
            self.get_asynch_replies(timeout);
        })

        .def("set_asynch_cb_sub_model", &Tango::ApiUtil::set_asynch_cb_sub_model)
        .def("get_asynch_cb_sub_model", &Tango::ApiUtil::get_asynch_cb_sub_model)

        .def_static("get_env_var", [](std::string& name) {
            std::string value;
            return (Tango::ApiUtil::get_env_var(name.c_str(), value) == 0) ? py::str(value) : py::object();
        })

        // As a binding we should not care whether the underlying event mechanism zmq or CORBA.
        // Replace these methods with a generic method which is event mechanism agnostic.
        // It can only be one or the other and not both.
        .def("is_event_consumer_created", []() {
            // notifd is removed in version 10
            // return Tango::ApiUtil::instance()->is_notifd_event_consumer_created()
            return Tango::ApiUtil::instance()->is_zmq_event_consumer_created();
        })
        .def("get_user_connect_timeout", &Tango::ApiUtil::get_user_connect_timeout)

        .def("get_ip_from_if", &Tango::ApiUtil::get_ip_from_if)
    ;
}
