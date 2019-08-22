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
#include <pybind11/stl.h>

namespace py = pybind11;

namespace PyGroup
{

    static void __update_data_format(Tango::Group &self, Tango::GroupAttrReplyList& r)
    {
        // Usually we pass a device_proxy to "convert_to_python" in order to
        // get the data_format of the DeviceAttribute for Tango versions
        // older than 7.0. However, GroupAttrReply has no device_proxy to use!
        // So, we are using update_data_format() in here.
        // The conver_to_python method is called, without the usual
        // device_proxy argument, in PyGroupAttrReply::get_data().
        Tango::GroupAttrReplyList::iterator i, e = r.end();
        for (i=r.begin(); i != e; ++i) {
            Tango::DeviceProxy* dev_proxy = self.get_device(i->dev_name());
            if (!dev_proxy)
                continue;
//            PyDeviceAttribute::update_data_format( *dev_proxy, &(i->get_data()), 1 );
        }
    }
static long _write_attribute_asynch(Tango::Group &self, const std::string& attr_name, py::object py_value, bool forward = true,
        bool multi = false) {
    Tango::DeviceProxy* dev_proxy = self.get_device(1);
    // If !dev_proxy (no device added in self or his children) then we
    // don't initialize dev_attr. As a result, the reply will be empty.
    /// @todo or should we raise an exception instead?
    if (!dev_proxy) {
        Tango::DeviceAttribute dev_attr;
        dev_attr.set_name(attr_name.c_str());
        AutoPythonAllowThreads guard;
        return self.write_attribute_asynch(dev_attr, forward);
    }

    // Try to see if we can get attribute information from any device in
    // the group
    Tango::AttributeInfoEx attr_info;
    bool has_attr_info = false;
    {
        AutoPythonAllowThreads guard;
        for (long dev_idx = 1; dev_idx <= self.get_size(); ++dev_idx) {
            try {
                attr_info = self[dev_idx]->get_attribute_config(attr_name);
                has_attr_info = true;
                break;
            } catch (...) {
            }
        }
    }

    if (multi) {
        if (!PySequence_Check(py_value.ptr())) {
//            raise_(PyExc_TypeError, "When multi is set, value must be a python sequence "
//                    "(ex: list or tuple)");
        }

        Py_ssize_t attr_nb = py::len(py_value);
        std::vector < Tango::DeviceAttribute > dev_attr(attr_nb);

        if (has_attr_info) {
            for (Py_ssize_t i = 0; i < attr_nb; ++i) {
//                PyDeviceAttribute::reset(dev_attr[i], attr_info, py_value[i]);
            }
        } else {
            for (Py_ssize_t i = 0; i < attr_nb; ++i) {
                dev_attr[i].set_name(attr_name.c_str());
            }
        }
        AutoPythonAllowThreads guard;
        return self.write_attribute_asynch(dev_attr, forward);
    } else {
        Tango::DeviceAttribute dev_attr;
        if (has_attr_info) {
//            PyDeviceAttribute::reset(dev_attr, attr_info, py_value);
        } else {
            dev_attr.set_name(attr_name.c_str());
        }
        // If !dev_proxy (no device added in self or his children) then we
        // don't initialize dev_attr. As a result, the reply will be empty.
        /// @todo or should we raise an exception instead?

        AutoPythonAllowThreads guard;
        return self.write_attribute_asynch(dev_attr, forward);
    }
}
}

void export_group(py::module &m) {
// TODO noncopyable
    py::class_<Tango::Group, std::shared_ptr<Tango::Group> >(m, "Group")
        .def(py::init<const std::string&>())
        .def("_add", [](Tango::Group& self, const std::string& pattern, int timeout_ms=-1) {
            self.add(pattern, timeout_ms);
        })
        .def("_add", [](Tango::Group& self, std::vector<std::string> patterns, int timeout_ms=-1) {
            self.add(patterns, timeout_ms);
        })
        .def("_add", [](Tango::Group& self, std::auto_ptr<Tango::Group> grp, int timeout_ms=-1) {
            Tango::Group* grp_ptr = grp.get();
            if (grp_ptr) {
                // After adding grp_ptr into self, self is the responsible of
                // deleting grp_ptr, so we "nullify" the grp object. It's python
                // counterpart will still be available, but any method call will
                // return an exception.
                self.add(grp_ptr, timeout_ms);
                grp.release();
            } else {
//                raise_(PyExc_TypeError,
//                   "Param \"group\" is null. It probably means that it has"
//                    " already been inserted in another group." );
            }
        })
        .def("_remove", [](Tango::Group& self, const std::string&  pattern,
                bool forward = true) -> void {
            self.remove(pattern, forward);
        })
        .def("_remove", [](Tango::Group& self, std::vector<std::string> patterns, bool forward=true){
             self.remove(patterns, forward);
        })
        .def("get_group", [](Tango::Group& self, const std::string& group_name) {
            return self.get_group(group_name);
        }) // py::return_value_policy::reference_internal)

        .def("get_size", [](Tango::Group& self, bool forward=true) {
            self.get_size(forward);
        })
        .def("remove_all", &Tango::Group::remove_all)

        // GroupElement redefinitions of enable/disable. If I didn't
        // redefine them, the later Group only definitions would
        // hide the ones defined in GroupElement.
        .def("enable", &Tango::GroupElement::enable)
        .def("disable", &Tango::GroupElement::disable)
        .def("enable", [](Tango::Group& self, const std::string& dev_name, bool forward=true){
            self.enable(dev_name, forward);
        })
        .def("disable", [](Tango::Group& self, const std::string& dev_name, bool forward=true){
            self.disable(dev_name, forward);
        })
        .def("get_device_list",[](Tango::Group& self, bool forward=true) {
            return self.get_device_list(forward);
        })
        .def("command_inout_asynch", [](Tango::Group& self,
                                        const std::string& cmd_name,
                                        bool forget=false,
                                        bool forward=true) -> long {
            return self.command_inout_asynch(cmd_name, forget, forward);
        })
        .def("command_inout_asynch", [](Tango::Group& self,
                                        const std::string& cmd_name,
                                        const Tango::DeviceData& param,
                                        bool forget=false,
                                        bool forward=true) -> long {
            return self.command_inout_asynch(cmd_name, param, forget, forward);
        })
        .def("command_inout_asynch", [](Tango::Group& self,
                                         const std::string& cmd_name,
                                         const std::vector<Tango::DeviceData>& params,
                                         bool forget=false,
                                         bool forward=true) -> long {
            return self.command_inout_asynch(cmd_name, params, forget, forward);
        })
        .def("command_inout_reply", [](Tango::Group& self,
                                       long req_id,
                                       long timeout_ms=0) ->Tango::GroupCmdReplyList {
            AutoPythonAllowThreads guard;
            return self.command_inout_reply(req_id, timeout_ms);
        })
        .def("read_attribute_asynch", [](Tango::Group& self,
                                         const std::string& attr_name,
                                         bool forward=true) {
            self.read_attribute_asynch(attr_name, forward);
        })
        .def("read_attribute_reply", [](Tango::Group &self,
                                        long req_id,
                                        long timeout_ms=0) -> Tango::GroupAttrReplyList {
            Tango::GroupAttrReplyList r;
            AutoPythonAllowThreads guard;
            r = self.read_attribute_reply(req_id, timeout_ms);
            PyGroup::__update_data_format(self, r);
            return r;
        })
        .def("read_attributes_asynch",[](Tango::Group& self,
                                         py::list attr_names,
                                         bool forward=true) {
            std::vector<std::string> r = attr_names.cast<std::vector<std::string>>();
            return self.read_attributes_asynch(r, forward);
        })
        .def("read_attributes_reply", [](Tango::Group& self,
                                         long req_id,
                                         long timeout_ms=0) {
            Tango::GroupAttrReplyList r;
            AutoPythonAllowThreads guard;
            r = self.read_attributes_reply(req_id, timeout_ms);
            PyGroup::__update_data_format(self, r);
            return r;
        })
        .def("write_attribute_asynch", [](Tango::Group &self, 
                                          const std::string& attr_name,
                                          py::object py_value,
                                          bool forward=true,
                                          bool multi=false) {
            PyGroup::_write_attribute_asynch(self, attr_name, py_value, forward, multi);
        })
        .def("write_attribute_reply", [](Tango::Group &self,
                                         long req_id,
                                         long timeout_ms = 0) -> Tango::GroupReplyList {
            AutoPythonAllowThreads guard;
            return self.write_attribute_reply(req_id, timeout_ms);
        })
        .def("get_parent", &Tango::Group::get_parent,
             py::return_value_policy::reference_internal)
        .def("contains", [](Tango::Group& self, const std::string& pattern, bool forward=true) {
            self.contains(pattern, forward);
        })
        .def("get_device",
            (Tango::DeviceProxy* (Tango::Group::*) (const std::string& ))
            &Tango::Group::get_device,
            (py::arg("self"), py::arg("dev_name")),
            py::return_value_policy::reference_internal)
        .def("get_device",
            (Tango::DeviceProxy* (Tango::Group::*) (long))
            &Tango::Group::get_device,
            (py::arg("self"), py::arg("idx")),
            py::return_value_policy::reference_internal)
        .def("ping", [](Tango::Group& self, bool forward=true) {
            self.ping(forward);
        })
        .def("set_timeout_millis", &Tango::Group::set_timeout_millis,
            (py::arg("self"), py::arg("timeout_ms")))
        .def("get_name", &Tango::Group::get_name,
            py::return_value_policy::copy)
        .def("get_fully_qualified_name", &Tango::Group::get_fully_qualified_name)
        .def("enable", &Tango::Group::enable)
        .def("disable", &Tango::Group::disable)
        .def("is_enabled", &Tango::Group::is_enabled)
        .def("name_equals", &Tango::Group::name_equals)
        .def("name_matches", &Tango::Group::name_matches)
    ;
}
