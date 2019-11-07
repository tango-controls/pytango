/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <iostream>
#include <memory>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "exception.h"
#include "server/device_class.h"
#include "server/attr.h"
#include "server/command.h"
#include "server/pipe.h"
#include "pyutils.h"

namespace py = pybind11;

DeviceClassWrap::DeviceClassWrap(std::string& name, py::object py_self)
    :Tango::DeviceClass(name)
{
    m_self = py_self;
    AutoPythonGIL python_guard;
    signal_handler_defined = is_method_defined(m_self, "signal_handler");
}

DeviceClassWrap::~DeviceClassWrap()
{}

void DeviceClassWrap::create_command(std::string& cmd_name,
                                     Tango::CmdArgType param_type,
                                     Tango::CmdArgType result_type,
                                     std::string& param_desc,
                                     std::string& result_desc,
                                     Tango::DispLevel display_level,
                                     bool default_command,
                                     long polling_period,
                                     const std::string& is_allowed)
{
    PyCmd *cmd_ptr = new PyCmd(cmd_name, param_type, result_type,
                               param_desc, result_desc, display_level);

    if (!is_allowed.empty())
    {
        cmd_ptr->set_allowed(is_allowed);
    }

    if (polling_period > 0)
        cmd_ptr->set_polling_period(polling_period);
    if (default_command)
        set_default_command(cmd_ptr);
    else
        command_list.push_back(cmd_ptr);
}

void DeviceClassWrap::create_fwd_attribute(
        const std::string& attr_name, Tango::UserDefaultFwdAttrProp *att_prop)
{
    Tango::FwdAttr* attr_ptr = new Tango::FwdAttr(attr_name);
    attr_ptr->set_default_properties(*att_prop);
    this->local_attr_list->push_back(attr_ptr);
}

void DeviceClassWrap::create_attribute(
        const std::string& attr_name,
        Tango::CmdArgType attr_type,
        Tango::AttrDataFormat attr_format,
        Tango::AttrWriteType attr_write,
        long dim_x, long dim_y,
        Tango::DispLevel display_level,
        long polling_period,
        bool memorized, bool hw_memorized,
        const std::string& read_method_name,
        const std::string& write_method_name,
        const std::string& is_allowed_name,
        Tango::UserDefaultAttrProp *att_prop)
{
    //
    // Create the attribute object according to attribute format
    //
    PyScaAttr *sca_attr_ptr = nullptr;
    PySpecAttr *spec_attr_ptr = nullptr;
    PyImaAttr *ima_attr_ptr= nullptr;
    PyAttr *py_attr_ptr = nullptr;
    Tango::Attr *attr_ptr = nullptr;

    switch (attr_format)
    {
        case Tango::SCALAR:
            sca_attr_ptr = new PyScaAttr(attr_name, attr_type, attr_write);
            py_attr_ptr = sca_attr_ptr;
            attr_ptr = sca_attr_ptr;
            break;
        case Tango::SPECTRUM:
            spec_attr_ptr = new PySpecAttr(attr_name.c_str(), attr_type,
                                           attr_write, dim_x);
            py_attr_ptr = spec_attr_ptr;
            attr_ptr = spec_attr_ptr;
            break;
        case Tango::IMAGE:
            ima_attr_ptr = new PyImaAttr(attr_name.c_str(), attr_type,
                                         attr_write, dim_x, dim_y);
            py_attr_ptr = ima_attr_ptr;
            attr_ptr = ima_attr_ptr;
            break;
        default:
            std::stringstream o;
            o << "Attribute " << attr_name << " has an unexpected data format\n"
              << "Please report this bug to the PyTango development team"
              << ends;
            Tango::Except::throw_exception(
                    (const char *)"PyDs_UnexpectedAttributeFormat",
                    o.str(),
                    (const char *)"create_attribute");
            break;
    }

    py_attr_ptr->set_read_name(read_method_name);
    py_attr_ptr->set_write_name(write_method_name);
    py_attr_ptr->set_allowed_name(is_allowed_name);

    if (att_prop) {
        attr_ptr->set_default_properties(*att_prop);
    }
    attr_ptr->set_disp_level(display_level);
    if (memorized)
    {
        attr_ptr->set_memorized();
        attr_ptr->set_memorized_init(hw_memorized);
    }

    if (polling_period > 0)
        attr_ptr->set_polling_period(polling_period);

    this->local_attr_list->push_back(attr_ptr);
}

void DeviceClassWrap::create_pipe(const std::string& name,
            Tango::PipeWriteType access,
            Tango::DispLevel display_level,
            const std::string& read_method_name,
            const std::string& write_method_name,
            const std::string& is_allowed_name,
            Tango::UserDefaultPipeProp *prop)
{
    Tango::Pipe *pipe_ptr = nullptr;
    if(access == Tango::PIPE_READ)
    {
        PyTango::Pipe::PyPipe* py_pipe_ptr = new PyTango::Pipe::PyPipe(name, display_level, access);
        py_pipe_ptr->set_read_name(read_method_name);
        py_pipe_ptr->set_allowed_name(is_allowed_name);
        pipe_ptr = py_pipe_ptr;
    }
    else
    {
        PyTango::Pipe::PyWPipe* py_pipe_ptr = new PyTango::Pipe::PyWPipe(name, display_level);
        py_pipe_ptr->set_read_name(read_method_name);
        py_pipe_ptr->set_allowed_name(is_allowed_name);
        py_pipe_ptr->set_write_name(write_method_name);
        pipe_ptr = py_pipe_ptr;
    }
    if (prop) {
        pipe_ptr->set_default_properties(*prop);
    }
    this->local_pipe_list->push_back(pipe_ptr);
}

void DeviceClassWrap::attribute_factory(std::vector<Tango::Attr *>& attr_list)
{
    AutoPythonGIL python_guard;
    try
    {
        this->local_attr_list = &attr_list;
        m_self.attr("_attribute_factory")();
    }
    catch(py::error_already_set &eas)
    {
        handle_python_exception(eas);
    }
}

void DeviceClassWrap::pipe_factory()
{
    AutoPythonGIL python_guard;
    try
    {
        this->local_pipe_list = &pipe_list;
        m_self.attr("_pipe_factory")();
    }
    catch(py::error_already_set &eas)
    {
        std::cerr << eas.what() << std::endl;
        handle_python_exception(eas);
    }
}

void DeviceClassWrap::command_factory()
{
    AutoPythonGIL python_guard;
    try {
        m_self.attr("_command_factory")();
    }
    catch(py::error_already_set &eas) \
    {
        handle_python_exception(eas);
    }
}

void DeviceClassWrap::device_name_factory(std::vector<std::string> &dev_list)
{
    AutoPythonGIL python_guard;
    try
    {
        m_self.attr("device_name_factory")(dev_list);
    }
    catch(py::error_already_set &eas)
    {
        handle_python_exception(eas);
    }
}

void DeviceClassWrap::device_factory(const Tango::DevVarStringArray *dev_list)
{
    AutoPythonGIL python_guard;
    try {
        py::list py_dev_list;
        for(auto i = 0; i < dev_list->length(); ++i) {
            py_dev_list.append((*dev_list)[i].in());
        }
        m_self.attr("device_factory")(py_dev_list);
    }
    catch(py::error_already_set &eas)
    {
        handle_python_exception(eas);
    }
}

void DeviceClassWrap::signal_handler(long signo)
{
    if (signal_handler_defined == true)
    {
        AutoPythonGIL python_guard;
        try {
            m_self.attr("signal_handler")(signo);
        }
        catch(py::error_already_set &eas) \
        {
            handle_python_exception(eas);
        }
    }
    else
    {
        Tango::DeviceClass::signal_handler(signo);
    }
}

//void DeviceClassWrap::default_signal_handler(long signo)
//{
////    this->Tango::DeviceClass::signal_handler(signo);
//}

void DeviceClassWrap::delete_class()
{
    AutoPythonGIL python_guard;
    try
    {
        //
        // Call the delete_class_list function in order to clear the global
        // constructed class Python list. It is MANDATORY to destroy these objects
        // from Python. Otherwise, there are "seg fault" when Python exit.
        // It tooks me quite a long time to find this...
        py::object tango = py::cast<py::object>(PyImport_AddModule("tango"));
        py::list cpp_class_list = tango.attr("delete_class_list")();
    }
    catch(py::error_already_set &eas)
    {
        handle_python_exception(eas);
    }
}

void export_device_class(py::module &m)
{
    py::class_<DeviceClassWrap>(m, "DeviceClass")
        .def(py::init([](std::string name, py::object py_self) {
            DeviceClassWrap* cpp = new DeviceClassWrap(name, py_self);
            return cpp;
        }))
        .def("device_factory", [](DeviceClassWrap& self, const Tango::DevVarStringArray *dev_list) {
            self.device_factory(dev_list);
        })
        .def("device_name_factory", [](DeviceClassWrap& self, std::vector<std::string> &dev_list) {
            self.device_name_factory(dev_list);
        })
        .def("export_device", [](DeviceClassWrap& self, DeviceImplWrap *dev, std::string& dev_name) -> void {
            self.export_device(dev, const_cast<char*>(dev_name.c_str()));
        }, py::arg("dev"), py::arg("dev_name")="Unused")

        .def("_add_device", [](DeviceClassWrap& self, DeviceImplWrap *dev) -> void {
            self.add_device(dev);
        })
        .def("register_signal", [](DeviceClassWrap& self, long signo) -> void {
            self.register_signal(signo);
        })
#if defined __linux
        .def("register_signal", [](DeviceClassWrap& self, long signo, bool own_handler) -> void {
            return self.register_signal(signo, own_handler);
        }, py::arg("signo"), py::arg("own_handler")=false)
#else
        .def("register_signal", [](DeviceClassWrap& self, long signo) -> void {
            self.register_signal(signo);
        })
#endif
        .def("signal_handler", [](DeviceClassWrap& self, long signo) {
            self.signal_handler(signo);
        })
        .def("unregister_signal", [](DeviceClassWrap& self, long signo) -> void {
            self.unregister_signal(signo);
        })
        .def("get_name", [](DeviceClassWrap& self) -> std::string& {
            return self.get_name();
        })
        .def("get_type", [](DeviceClassWrap& self) -> std::string {
            return self.get_type();
        })
        .def("get_doc_url", [](DeviceClassWrap& self) -> std::string {
            return self.get_doc_url();
        })
        .def("get_cvs_tag", [](DeviceClassWrap& self) -> std::string {
            return self.get_cvs_tag();
        })
        .def("get_cvs_location", [](DeviceClassWrap& self) -> std::string {
            return self.get_cvs_location();
        })
        .def("get_device_list", [](DeviceClassWrap& self) -> py::list {
            py::list py_dev_list;
            std::vector<Tango::DeviceImpl *> dev_list = self.get_device_list();
            for(std::vector<Tango::DeviceImpl *>::iterator it = dev_list.begin();
                it != dev_list.end(); ++it)
            {
                py::object py_value = py::cast(*it);
                py_dev_list.append(py_value);
            }
            // Could this be?
            // return py::cast(self.get_device_list())
            return py_dev_list;
        })
        .def("get_command_list", [](DeviceClassWrap& self) -> py::list {
            py::list py_cmd_list;
            std::vector<Tango::Command *> cmd_list = self.get_command_list();
            for(std::vector<Tango::Command *>::iterator it = cmd_list.begin();
                    it != cmd_list.end(); ++it)
            {
                py::object py_value = py::cast(*it);
                py_cmd_list.append(py_value);
            }
            // Could this be?
            // return py::cast(self.get_command_list())
            return py_cmd_list;
        })
        .def("get_pipe_list", [](DeviceClassWrap& self, const std::string& dev_name) -> py::list {
            py::list py_pipe_list;
            std::vector<Tango::Pipe *> pipe_list = self.get_pipe_list(dev_name);
            for(std::vector<Tango::Pipe *>::iterator it = pipe_list.begin();
                it != pipe_list.end(); ++it)
            {
                py::object py_value = py::cast(*it);
                py_pipe_list.append(py_value);
            }
            // Could this be?
            // return py::cast(self.get_pipe_list())
            return py_pipe_list;
        })
        .def("get_cmd_by_name", [](DeviceClassWrap& self, const std::string& name) -> Tango::Command& {
            return self.get_cmd_by_name(name);
        })
        .def("get_pipe_by_name", [](DeviceClassWrap& self, const std::string& pipe_name, const std::string& dev_name) -> Tango::Pipe& {
            return self.get_pipe_by_name(pipe_name, dev_name);
        })
        .def("set_type", [](DeviceClassWrap& self, std::string& type) -> void {
            self.set_type(type);
        })
        .def("add_wiz_dev_prop", [](DeviceClassWrap& self, std::string& name , std::string& desc) -> void {
            self.add_wiz_dev_prop(name, desc);
        })
        .def("add_wiz_dev_prop", [](DeviceClassWrap& self, std::string& name , std::string& desc, std::string& def) -> void {
            self.add_wiz_dev_prop(name, desc, def);
        })
        .def("add_wiz_class_prop", [](DeviceClassWrap& self, std::string& name , std::string& desc) -> void {
            self.add_wiz_class_prop(name, desc);
        })
        .def("add_wiz_class_prop", [](DeviceClassWrap& self, std::string& name , std::string& desc, std::string& def) -> void {
            self.add_wiz_class_prop(name, desc, def);
        })
        .def("_device_destroyer", [](DeviceClassWrap& self, const std::string& dev_name) -> void {
            self.device_destroyer(dev_name);
        })
        .def("_create_attribute", [](DeviceClassWrap& self,
                const std::string& attr_name,
                Tango::CmdArgType attr_type,
                Tango::AttrDataFormat attr_format,
                Tango::AttrWriteType attr_write,
                long dim_x, long dim_y,
                Tango::DispLevel display_level,
                long polling_period,
                bool memorized, bool hw_memorized,
                const std::string& read_method_name,
                const std::string& write_method_name,
                const std::string& is_allowed_name,
                Tango::UserDefaultAttrProp *att_prop) -> void {
            self.create_attribute(
                    attr_name,
                    attr_type,
                    attr_format,
                    attr_write,
                    dim_x, dim_y,
                    display_level,
                    polling_period,
                    memorized, hw_memorized,
                    read_method_name,
                    write_method_name,
                    is_allowed_name,
                    att_prop);
        })
        .def("_create_fwd_attribute", [](DeviceClassWrap& self,
                const std::string& attr_name,
                Tango::UserDefaultFwdAttrProp *att_prop) -> void {
            self.create_fwd_attribute(attr_name, att_prop);
        })
        .def("_create_pipe", [](DeviceClassWrap& self,
                const std::string& name,
                Tango::PipeWriteType access,
                Tango::DispLevel display_level,
                const std::string& read_method_name,
                const std::string& write_method_name,
                const std::string& is_allowed_name,
                Tango::UserDefaultPipeProp *prop) -> void {
            self.create_pipe(
                    name,
                    access,
                    display_level,
                    read_method_name,
                    write_method_name,
                    is_allowed_name,
                    prop);
        })
        .def("_create_command", [](DeviceClassWrap& self,
                std::string& cmd_name,
                Tango::CmdArgType param_type,
                Tango::CmdArgType result_type,
                std::string& param_desc,
                std::string& result_desc,
                Tango::DispLevel display_level,
                bool default_command,
                long polling_period,
                const std::string& is_allowed) -> void {
            self.create_command(cmd_name,
                    param_type,
                    result_type,
                    param_desc,
                    result_desc,
                    display_level,
                    default_command,
                    polling_period,
                    is_allowed);
        })
        .def("get_class_attr", [](DeviceClassWrap& self) -> Tango::MultiClassAttribute* {
            return self.get_class_attr();
        })
    ;
}

