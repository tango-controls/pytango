/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
   
   PyTango is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   PyTango is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

#include "pytgutils.h"
#include "exception.h"
#include "server/device_class.h"
#include "server/attr.h"
#include "server/command.h"

#include <sstream>

using namespace boost::python;

#define __AUX_DECL_CALL_DEVCLASS_METHOD \
    AutoPythonGIL __py_lock;

#define __AUX_CATCH_PY_EXCEPTION \
    catch(boost::python::error_already_set &eas) \
    { handle_python_exception(eas); }

#define CALL_DEVCLASS_METHOD(name) \
    __AUX_DECL_CALL_DEVCLASS_METHOD \
    try { boost::python::call_method<void>(m_self, #name); } \
    __AUX_CATCH_PY_EXCEPTION

#define CALL_DEVCLASS_METHOD_VARGS(name, ...) \
    __AUX_DECL_CALL_DEVCLASS_METHOD \
    try { boost::python::call_method<void>(m_self, #name, __VA_ARGS__); } \
    __AUX_CATCH_PY_EXCEPTION

CppDeviceClass::CppDeviceClass(const string &name)
    :Tango::DeviceClass(const_cast<string&>(name))
{}

CppDeviceClass::~CppDeviceClass()
{}

void CppDeviceClass::create_command(const std::string &cmd_name,
                                    Tango::CmdArgType param_type,
                                    Tango::CmdArgType result_type,
                                    const std::string &param_desc,
                                    const std::string &result_desc,
                                    Tango::DispLevel display_level,
                                    bool default_command,
                                    long polling_period,
                                    const std::string &is_allowed)
{
    PyCmd *cmd_ptr = new PyCmd(cmd_name.c_str(), param_type, result_type,
                               param_desc.c_str(), result_desc.c_str(),
                               display_level);

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

void CppDeviceClass::create_attribute(vector<Tango::Attr *> &att_list,
                                    const std::string &attr_name,
                                    Tango::CmdArgType attr_type,
                                    Tango::AttrDataFormat attr_format,
                                    Tango::AttrWriteType attr_write,
                                    long dim_x, long dim_y,
                                    Tango::DispLevel display_level,
                                    long polling_period,
                                    bool memorized, bool hw_memorized,
                                    const std::string &read_method_name,
                                    const std::string &write_method_name,
                                    const std::string &is_allowed_name,
                                    Tango::UserDefaultAttrProp *att_prop)
{
    //
    // Create the attribute objet according to attribute format
    //

    PyScaAttr *sca_attr_ptr = NULL;
    PySpecAttr *spec_attr_ptr = NULL;
    PyImaAttr *ima_attr_ptr= NULL;
    PyAttr *py_attr_ptr = NULL;
    Tango::Attr *attr_ptr = NULL;

    switch (attr_format)
    {
        case Tango::SCALAR:
            sca_attr_ptr = new PyScaAttr(attr_name, attr_type, attr_write);
            py_attr_ptr = sca_attr_ptr;
            attr_ptr = sca_attr_ptr;
            break;

        case Tango::SPECTRUM:
            spec_attr_ptr = new PySpecAttr(attr_name.c_str(), attr_type, attr_write, dim_x);
            py_attr_ptr = spec_attr_ptr;
            attr_ptr = spec_attr_ptr;
            break;

        case Tango::IMAGE:
            ima_attr_ptr = new PyImaAttr(attr_name.c_str(), attr_type, attr_write, dim_x, dim_y);
            py_attr_ptr = ima_attr_ptr;
            attr_ptr = ima_attr_ptr;
            break;

        default:
            TangoSys_OMemStream o;
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

    if (att_prop)
        attr_ptr->set_default_properties(*att_prop);
        
    attr_ptr->set_disp_level(display_level);
    if (memorized)
    {
        attr_ptr->set_memorized();
        attr_ptr->set_memorized_init(hw_memorized);
    }

    if (polling_period > 0)
        attr_ptr->set_polling_period(polling_period);

    att_list.push_back(attr_ptr);
}

CppDeviceClassWrap::CppDeviceClassWrap(PyObject *self, const std::string &name)
    : CppDeviceClass(name), m_self(self)
{
    init_class();
}

/**
 * Destructor
 */
CppDeviceClassWrap::~CppDeviceClassWrap()
{}

void CppDeviceClassWrap::init_class()
{
    AutoPythonGIL python_guard;
    signal_handler_defined = is_method_defined(m_self, "signal_handler");
}

void CppDeviceClassWrap::attribute_factory(std::vector<Tango::Attr *> &att_list)
{
    //
    // make sure we pass the same vector object to the python method
    //
    object py_att_list(
                handle<>(
                    to_python_indirect<
                        std::vector<Tango::Attr *>,
                        detail::make_reference_holder>()(att_list)));
    CALL_DEVCLASS_METHOD_VARGS(_DeviceClass__attribute_factory, py_att_list)
}

void CppDeviceClassWrap::command_factory()
{
    CALL_DEVCLASS_METHOD(_DeviceClass__command_factory)
}

void CppDeviceClassWrap::device_factory(const Tango::DevVarStringArray *dev_list)
{
    CALL_DEVCLASS_METHOD_VARGS(device_factory, dev_list)
}

void CppDeviceClassWrap::signal_handler(long signo)
{
    if (signal_handler_defined == true)
    {
        CALL_DEVCLASS_METHOD_VARGS(signal_handler, signo)
    }
    else
    {
        Tango::DeviceClass::signal_handler(signo);
    }
}

void CppDeviceClassWrap::default_signal_handler(long signo)
{
    this->Tango::DeviceClass::signal_handler(signo);
}

void CppDeviceClassWrap::delete_class()
{
    AutoPythonGIL __py_lock;

    try
    {
        //
        // Call the delete_class_list function in order to clear the global
        // constructed class Python list. It is MANDATORY to destroy these objects
        // from Python. Otherwise, there are "seg fault" when Python exit.
        // It tooks me quite a long time to find this...
        //
        PYTANGO_MOD
        pytango.attr("delete_class_list")();
    }
    catch(error_already_set &eas)
    {
        handle_python_exception(eas);
    }

}

namespace PyDeviceClass
{

    object get_device_list(CppDeviceClass &self)
    {
        boost::python::list py_dev_list;
        vector<Tango::DeviceImpl *> dev_list = self.get_device_list();
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
    
    /*
    void add_device(CppDeviceClass &self, auto_ptr<Tango::DeviceImpl> dev)
    {
        self.add_device(dev.get());
        dev.release();
    }

    void add_device(CppDeviceClass &self, auto_ptr<Tango::Device_4Impl> dev)
    {
        self.add_device(dev.get());
        dev.release();
    }

    void (*add_device1)(CppDeviceClass &, auto_ptr<Tango::DeviceImpl>) = &add_device;
    void (*add_device2)(CppDeviceClass &, auto_ptr<Tango::Device_4Impl>) = &add_device;
    */
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS (export_device_overload,
                                        CppDeviceClass::export_device, 1, 2)

#if !(defined __linux)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS (register_signal_overload,
                                        Tango::DeviceClass::register_signal, 1, 1)
#else
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS (register_signal_overload,
                                        Tango::DeviceClass::register_signal, 1, 2)
#endif


void export_device_class()
{
    void (Tango::DeviceClass::*add_wiz_dev_prop_)(string &,string &) =
        &Tango::DeviceClass::add_wiz_dev_prop;
    void (Tango::DeviceClass::*add_wiz_dev_prop__)(string &,string &,string &) =
        &Tango::DeviceClass::add_wiz_dev_prop;
    void (Tango::DeviceClass::*add_wiz_class_prop_)(string &,string &) =
        &Tango::DeviceClass::add_wiz_class_prop;
    void (Tango::DeviceClass::*add_wiz_class_prop__)(string &,string &,string &) =
        &Tango::DeviceClass::add_wiz_class_prop;

    class_<CppDeviceClass, auto_ptr<CppDeviceClassWrap>, boost::noncopyable>("_DeviceClass",
        init<const std::string &>())

        .def("device_factory", &CppDeviceClassWrap::device_factory)
        .def("export_device", &CppDeviceClass::export_device,
            export_device_overload())
        //.def("_add_device", PyDeviceClass::add_device1)
        //.def("_add_device", PyDeviceClass::add_device2)
        .def("_add_device", &CppDeviceClass::add_device)
        .def("register_signal",&Tango::DeviceClass::register_signal,
            register_signal_overload())
        .def("unregister_signal", &Tango::DeviceClass::unregister_signal)
        .def("signal_handler", &Tango::DeviceClass::signal_handler,
            &CppDeviceClassWrap::default_signal_handler)
        .def("get_name", &Tango::DeviceClass::get_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_type", &Tango::DeviceClass::get_type,
            return_value_policy<copy_non_const_reference>())
        .def("get_doc_url", &Tango::DeviceClass::get_doc_url,
            return_value_policy<copy_non_const_reference>())
        .def("get_cvs_tag", &Tango::DeviceClass::get_cvs_tag,
            return_value_policy<copy_non_const_reference>())
        .def("get_cvs_location",&Tango::DeviceClass::get_cvs_location,
            return_value_policy<copy_non_const_reference>())
        .def("get_device_list",&PyDeviceClass::get_device_list)
        .def("set_type",
            (void (Tango::DeviceClass::*) (const char *))
            &Tango::DeviceClass::set_type)
        .def("add_wiz_dev_prop",
            (void (Tango::DeviceClass::*) (const std::string &, const std::string &))
            add_wiz_dev_prop_)
        .def("add_wiz_dev_prop",
            (void (Tango::DeviceClass::*) (const std::string &, const std::string &, const std::string &))
            add_wiz_dev_prop__)
        .def("add_wiz_class_prop",
            (void (Tango::DeviceClass::*) (const std::string &, const std::string &))
            add_wiz_class_prop_)
        .def("add_wiz_class_prop",
            (void (Tango::DeviceClass::*) (const std::string &, const std::string &, const std::string &))
            add_wiz_class_prop__)
        .def("_device_destroyer",
            (void (Tango::DeviceClass::*) (const char *))
            &Tango::DeviceClass::device_destroyer)
        .def("_create_attribute", &CppDeviceClass::create_attribute)
        .def("_create_command", &CppDeviceClass::create_command)
    ;
    implicitly_convertible<auto_ptr<CppDeviceClassWrap>, auto_ptr<CppDeviceClass> >();
}

