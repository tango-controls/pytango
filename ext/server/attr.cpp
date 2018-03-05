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
#include <pybind11/functional.h>
#include <server/attr.h>
#include <pyutils.h>

namespace py = pybind11;

//
//int func_arg(const std::function<int(int)> &f) {
//    return f(10);
//}

void PyAttr::read(Tango::DeviceImpl *dev, Tango::Attribute &att)
{
    std::cout << "in read attr.cpp" << std::endl;
    if (!_is_method(dev, read_name))
    {
        TangoSys_OMemStream o;
        o << read_name << " method not found for " << att.get_name();
        Tango::Except::throw_exception("PyTango_ReadAttributeMethodNotFound",
            o.str(), "PyTango::Attr::read");
    }
//    CALL_ATTR_METHOD_VARGS(dev, read_name.c_str(), att);
    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
    AutoPythonGIL __py_lock;
    try {
//gm        py::call_method<void>(__dev_ptr->the_self, read_name.c_str(), att);
    } catch (py::error_already_set &eas) {
        handle_python_exception(eas);
    }
}

void PyAttr::write(Tango::DeviceImpl *dev, Tango::WAttribute &att)
{
    if (!_is_method(dev, write_name))
    {
        TangoSys_OMemStream o;
        o << write_name << " method not found for " << att.get_name();
        Tango::Except::throw_exception("PyTango_WriteAttributeMethodNotFound",
            o.str(), "PyTango::Attr::write");
    }
//    CALL_ATTR_METHOD_VARGS(dev, write_name.c_str(), att);
    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
    AutoPythonGIL __py_lock;
    try {
//gm        py::call_method<void>(__dev_ptr->the_self, write_name.c_str(), att);
    } catch (py::error_already_set &eas) {
        handle_python_exception(eas);
    }
}

bool PyAttr::is_allowed(Tango::DeviceImpl *dev, Tango::AttReqType type)
{
    if (_is_method(dev, py_allowed_name))
    {
        PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
        AutoPythonGIL __py_lock;
        try {
    //gm        return py::call_method<bool>(__dev_ptr->the_self, py_allowed_name.c_str(), type);
        } catch (py::error_already_set &eas) {
            handle_python_exception(eas);
        }
    }
    // keep compiler quiet
    return true;
}

bool PyAttr::_is_method(Tango::DeviceImpl *dev, const std::string &name)
{
    AutoPythonGIL __py_lock;
    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
    PyObject *__dev_py = __dev_ptr->the_self;
    return is_method_defined(__dev_py, name);
}

void PyAttr::set_user_prop(std::vector<Tango::AttrProperty> &user_prop,
                           Tango::UserDefaultAttrProp &def_prop)
{

//
// Is there any user defined prop. defined ?
//

    size_t nb_prop = user_prop.size();
    if (nb_prop == 0)
        return;

    for (size_t loop = 0;loop < nb_prop;loop++)
    {
        Tango::AttrProperty  prop = user_prop[loop];
        std::string &prop_name = prop.get_name();
        const char *prop_value = prop.get_value().c_str();

        if (prop_name == "label")
            def_prop.set_label(prop_value);
        else if (prop_name == "description")
            def_prop.set_description(prop_value);
        else if (prop_name == "unit")
            def_prop.set_unit(prop_value);
        else if (prop_name == "standard_unit")
            def_prop.set_standard_unit(prop_value);
        else if (prop_name == "display_unit")
            def_prop.set_display_unit(prop_value);
        else if (prop_name == "format")
            def_prop.set_format(prop_value);
        else if (prop_name == "min_value")
            def_prop.set_min_value(prop_value);
        else if (prop_name == "max_value")
            def_prop.set_max_value(prop_value);
        else if (prop_name == "min_alarm")
            def_prop.set_min_alarm(prop_value);
        else if (prop_name == "max_alarm")
            def_prop.set_max_alarm(prop_value);
        else if (prop_name == "min_warning")
            def_prop.set_min_warning(prop_value);
        else if (prop_name == "max_warning")
            def_prop.set_max_warning(prop_value);
        else if (prop_name == "delta_val")
            def_prop.set_delta_val(prop_value);
        else if (prop_name == "delta_t")
            def_prop.set_delta_t(prop_value);
        else if (prop_name == "abs_change")
            def_prop.set_event_abs_change(prop_value);
        else if (prop_name == "rel_change")
            def_prop.set_event_rel_change(prop_value);
        else if (prop_name == "period")
            def_prop.set_event_period(prop_value);
        else if (prop_name == "archive_abs_change")
            def_prop.set_archive_event_abs_change(prop_value);
        else if (prop_name == "archive_rel_change")
            def_prop.set_archive_event_rel_change(prop_value);
        else if (prop_name == "archive_period")
            def_prop.set_archive_event_period(prop_value);
    }
}

void export_attr(py::module &m) {
    py::class_<Tango::Attr>(m, "Attr")
        .def(py::init<const char *, long>())
//        .def(py::init<const char *, long, optional<Tango::AttrWriteType, const char *>>())
        .def("set_default_properties", &Tango::Attr::set_default_properties)
        .def("set_disp_level", &Tango::Attr::set_disp_level)
        .def("set_polling_period", &Tango::Attr::set_polling_period)
        .def("set_memorized", &Tango::Attr::set_memorized)
        .def("set_memorized_init", &Tango::Attr::set_memorized_init)
        .def("set_change_event", &Tango::Attr::set_change_event)
        .def("is_change_event", &Tango::Attr::is_change_event)
        .def("is_check_change_criteria", &Tango::Attr::is_check_change_criteria)
        .def("set_archive_event", &Tango::Attr::set_archive_event)
        .def("is_archive_event", &Tango::Attr::is_archive_event)
        .def("is_check_archive_criteria", &Tango::Attr::is_check_archive_criteria)
        .def("set_data_ready_event", &Tango::Attr::set_data_ready_event)
        .def("is_data_ready_event", &Tango::Attr::is_data_ready_event)
        .def("get_name", &Tango::Attr::get_name,
                py::return_value_policy::copy)
        .def("get_format", &Tango::Attr::get_format)
        .def("get_writable", &Tango::Attr::get_writable)
        .def("get_type", &Tango::Attr::get_type)
        .def("get_disp_level", &Tango::Attr::get_disp_level)
        .def("get_polling_period", &Tango::Attr::get_polling_period)
        .def("get_memorized", &Tango::Attr::get_memorized)
        .def("get_memorized_init", &Tango::Attr::get_memorized_init)
        .def("get_assoc", &Tango::Attr::get_assoc,
                py::return_value_policy::copy)
        .def("is_assoc", &Tango::Attr::is_assoc)
        .def("get_cl_name", &Tango::Attr::get_cl_name,
                py::return_value_policy::copy)
        .def("set_cl_name", &Tango::Attr::set_cl_name)
        .def("get_class_properties", &Tango::Attr::get_class_properties,
                py::return_value_policy::reference)
        .def("get_user_default_properties", &Tango::Attr::get_user_default_properties,
                py::return_value_policy::reference)
        .def("set_class_properties", &Tango::Attr::set_class_properties)
        .def("check_type", &Tango::Attr::check_type)
        .def("read", &Tango::Attr::read)
        .def("write", &Tango::Attr::write)
        .def("is_allowed", &Tango::Attr::is_allowed)
    ;

    py::class_<Tango::SpectrumAttr, Tango::Attr>(m, "SpectrumAttr")
        .def(py::init<const char *, long, Tango::AttrWriteType, long>())
    ;

    py::class_<Tango::ImageAttr, Tango::SpectrumAttr>(m, "ImageAttr")
        .def(py::init<const char *, long, Tango::AttrWriteType, long, long>())
    ;

    py::class_<Tango::AttrProperty>(m, "AttrProperty")
        .def(py::init<const char *, const char *>())
        .def(py::init<const char *, long>())
        .def("get_value", &Tango::AttrProperty::get_value,
                py::return_value_policy::copy)
        .def("get_lg_value", &Tango::AttrProperty::get_lg_value)
        .def("get_name", &Tango::AttrProperty::get_name,
                py::return_value_policy::copy)
    ;
}
