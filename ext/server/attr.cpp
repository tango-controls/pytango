/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <server/attr.h>
#include <server/attribute.h>
#include <pyutils.h>
#include <tgutils.h>

namespace py = pybind11;

template<long tangoTypeConst>
inline py::object __get_attr_write_value(Tango::WAttribute& att)
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    TangoScalarType value;
    att.get_write_value(value);
    return py::cast(value);
}

template<>
inline py::object __get_attr_write_value<Tango::DEV_ENCODED>(Tango::WAttribute& att)
{
    typedef typename TANGO_const2type(Tango::DEV_ENCODED) TangoScalarType;
    TangoScalarType value;
    att.get_write_value(value);
    std::string encoded_format = std::string(value.encoded_format);
    Tango::DevVarCharArray encoded_data = value.encoded_data;
    py::list encoded_list;
    for (auto i=0; i<encoded_data.length(); i++) {
        encoded_list.append(py::cast(encoded_data[i]));
    }
    py::str py_str = py::cast(encoded_format);
    return py::make_tuple(py_str, encoded_list);
}

template<long tangoTypeConst>
inline py::object __get_array_attr_write_value(Tango::WAttribute& att, bool isImage)
{
    typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
    const TangoScalarType* ptr;
    att.get_write_value(ptr);

    long dim_x = att.get_w_dim_x();
    long dim_y = att.get_w_dim_y();
    py::list values;
    if (isImage) {
        int k = 0;
        for (int i=0; i<dim_y; i++) {
            py::list sub_values_list;
            for (int j=0; j<dim_x; j++) {
                sub_values_list.append(py::cast(ptr[k++]));
            }
            values.append(sub_values_list);
        }
    } else {
        for (int i=0; i<dim_x; i++) {
            values.append(py::cast(ptr[i]));
        }
    }
    return values;
}

template<>
inline py::object __get_array_attr_write_value<Tango::DEV_STRING>(Tango::WAttribute& att, bool isImage)
{
    const Tango::ConstDevString* ptr;
    att.get_write_value(ptr);
    long dim_x = att.get_w_dim_x();
    long dim_y = att.get_w_dim_y();
    py::list values;
    if (isImage) {
        int k = 0;
        for (int i=0; i<dim_y; i++) {
            py::list sub_values_list;
            for (int j=0; j<dim_x; j++) {
                sub_values_list.append(py::cast(ptr[k++]));
            }
            values.append(sub_values_list);
        }
    } else {
        for (int i=0; i<dim_x; i++) {
            values.append(py::cast(ptr[i]));
        }
    }
    return values;
}

py::object PyAttr::get_attr_write_value(Tango::WAttribute& att)
{
    long type = att.get_data_type();
    Tango::AttrDataFormat format = att.get_data_format();

    const bool isScalar = (format == Tango::SCALAR);
    const bool isImage = (format == Tango::IMAGE);

    if (isScalar) {
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, return __get_attr_write_value, att);
    } else {
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, return __get_array_attr_write_value, att, isImage);
    }
}

void PyAttr::read(Tango::DeviceImpl* dev, Tango::Attribute& att)
{
    DeviceImplWrap* dev_ptr = dynamic_cast<DeviceImplWrap*>(dev);
    AutoPythonGILEnsure __py_lock;
    if (!is_method_callable(dev_ptr->py_self, read_name))
    {
        std::stringstream o;
        o << read_name << " method not found for " << &att.get_name();
        Tango::Except::throw_exception("PyTango_ReadAttributeMethodNotFound",
            o.str(), "PyTango::Attr::read");
    }
    try {
        py::object obj = dev_ptr->py_self.attr(read_name.c_str())();
        if (!att.get_value_flag()) {
            PyAttribute::set_complex_value(att, obj);
        }
    } catch (py::error_already_set &eas) {
        handle_python_exception(eas);
    }
}

void PyAttr::write(Tango::DeviceImpl* dev, Tango::WAttribute& att)
{
    DeviceImplWrap* dev_ptr = dynamic_cast<DeviceImplWrap*>(dev);
    AutoPythonGILEnsure __py_lock;
    if (!is_method_callable(dev_ptr->py_self, write_name))
    {
        std::stringstream o;
        o << write_name << " method not found for " << att.get_name();
        Tango::Except::throw_exception("PyTango_WriteAttributeMethodNotFound",
            o.str(), "PyTango::Attr::write");
    }
    try {
        py::object obj = get_attr_write_value(att);
        dev_ptr->py_self.attr(write_name.c_str())(obj);
    } catch (py::error_already_set &eas) {
        handle_python_exception(eas);
    }
}

bool PyAttr::is_allowed(Tango::DeviceImpl* dev, Tango::AttReqType type)
{
    DeviceImplWrap* dev_ptr = dynamic_cast<DeviceImplWrap*>(dev);
    AutoPythonGILEnsure __py_lock;
    if (is_method_callable(dev_ptr->py_self, py_allowed_name))
    {
        try {
            py::object obj = dev_ptr->py_self.attr(py_allowed_name.c_str())(type);
            return obj.cast<bool>();
        } catch (py::error_already_set &eas) {
            handle_python_exception(eas);
        }
    }
    return true; // keep compiler quiet
}

void PyAttr::set_user_prop(std::vector<Tango::AttrProperty>& user_prop,
                           Tango::UserDefaultAttrProp& def_prop)
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
        std::string& prop_name = prop.get_name();
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
        .def(py::init([](std::string& name, long data_type, Tango::AttrWriteType w_type, std::string& assoc) {
            return new Tango::Attr(name.c_str(), data_type, w_type, assoc.c_str());
        }), py::arg("name"), py::arg("data_type"),py::arg("w_type")=Tango::READ, py::arg("assoc")=Tango::AssocWritNotSpec)

        .def(py::init([](std::string& name, long data_type, Tango::DispLevel level, Tango::AttrWriteType w_type, std::string& assoc) {
            return new Tango::Attr(name.c_str(), data_type, level, w_type, assoc.c_str());
        }), py::arg("name"), py::arg("data_type"),py::arg("level"), py::arg("w_type")=Tango::READ, py::arg("assoc")=Tango::AssocWritNotSpec)

        .def(py::init([](const Tango::Attr& other) {
            return new Tango::Attr(other);
        }))
        .def("set_default_properties", [](Tango::Attr& self, Tango::UserDefaultAttrProp &prop) -> void {
            self.set_default_properties(prop);
        })
        .def("set_disp_level", [](Tango::Attr& self, Tango::DispLevel level) -> void {
            self.set_disp_level(level);
        })
        .def("set_polling_period", [](Tango::Attr& self, long update) -> void {
            self.set_polling_period(update);
        })
        .def("set_memorized", [](Tango::Attr& self) -> void {
            self.set_memorized();
        })
        .def("set_memorized_init", [](Tango::Attr& self, bool write_on_init) -> void {
            return self.set_memorized_init(write_on_init);
        })
        .def("set_change_event", [](Tango::Attr& self, bool implemented, bool detect) -> void {
            return self.set_change_event(implemented, detect);
        })
        .def("is_change_event", [](Tango::Attr& self) -> bool {
            return self.is_change_event();
        })
        .def("is_check_change_criteria", [](Tango::Attr& self) -> bool {
            return self.is_check_change_criteria();
        })
        .def("set_archive_event", [](Tango::Attr& self, bool implemented, bool detect) -> void {
            return self.set_archive_event(implemented, detect);
        })
        .def("is_archive_event", [](Tango::Attr& self) -> bool {
            return self.is_archive_event();
        })
        .def("is_check_archive_criteria", [](Tango::Attr& self) -> bool {
            return self.is_check_archive_criteria();
        })
        .def("set_data_ready_event", [](Tango::Attr& self, bool implemented) -> void {
            return self.set_data_ready_event(implemented);
        })
        .def("is_data_ready_event", [](Tango::Attr& self) -> bool {
            return self.is_data_ready_event();
        })
        .def("get_name", [](Tango::Attr& self) -> std::string& {
            return self.get_name();
        })
        .def("get_format", [](Tango::Attr& self) -> Tango::AttrDataFormat {
            return self.get_format();
        })
        .def("get_writable", [](Tango::Attr& self) -> Tango::AttrWriteType {
            return self.get_writable();
        })
        .def("get_type", [](Tango::Attr& self) -> long {
            return self.get_type();
        })
        .def("get_disp_level",[](Tango::Attr& self) -> Tango::DispLevel {
            return self.get_disp_level();
        })
         .def("get_polling_period", [](Tango::Attr& self) -> long {
            return self.get_polling_period();
        })
        .def("get_memorized", [](Tango::Attr& self) -> bool {
            return self.get_memorized();
        })
        .def("get_memorized_init", [](Tango::Attr& self) -> bool {
            return self.get_memorized_init();
        })
        .def("get_assoc", [](Tango::Attr& self) -> std::string& {
            return self.get_assoc();
        })
        .def("is_assoc", [](Tango::Attr& self) -> bool {
            return self.is_assoc();
        })
        .def("get_cl_name", [](Tango::Attr& self) -> const std::string& {
            return self.get_cl_name();
        })
        .def("set_cl_name", [](Tango::Attr& self, std::string& cl) -> void {
            return self.set_cl_name(cl.c_str());
        })
        .def("get_class_properties", [](Tango::Attr& self) -> std::vector<Tango::AttrProperty>& {
            return self.get_class_properties();
        })
        .def("get_user_default_properties", [](Tango::Attr& self) -> std::vector<Tango::AttrProperty>& {
            return self.get_user_default_properties();
        })
        .def("set_class_properties", [](Tango::Attr& self, std::vector<Tango::AttrProperty>& in_prop) -> void {
            return self.set_class_properties(in_prop);
        })
        .def("check_type", [](Tango::Attr& self) -> void {
            return self.check_type();
        })
    ;

    py::class_<Tango::SpectrumAttr, Tango::Attr>(m, "SpectrumAttr")
        .def(py::init([](std::string& name, long data_type, long max_x) {
            return new Tango::SpectrumAttr(name.c_str(), data_type, max_x);
        }))
        .def(py::init([](std::string& name, long data_type, Tango::AttrWriteType type, long max_x) {
            return new Tango::SpectrumAttr(name.c_str(), data_type, type, max_x);
        }))
        .def(py::init([](std::string& name, long data_type, long max_x, Tango::DispLevel level) {
            return new Tango::SpectrumAttr(name.c_str(), data_type, max_x);
        }))
        .def(py::init([](std::string& name, long data_type, Tango::AttrWriteType type, long max_x, Tango::DispLevel level) {
            return new Tango::SpectrumAttr(name.c_str(), data_type, type, max_x, level);
        }))
    ;

    py::class_<Tango::ImageAttr, Tango::SpectrumAttr>(m, "ImageAttr")
        .def(py::init([](std::string& name, long data_type, long max_x, long max_y) {
            return new Tango::ImageAttr(name.c_str(), data_type, max_x, max_y);
        }))
        .def(py::init([](std::string& name, long data_type, Tango::AttrWriteType type, long max_x, long max_y) {
            return new Tango::ImageAttr(name.c_str(), data_type, type, max_x, max_y);
        }))
        .def(py::init([](std::string& name, long data_type, long max_x, long max_y, Tango::DispLevel level) {
            return new Tango::ImageAttr(name.c_str(), data_type, max_x, max_y, level);
        }))
        .def(py::init([](std::string& name, long data_type, Tango::AttrWriteType type, long max_x, long max_y, Tango::DispLevel level) {
            return new Tango::ImageAttr(name.c_str(), data_type, type, max_x, max_y, level);
        }))
    ;

    py::class_<Tango::AttrProperty>(m, "AttrProperty")
        .def(py::init([](std::string& name, std::string& value) {
            return new Tango::AttrProperty(name, value);
        }))
        .def(py::init([](std::string& name, long value) {
            return new Tango::AttrProperty(name.c_str(), value);
        }))
        .def("get_value", [](Tango::AttrProperty& self) -> std::string& {
            return self.get_value();
        })
        .def("get_lg_value", [](Tango::AttrProperty& self) -> long {
            return self.get_lg_value();
        })
        .def("get_name", [](Tango::AttrProperty& self) -> std::string& {
            return self.get_name();
        })
    ;
}
