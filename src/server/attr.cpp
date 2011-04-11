#include <boost/python.hpp>
#include <tango.h>

#include "server/attr.h"

using namespace boost::python;

void PyAttr::set_user_prop(vector<Tango::AttrProperty> &user_prop,
                           Tango::UserDefaultAttrProp &def_prop)
{

//
// Is there any user defined prop. defined ?
//

    long nb_prop = user_prop.size();
    if (nb_prop == 0)
        return;

    for (long loop = 0;loop < nb_prop;loop++)
    {
        Tango::AttrProperty  prop = user_prop[loop];
        string &prop_name = prop.get_name();
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
            def_prop.set_abs_change(prop_value);
        else if (prop_name == "rel_change")
            def_prop.set_rel_change(prop_value);
        else if (prop_name == "period")
            def_prop.set_period(prop_value);
        else if (prop_name == "archive_abs_change")
            def_prop.set_archive_abs_change(prop_value);
        else if (prop_name == "archive_rel_change")
            def_prop.set_archive_rel_change(prop_value);
        else if (prop_name == "archive_period")
            def_prop.set_archive_period(prop_value);
    }
}

void export_attr()
{
    class_<Tango::Attr, boost::noncopyable>("Attr",
        init<const char *, long, optional<Tango::AttrWriteType, const char *> >())

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
        .def("get_name", &Tango::Attr::get_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_format", &Tango::Attr::get_format)
        .def("get_writable", &Tango::Attr::get_writable)
        .def("get_type", &Tango::Attr::get_type)
        .def("get_disp_level", &Tango::Attr::get_disp_level)
        .def("get_polling_period", &Tango::Attr::get_polling_period)
        .def("get_memorized", &Tango::Attr::get_memorized)
        .def("get_memorized_init", &Tango::Attr::get_memorized_init)
        .def("get_assoc", &Tango::Attr::get_assoc,
            return_value_policy<copy_non_const_reference>())
        .def("is_assoc", &Tango::Attr::is_assoc)
        .def("get_class_properties", &Tango::Attr::get_class_properties,
            return_internal_reference<>())
        .def("get_user_default_properties", &Tango::Attr::get_user_default_properties,
            return_internal_reference<>())
        .def("set_class_properties", &Tango::Attr::set_class_properties)
        .def("check_type", &Tango::Attr::check_type)
        .def("read", &Tango::Attr::read)
        .def("write", &Tango::Attr::write)
        .def("is_allowed", &Tango::Attr::is_allowed)
    ;

    class_<Tango::SpectrumAttr, bases<Tango::Attr>, boost::noncopyable>("SpectrumAttr",
        init<const char *, long, Tango::AttrWriteType, long>())
    ;

    class_<Tango::ImageAttr, bases<Tango::SpectrumAttr>, boost::noncopyable>("ImageAttr",
        init<const char *, long, Tango::AttrWriteType, long, long>())
    ;

//    class_<PyAttr>("PyAttr", no_init)
//        .def("set_allowed", &PyAttr::set_allowed)
//        .def("set_read_name", &PyAttr::set_read_name)
//        .def("set_write_name", &PyAttr::set_write_name)
//    ;
//
//    class_<PyScaAttr, bases<Tango::Attr, PyAttr>, boost::noncopyable>("PyScaAttr",
//        init<const std::string &, long , Tango::AttrWriteType>())
//    ;
//
//    class_<PySpecAttr, bases<Tango::SpectrumAttr, PyAttr>, boost::noncopyable>("PySpecAttr",
//        init<const std::string &, long , Tango::AttrWriteType, long>())
//    ;
//
//    class_<PyImaAttr, bases<Tango::ImageAttr, PyAttr>, boost::noncopyable>("PyImaAttr",
//        init<const std::string &, long , Tango::AttrWriteType, long, long>())
//    ;

    class_<Tango::AttrProperty>("AttrProperty",
        init<const char *, const char *>())
        .def(init<const char *, long>())
        .def("get_value", &Tango::AttrProperty::get_value,
            return_value_policy<copy_non_const_reference>())
        .def("get_lg_value", &Tango::AttrProperty::get_lg_value)
        .def("get_name", &Tango::AttrProperty::get_name,
            return_value_policy<copy_non_const_reference>())
    ;
}
