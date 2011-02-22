#include <boost/python.hpp>
#include <tango.h>

using namespace boost::python;

void export_user_default_attr_prop()
{
    class_<Tango::UserDefaultAttrProp, boost::noncopyable>("UserDefaultAttrProp")
        .def("set_label", &Tango::UserDefaultAttrProp::set_label)
        .def("set_description", &Tango::UserDefaultAttrProp::set_description)
        .def("set_format", &Tango::UserDefaultAttrProp::set_format)
        .def("set_unit", &Tango::UserDefaultAttrProp::set_unit)
        .def("set_standard_unit", &Tango::UserDefaultAttrProp::set_standard_unit)
        .def("set_display_unit", &Tango::UserDefaultAttrProp::set_display_unit)
        .def("set_min_value", &Tango::UserDefaultAttrProp::set_min_value)
        .def("set_max_value", &Tango::UserDefaultAttrProp::set_max_value)
        .def("set_min_alarm", &Tango::UserDefaultAttrProp::set_min_alarm)
        .def("set_max_alarm", &Tango::UserDefaultAttrProp::set_max_alarm)
        .def("set_min_warning", &Tango::UserDefaultAttrProp::set_min_warning)
        .def("set_max_warning", &Tango::UserDefaultAttrProp::set_max_warning)
        .def("set_delta_t", &Tango::UserDefaultAttrProp::set_delta_t)
        .def("set_delta_val", &Tango::UserDefaultAttrProp::set_delta_val)
        .def("set_abs_change", &Tango::UserDefaultAttrProp::set_abs_change)
        .def("set_rel_change", &Tango::UserDefaultAttrProp::set_rel_change)
        .def("set_period", &Tango::UserDefaultAttrProp::set_period)
        .def("set_archive_abs_change", &Tango::UserDefaultAttrProp::set_archive_abs_change)
        .def("set_archive_rel_change", &Tango::UserDefaultAttrProp::set_archive_rel_change)
        .def("set_archive_period", &Tango::UserDefaultAttrProp::set_archive_period)
    ;

}

