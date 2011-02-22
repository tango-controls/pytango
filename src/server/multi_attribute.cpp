#include "pytgutils.h"
#include <sstream>

using namespace boost::python;

void export_multi_attribute()
{
    class_<Tango::MultiAttribute, boost::noncopyable>("MultiAttribute", no_init)
        .def("get_attr_by_name", &Tango::MultiAttribute::get_attr_by_name,
            return_value_policy<reference_existing_object>())
        .def("get_attr_by_ind", &Tango::MultiAttribute::get_attr_by_ind,
            return_value_policy<reference_existing_object>())
        .def("get_w_attr_by_name", &Tango::MultiAttribute::get_w_attr_by_name,
            return_value_policy<reference_existing_object>())
        .def("get_w_attr_by_ind", &Tango::MultiAttribute::get_w_attr_by_ind,
            return_value_policy<reference_existing_object>())
        .def("get_attr_ind_by_name", &Tango::MultiAttribute::get_attr_ind_by_name) // New in 7.0.0
        .def("get_alarm_list", &Tango::MultiAttribute::get_alarm_list,
            return_internal_reference<>()) // New in 7.0.0
        .def("get_attr_nb", &Tango::MultiAttribute::get_attr_nb) // New in 7.0.0
        .def("check_alarm",
            (bool (Tango::MultiAttribute::*) ())
            &Tango::MultiAttribute::check_alarm) // New in 7.0.0
        .def("check_alarm",
            (bool (Tango::MultiAttribute::*) (const long))
            &Tango::MultiAttribute::check_alarm) // New in 7.0.0
        .def("check_alarm",
            (bool (Tango::MultiAttribute::*) (const char *))
            &Tango::MultiAttribute::check_alarm) // New in 7.0.0
        .def("read_alarm",
            (void (Tango::MultiAttribute::*) (const std::string &))
            &Tango::MultiAttribute::read_alarm) // New in 7.0.0
    ;
}
