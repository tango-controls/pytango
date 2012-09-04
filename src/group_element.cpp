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

#include "precompiled_header.hpp"
#include "pytgutils.h"
#include "device_attribute.h"

void export_group_element()
{
    using namespace boost::python;

    class_<Tango::GroupElement, unique_pointer<Tango::GroupElement>,
           boost::noncopyable> GroupElement("GroupElement",
        "The abstract GroupElement class for Group. Not to be initialized\n"
        "directly.", no_init)
    ;

    GroupElement
    //
    // Group management methods
    //
//        .def("__add",
//            (void (Tango::GroupElement::*) (const std::string &, int))
//            &Tango::GroupElement::add,
//            (arg_("self"), arg_("pattern"), arg_("timeout_ms")=-1) )
//        .def("__add",
//            (void (Tango::GroupElement::*) (const std::vector<std::string> &, int))
//            &Tango::GroupElement::add,
//            (arg_("self"), arg_("patterns"), arg_("timeout_ms")=-1))
//        .def("__remove",
//            (void (Tango::GroupElement::*) (const std::string &, bool))
//            &Tango::GroupElement::remove,
//            (arg_("self"), arg_("pattern"), arg_("forward")=true))
//        .def("__remove",
//            (void (Tango::GroupElement::*) (const std::vector<std::string> &, bool))
//            &Tango::GroupElement::remove,
//            (arg_("self"), arg_("patterns"), arg_("forward")=true))
        .def("contains",
            &Tango::GroupElement::contains,
            (arg_("self"), arg_("pattern"), arg_("forward")=true) )
        .def("get_device",
            (Tango::DeviceProxy* (Tango::GroupElement::*) (const std::string &))
            &Tango::GroupElement::get_device,
            (arg_("self"), arg_("dev_name")),
            return_internal_reference<1>() )
        .def("get_device",
            (Tango::DeviceProxy* (Tango::GroupElement::*) (long))
            &Tango::GroupElement::get_device,
            (arg_("self"), arg_("idx")),
            return_internal_reference<1>() )
//        .def("get_group",
//            &Tango::GroupElement::get_group,
//            (arg_("self"), arg_("group_name")),
//            return_internal_reference<1>() )

    //
    // Tango methods (~ DeviceProxy interface)
    //
        .def("ping",
            pure_virtual(&Tango::GroupElement::ping),
            (arg_("self"), arg_("forward")=true) )
        .def("set_timeout_millis",
            pure_virtual(&Tango::GroupElement::set_timeout_millis),
            (arg_("self"), arg_("timeout_ms")) )


    //
    // Misc
    //
        .def("get_name",
            &Tango::GroupElement::get_name,
            (arg_("self")),
            return_value_policy<copy_const_reference>() )
        .def("get_fully_qualified_name",
            &Tango::GroupElement::get_fully_qualified_name,
            (arg_("self")) )
        .def("enable",
            &Tango::GroupElement::enable,
            (arg_("self")) )
        .def("disable",
            &Tango::GroupElement::disable,
            (arg_("self")) )
        .def("is_enabled",
            &Tango::GroupElement::is_enabled,
            (arg_("self")) )
        .def("name_equals",
            &Tango::GroupElement::name_equals,
            (arg_("self")) )
        .def("name_matches",
            &Tango::GroupElement::name_matches,
            (arg_("self")) )
//        .def("get_size",
//            &Tango::GroupElement::get_size,
//            (arg_("self"), arg_("forward")=true) )

    //
    // "Should not be used" methods
    //
        
        // According to the header,
        // The methods are: find, get_parent, set_parent, is_device,
        // is_group, dump.
        // I've choosed according to my very personal criteria which are really
        // needed and which aren't.

        // find looks in both devices and groups, and returns a GroupElement.
        // Problem is, it is impossible to get DeviceProxy from
        // GroupDeviceElement (It can be done only by private methods) so
        // it is useless for us. Our own find can be implemented with
        // get_device and get_group, and they have different return values
        // DeviceProxy and Group.
// 		.def("find",
// 			&Tango::GroupElement::find,
// 			(arg_("self"), arg_("pattern?"), arg_("forward")=true ),
// 			return_internal_reference<1>() )

        // It would return a x.get_group("grp").get_parent() != x
        // So, as it is not really necessary, we get rid of him...
//         .def("get_parent",
//             &Tango::GroupElement::get_parent,
//             (arg_("self")),
//             return_internal_reference<1>() )

        // I am not exporting "find", so all the GroupElemens will be
        // Groups (there's no way to access a GroupDeviceElement)
//         .def("is_device",
//             pure_virtual(&Tango::GroupElement::is_device),
//             (arg_("self")) )
// 
//         .def("is_group",
//             pure_virtual(&Tango::GroupElement::is_group),
//             (arg_("self")) )
    ;

}
