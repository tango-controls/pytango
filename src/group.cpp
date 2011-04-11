/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   (copyleft) CELLS / ALBA Synchrotron, Bellaterra, Spain
  
   This is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.
  
   This software is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with this program; if not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

#include <boost/python/copy_const_reference.hpp>
#include <boost/python/copy_non_const_reference.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <tango.h>
#include <memory>

#include "pytgutils.h"
#include "device_attribute.h"

void export_group_reply_list();
void export_group_reply();
void export_group_element();


namespace PyGroup
{
    using namespace boost::python;

    void add(Tango::Group& self, std::auto_ptr<Tango::Group> grp, int timeout_ms)
    {
        Tango::Group* grp_ptr = grp.get();
        
        if (grp_ptr) {
            // After adding grp_ptr into self, self is the responsible of
            // deleting grp_ptr, so we "nullify" the grp object. It's python
            // counterpart will still be available, but any method call will
            // return an exception.
            self.add(grp_ptr, timeout_ms);
            grp.release();
        } else {
            raise_(PyExc_TypeError,
                   "Param \"group\" is null. It probably means that it has"
                    " already been inserted in another group." );
        }
    }
}

void export_group()
{
    using namespace boost::python;
    
    export_group_reply();
    export_group_reply_list();
    export_group_element();

    class_<Tango::Group, bases<Tango::GroupElement>, std::auto_ptr<Tango::Group>, boost::noncopyable > Group(
        "__Group",
        init<const std::string&>())
    ;
    
    Group
        .def("__add",
            (void (Tango::Group::*) (const std::string &, int))
            &Tango::Group::add,
            (arg_("self"), arg_("pattern"), arg_("timeout_ms")=-1) )
        .def("__add",
            (void (Tango::Group::*) (const std::vector<std::string> &, int))
            &Tango::Group::add,
            (arg_("self"), arg_("patterns"), arg_("timeout_ms")=-1))
        .def("__add",
            PyGroup::add,
            (arg_("self"), arg_("group"), arg_("timeout_ms")=-1) )
            
        .def("remove_all", &Tango::Group::remove_all)

        // GroupElement redefinitions of enable/disable. If I didn't
        // redefine them, the later Group only definitions would
        // hide the ones defined in GroupElement.
        .def("enable",
            &Tango::GroupElement::enable,
            (arg_("self")) )
        .def("disable",
            &Tango::GroupElement::disable,
            (arg_("self")) )
        .def("enable",
            &Tango::Group::enable,
            (arg_("self"), arg_("dev_name"), arg_("forward")=true) )
        .def("disable",
            &Tango::Group::disable,
            (arg_("self"), arg_("dev_name"), arg_("forward")=true) )
        
        .def("get_device_list",
            &Tango::Group::get_device_list,
            (arg_("self"), arg_("forward")=true) )
    ;

    // I am not exporting "find", so all the GroupElemens will be
    // Groups (there's no way to access a GroupDeviceElement)
//     class_<Tango::GroupDeviceElement, bases<Tango::GroupElement>, boost::noncopyable > GroupDeviceElement(
//         "GroupDeviceElement",
//         no_init)
//     ;
}
