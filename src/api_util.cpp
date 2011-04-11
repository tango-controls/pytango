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

#include <boost/python.hpp>
#include <boost/python/return_value_policy.hpp>
#include <tango.h>

#include "pyutils.h"

using namespace boost::python;

void (Tango::ApiUtil::*get_asynch_replies1)() = &Tango::ApiUtil::get_asynch_replies;
void (Tango::ApiUtil::*get_asynch_replies2)(long) = &Tango::ApiUtil::get_asynch_replies;

bool (Tango::ApiUtil::*in_server1)() = &Tango::ApiUtil::in_server;
void (Tango::ApiUtil::*in_server2)(bool) = &Tango::ApiUtil::in_server;

void export_api_util()
{
    class_<Tango::ApiUtil, boost::noncopyable>("ApiUtil", no_init)
        
        .def("instance", &Tango::ApiUtil::instance,
            return_value_policy<reference_existing_object>())
        .staticmethod("instance")
        
        .def("pending_asynch_call", &Tango::ApiUtil::pending_asynch_call)
        
        .def("get_asynch_replies", get_asynch_replies1)
        .def("get_asynch_replies", get_asynch_replies2)
        
        .def("set_asynch_cb_sub_model", &Tango::ApiUtil::set_asynch_cb_sub_model)
        .def("get_asynch_cb_sub_model", &Tango::ApiUtil::get_asynch_cb_sub_model)
    ;
}