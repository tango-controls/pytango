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

#include "pytgutils.h"
#include "device_attribute.h"

void export_group_reply_list()
{
    using namespace boost::python;

    typedef std::vector<Tango::GroupReply> StdGroupReplyVector_;
    typedef std::vector<Tango::GroupCmdReply> StdGroupCmdReplyVector_;
    typedef std::vector<Tango::GroupAttrReply> StdGroupAttrReplyVector_;
    
    class_<Tango::GroupReplyList, bases<StdGroupReplyVector_> > GroupReplyList(
        "GroupReplyList",
        init<>())
    ;
    GroupReplyList
        .def("has_failed", &Tango::GroupReplyList::has_failed)
        .def("reset", &Tango::GroupReplyList::reset)
        .def("push_back", &Tango::GroupReplyList::push_back)
    ;

    class_<Tango::GroupCmdReplyList, bases<StdGroupCmdReplyVector_> > GroupCmdReplyList(
        "GroupCmdReplyList",
        init<>())
    ;
    GroupCmdReplyList
        .def("has_failed", &Tango::GroupCmdReplyList::has_failed)
        .def("reset", &Tango::GroupCmdReplyList::reset)
        .def("push_back", &Tango::GroupCmdReplyList::push_back)
    ;

    class_<Tango::GroupAttrReplyList, bases<StdGroupAttrReplyVector_> > GroupAttrReplyList(
        "GroupAttrReplyList",
        init<>())
    ;
    GroupAttrReplyList
        .def("has_failed", &Tango::GroupAttrReplyList::has_failed)
        .def("reset", &Tango::GroupAttrReplyList::reset)
        .def("push_back", &Tango::GroupAttrReplyList::push_back)
    ;
}
