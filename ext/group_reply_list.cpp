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
#include <pybind11/stl.h>

namespace py = pybind11;

void export_group(py::module &m);
void export_group_reply(py::module &m);

void export_group_reply_list(py::module &m) {
    typedef std::vector<Tango::GroupCmdReply> StdGroupCmdReplyVector;
    typedef std::vector<Tango::GroupAttrReply> StdGroupAttrReplyVector;

    py::class_<Tango::GroupReplyList, std::vector<Tango::GroupReply>>(m, "GroupReplyList")
        .def(py::init<>())
        .def("has_failed", &Tango::GroupReplyList::has_failed)
        .def("reset", &Tango::GroupReplyList::reset)
        .def("push_back", &Tango::GroupReplyList::push_back)
    ;
    py::class_<Tango::GroupCmdReplyList, StdGroupCmdReplyVector>(m, "GroupCmdReplyList")
        .def(py::init<>())
        .def("has_failed", &Tango::GroupCmdReplyList::has_failed)
        .def("reset", &Tango::GroupCmdReplyList::reset)
        .def("push_back", &Tango::GroupCmdReplyList::push_back)
    ;
    py::class_<Tango::GroupAttrReplyList, StdGroupAttrReplyVector>(m, "GroupAttrReplyList")
        .def(py::init<>())
        .def("has_failed", &Tango::GroupAttrReplyList::has_failed)
        .def("reset", &Tango::GroupAttrReplyList::reset)
        .def("push_back", &Tango::GroupAttrReplyList::push_back)
    ;
}
