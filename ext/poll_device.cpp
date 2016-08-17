/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include <tango.h>

using namespace boost::python;

void export_poll_device()
{
    class_<Tango::PollDevice>("PollDevice",
        "A structure containing PollDevice information\n"
        "the following members,\n"
        " - dev_name : string\n"
        " - ind_list : sequence<long>\n"
        "\nNew in PyTango 7.0.0")
        .def_readwrite("dev_name", &Tango::PollDevice::dev_name)
        .def_readwrite("ind_list", &Tango::PollDevice::ind_list)
    ;
}
