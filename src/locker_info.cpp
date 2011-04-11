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
#include <tango.h>

using namespace boost::python;

struct PyLockerInfo
{
    static inline object get_locker_id(Tango::LockerInfo &li)
    {
        return (li.ll == Tango::CPP) ? 
            object(li.li.LockerPid) :
            tuple(li.li.UUID);
    }
};

void export_locker_info()
{
    class_<Tango::LockerInfo>("LockerInfo",
        "A structure with information about the locker\n"
        "with the folowing members,\n"
        " - ll : (PyTango.LockerLanguage) the locker language\n"
        " - li : (pid_t / UUID) the locker id\n"
        " - locker_host : (string) the host\n"
        " - locker_class : (string) the class\n"
        "\npid_t should be an int, UUID should be a tuple of four numbers.\n"
        "\nNew in PyTango 7.0.0"
        )
        .def_readonly("ll", &Tango::LockerInfo::ll)
        .add_property("li", &PyLockerInfo::get_locker_id)
        .def_readonly("locker_host", &Tango::LockerInfo::locker_host)
        .def_readonly("locker_class", &Tango::LockerInfo::locker_class)
    ;
}
