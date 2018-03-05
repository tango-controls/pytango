/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>
#include <cstring>

namespace py = pybind11;

void export_dev_error(py::module &m)
{
    py::class_<Tango::DevError>(m, "DevError")
        .def_property("reason", [](Tango::DevError &self) {
                return py::str(self.reason);
        }, [](Tango::DevError &self, std::string& res) {
            // strdup here cos its a crappy corba string
            self.reason = ::strdup(res.c_str());
        })
        .def_readwrite("severity", &Tango::DevError::severity)
        .def_property("desc", [](Tango::DevError &self) {
                return py::str(self.desc);
        }, [](Tango::DevError &self, std::string& des) {
            self.desc = ::strdup(des.c_str());
        })
        .def_property("origin", [](Tango::DevError &self) {
                return py::str(self.origin);
        }, [](Tango::DevError &self, std::string& orig) {
            self.origin = ::strdup(orig.c_str());
        })
        .def(py::pickle(
            [](const Tango::DevError &p) { //__getstate__
                // TODO check we may need to strdup here!!!
                return py::make_tuple(p.reason,
                        p.severity,
                        p.desc,
                        p.origin);
            },
            [](py::tuple t) { //__setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");
                Tango::DevError p = Tango::DevError();
                p.reason = ::strdup(t[0].cast<std::string>().c_str());
                p.severity = t[1].cast<Tango::ErrSeverity>();
                p.desc = ::strdup(t[2].cast<std::string>().c_str());
                p.origin = ::strdup(t[3].cast<std::string>().c_str());
                return p;
            }
        ));
    ;
}
