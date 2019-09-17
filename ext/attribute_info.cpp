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

namespace py = pybind11;

void export_attribute_info(py::module &m) {
    py::class_<Tango::AttributeInfo, Tango::DeviceAttributeConfig>(m, "AttributeInfo")
        .def(py::init<const Tango::AttributeInfo&>())
        .def_readwrite("disp_level", &Tango::AttributeInfo::disp_level)
        .def(py::pickle(
            [](const Tango::AttributeInfo &p) { //__getstate__
                return py::make_tuple(p.disp_level);
            },
            [](py::tuple t) { //__setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                Tango::AttributeInfo p = Tango::AttributeInfo();
                p.disp_level = t[0].cast<Tango::DispLevel>();
                return p;
            }
        ));
    ;
}
