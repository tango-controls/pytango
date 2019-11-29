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

void export_attribute_proxy(py::module& m)
{
    py::class_<Tango::AttributeProxy>(m, "__AttributeProxy")
        .def(py::init([](std::string& name) {
            return new Tango::AttributeProxy(name);
        }))
        //
        // general methods
        //
        .def("name", [](Tango::AttributeProxy& self) -> std::string {
            return self.name();
        })
        .def("get_device_proxy", [](Tango::AttributeProxy& self) -> Tango::DeviceProxy* {
            return self.get_device_proxy();
        })
        //
        // property methods
        //
        .def("_get_property", [](Tango::AttributeProxy& self, std::string& prop_name, Tango::DbData& db_data) -> std::vector<Tango::DbDatum> {
            self.get_property(prop_name, db_data);
            return db_data;
        })
        .def("_get_property", [](Tango::AttributeProxy& self, std::vector<std::string>& prop_names, Tango::DbData& db_data) -> std::vector<Tango::DbDatum> {
            self.get_property(prop_names, db_data);
            return db_data;
        })
        .def("_get_property", [](Tango::AttributeProxy& self, Tango::DbData& db_data) -> std::vector<Tango::DbDatum> {
            self.get_property(db_data);
            return db_data;
        })
        .def("_put_property", [](Tango::AttributeProxy& self, Tango::DbData& db_data) -> void {
            self.put_property(db_data);
        })
        .def("_delete_property", [](Tango::AttributeProxy& self, std::string& prop_name) -> void {
            self.delete_property(prop_name);
        })
        .def("_delete_property", [](Tango::AttributeProxy& self, std::vector<std::string>& prop_names) -> void {
            self.delete_property(prop_names);
        })
        .def("_delete_property", [](Tango::AttributeProxy& self, std::vector<Tango::DbDatum>& db_data) -> void {
            py::print(db_data);
            self.delete_property(db_data);
        })
        //
        // Pickle
        //
        .def(py::pickle(
            [](Tango::AttributeProxy& self) { //__getstate__
                Tango::DeviceProxy* dp = self.get_device_proxy();
                return py::make_tuple(dp->get_db_host() + ":" + dp->get_db_port() +
                        "/" + dp->dev_name() + "/" + self.name());
            },
            [](py::tuple t) { //__setstate__
                throw std::runtime_error("setstate not implemented");
                return nullptr;
            }
        ))

    ;
}

