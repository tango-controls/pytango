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

namespace py = pybind11;

void export_attribute_proxy(py::module& m)
{
    py::class_<Tango::AttributeProxy, std::shared_ptr<Tango::AttributeProxy>>(m, "__AttributeProxy")
        .def(py::init<const Tango::AttributeProxy &>())
        .def(py::init([](const std::string& name) {
            return std::shared_ptr<Tango::AttributeProxy>(new Tango::AttributeProxy(name.c_str()));
        }))
        .def(py::init([](const Tango::DeviceProxy *dev, const std::string& name) {
            return std::shared_ptr<Tango::AttributeProxy>(new Tango::AttributeProxy(dev, name.c_str()));
        }))
        //
        // general methods
        //
        .def("name", [](Tango::AttributeProxy& self) -> std::string {
            return self.name();
        })
        .def("get_device_proxy", [] (Tango::AttributeProxy& self) -> Tango::DeviceProxy* {
            return self.get_device_proxy();
        })
        //
        // property methods
        //
        .def("_get_property", [] (Tango::AttributeProxy& self, std::string& prop_name, Tango::DbData& db) -> void {
            self.get_property(prop_name, db);
        })
        .def("_get_property", [] (Tango::AttributeProxy& self, std::vector<std::string>& prop_names, Tango::DbData& db) -> void {
            self.get_property(prop_names, db);
        })
        .def("_get_property", [] (Tango::AttributeProxy& self, Tango::DbData& db) -> void {
            self.get_property(db);
        })
        .def("_put_property", [] (Tango::AttributeProxy& self, Tango::DbData& db) -> void {
            self.put_property(db);
        })
        .def("_delete_property", [] (Tango::AttributeProxy& self, std::string& prop_name) -> void {
            self.delete_property(prop_name);
        })
        .def("_delete_property", [] (Tango::AttributeProxy& self, std::vector<std::string>& prop_names) -> void {
            self.delete_property(prop_names);
        })
        .def("_delete_property", [] (Tango::AttributeProxy& self, Tango::DbData& db) -> void {
            self.delete_property(db);
        })
        //
        // Pickle
        //
        .def(py::pickle(
            [](Tango::AttributeProxy &self) { //__getstate__
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

