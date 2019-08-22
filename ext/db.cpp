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

void export_db(py::module& m) {
// Note: DbDatum in python is extended to support the python sequence API
//       in the file ../PyTango/db.py. This way the DbDatum behaves like a
//       sequence of strings. This allows the user to work with a DbDatum as
//       if it was working with the old list of strings

    py::class_<Tango::DbDatum>(m, "DbDatum")
        .def(py::init<>())
        .def(py::init<const std::string>())
        .def(py::init<const Tango::DbDatum &>())
        .def_property("name", [](Tango::DbDatum& self) -> std::string {
            return self.name;
        },[](Tango::DbDatum& self, std::string& nam) -> void {
            self.name = nam;
        })
        .def_property("value_string", [](Tango::DbDatum& self) -> py::list {
            py::list py_list;
            for(auto& item : self.value_string)
                py_list.append(item);
            return py_list;
        },[](Tango::DbDatum& self, py::list py_list) -> void {
            for(auto& item : py_list)
                self.value_string.push_back(item.cast<std::string>());
        })
        .def("size", [](Tango::DbDatum& self) -> size_t {
            return self.size();
        })
        .def("is_empty", [](Tango::DbDatum& self) -> bool {
            return self.is_empty();
        })
    ;

    py::class_<Tango::DbDevExportInfo>(m, "DbDevExportInfo")
        .def_readwrite("name", &Tango::DbDevExportInfo::name)
        .def_readwrite("ior", &Tango::DbDevExportInfo::ior)
        .def_readwrite("host", &Tango::DbDevExportInfo::host)
        .def_readwrite("version", &Tango::DbDevExportInfo::version)
        .def_readwrite("pid", &Tango::DbDevExportInfo::pid)
    ;

     py::class_<Tango::DbDevImportInfo>(m, "DbDevImportInfo")
        .def_readonly("name", &Tango::DbDevImportInfo::name)
        .def_readonly("exported", &Tango::DbDevImportInfo::exported)
        .def_readonly("ior", &Tango::DbDevImportInfo::ior)
        .def_readonly("version", &Tango::DbDevImportInfo::version)
    ;

    py::class_<Tango::DbDevFullInfo, Tango::DbDevImportInfo>(m, "DbDevFullInfo")
        .def_readonly("class_name", &Tango::DbDevFullInfo::class_name)
        .def_readonly("ds_full_name", &Tango::DbDevFullInfo::ds_full_name)
        .def_readonly("started_date", &Tango::DbDevFullInfo::started_date)
        .def_readonly("stopped_date", &Tango::DbDevFullInfo::stopped_date)
        .def_readonly("pid", &Tango::DbDevFullInfo::pid)
    ;

    py::class_<Tango::DbDevInfo>(m, "DbDevInfo")
        .def_readwrite("name", &Tango::DbDevInfo::name)
        .def_readwrite("_class", &Tango::DbDevInfo::_class)
        .def_readwrite("klass", &Tango::DbDevInfo::_class)
        .def_readwrite("server", &Tango::DbDevInfo::server)
    ;

    py::class_<Tango::DbHistory>(m, "DbHistory")
        .def(py::init<std::string, std::string, std::vector<std::string> &>())
        .def(py::init<std::string, std::string, std::string, std::vector<std::string> &>())
        .def("get_name", &Tango::DbHistory::get_name)
        .def("get_attribute_name", &Tango::DbHistory::get_attribute_name)
        .def("get_date", &Tango::DbHistory::get_date)
        .def("get_value", &Tango::DbHistory::get_value)
        .def("is_deleted", &Tango::DbHistory::is_deleted)
    ;

    py::class_<Tango::DbServerInfo>(m, "DbServerInfo")
        .def_readwrite("name", &Tango::DbServerInfo::name)
        .def_readwrite("host", &Tango::DbServerInfo::host)
        .def_readwrite("mode", &Tango::DbServerInfo::mode)
        .def_readwrite("level", &Tango::DbServerInfo::level)
    ;

    py::class_<Tango::DbServerData>(m, "DbServerData")
        .def(py::init<const std::string, const std::string>())
        .def("get_name", &Tango::DbServerData::get_name)
        .def("put_in_database", &Tango::DbServerData::put_in_database)
        .def("already_exist", &Tango::DbServerData::already_exist)
        .def("remove",
             (void (Tango::DbServerData::*) ())
             &Tango::DbServerData::remove)
        .def("remove",
             (void (Tango::DbServerData::*) (const std::string& ))
             &Tango::DbServerData::remove)
    ;
}
