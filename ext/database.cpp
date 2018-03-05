/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pyutils.h>

namespace py = pybind11;

const std::string param_numb_or_str_numb = "Second parameter must be an int or a "
                                            "string representing an int";

void export_database(py::module &m)
{
    py::class_<Tango::Database, Tango::Connection>(m, "Database")
        .def(py::init<>())
        .def(py::init<const Tango::Database &>())

        .def("__init__", [](const std::string &host, int port) {
            std::shared_ptr<Tango::Database>(
            new Tango::Database(const_cast<std::string&>(host), port));
        })

        .def("__init__", [](const std::string &host, const std::string &port_str){
            std::istringstream port_stream(port_str);
            int port = 0;
            if(!(port_stream >> port)) {
                raise_(PyExc_TypeError, param_numb_or_str_numb);
            }
            std::shared_ptr<Tango::Database>(
                new Tango::Database(const_cast<std::string&>(host), port));
        })

        .def("__init__", [](const std::string &filename){
            std::shared_ptr<Tango::Database>(
                new Tango::Database(const_cast<std::string&>(filename)));
        })

        //
        // general methods
        //
        .def("dev_name", [](Tango::Database& self){
            Tango::Connection *conn = static_cast<Tango::Connection *>(&self);
            return conn->dev_name();
        })
        .def("write_filedatabase", &Tango::Database::write_filedatabase)
        .def("reread_filedatabase", &Tango::Database::write_filedatabase)
        .def("build_connection", &Tango::Database::write_filedatabase)
        .def("check_tango_host", &Tango::Database::check_tango_host)
        .def("check_access_control", &Tango::Database::check_access_control)
        .def("is_control_access_checked",
            &Tango::Database::is_control_access_checked)
        .def("set_access_checked",
            &Tango::Database::set_access_checked)
        .def("get_access_except_errors",
            &Tango::Database::get_access_except_errors,
            py::return_value_policy::reference_internal)
        .def("is_multi_tango_host", &Tango::Database::is_multi_tango_host)
        .def("get_file_name", &Tango::Database::get_file_name,
            py::return_value_policy::copy)

        .def("get_info",&Tango::Database::get_info)
        .def("get_host_list",
            (Tango::DbDatum (Tango::Database::*) ())
            &Tango::Database::get_host_list)
        .def("get_host_list",
            (Tango::DbDatum (Tango::Database::*) (std::string &))
            &Tango::Database::get_host_list)
        .def("get_services",
            (Tango::DbDatum (Tango::Database::*) (const std::string &, const std::string &))
            &Tango::Database::get_services)
        .def("get_device_service_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_device_service_list)
        .def("register_service",
            (void (Tango::Database::*) (const std::string &, const std::string &, const std::string &))
            &Tango::Database::register_service)
        .def("unregister_service",
            (void (Tango::Database::*) (const std::string &, const std::string &))
            &Tango::Database::unregister_service)

        //
        // Device methods
        //
        .def("add_device", &Tango::Database::add_device)
        .def("delete_device", &Tango::Database::delete_device)
        .def("import_device", 
            (Tango::DbDevImportInfo (Tango::Database::*) (const std::string &))
            &Tango::Database::import_device)
        .def("export_device", &Tango::Database::export_device)
        .def("unexport_device", &Tango::Database::unexport_device)
        .def("get_device_info", 
            (Tango::DbDevFullInfo (Tango::Database::*) (const std::string &))
            &Tango::Database::get_device_info)
        .def("get_device_name",
            (Tango::DbDatum (Tango::Database::*) (string &, string &))
            &Tango::Database::get_device_name)
        .def("get_device_exported",
            (Tango::DbDatum (Tango::Database::*) (const string &))
            &Tango::Database::get_device_exported)
        .def("get_device_domain",
            (Tango::DbDatum (Tango::Database::*) (const string &))
            &Tango::Database::get_device_domain)
        .def("get_device_family",
            (Tango::DbDatum (Tango::Database::*) (const string &))
            &Tango::Database::get_device_family)
        .def("get_device_member",
            (Tango::DbDatum (Tango::Database::*) (const string &))
            &Tango::Database::get_device_member)
        .def("get_device_alias", [](Tango::Database& self, const std::string &alias) {
            std::string devname;
            self.get_device_alias(alias, devname);
            return devname;
        })
        .def("get_alias", [](Tango::Database& self, const std::string &devname) {
            std::string alias;
            self.get_alias(devname, alias);
            return alias;
        })
        .def("get_device_alias_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_device_alias_list)
        .def("get_class_for_device",
            (std::string (Tango::Database::*) (const std::string &))
            &Tango::Database::get_class_for_device)
        .def("get_class_inheritance_for_device",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_class_inheritance_for_device)
        .def("get_device_exported_for_class",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_device_exported_for_class)
        .def("put_device_alias",
            (void (Tango::Database::*) (const std::string &, const std::string &))
            &Tango::Database::put_device_alias)
        .def("delete_device_alias",
            (void (Tango::Database::*) (const std::string &))
            &Tango::Database::delete_device_alias)
        
        //
        // server methods
        //
        .def("_add_server",
            (void (Tango::Database::*) (const std::string &, Tango::DbDevInfos &))
            &Tango::Database::add_server)
        .def("delete_server",
            (void (Tango::Database::*) (const std::string &))
            &Tango::Database::delete_server)
        .def("_export_server", &Tango::Database::export_server)
        .def("unexport_server",
            (void (Tango::Database::*) (const std::string &))
            &Tango::Database::unexport_server)
        .def("rename_server", &Tango::Database::rename_server)
        .def("get_server_info",
            (Tango::DbServerInfo (Tango::Database::*) (const std::string &))
            &Tango::Database::get_server_info)
        .def("put_server_info", &Tango::Database::put_server_info)
        .def("delete_server_info",
            (void (Tango::Database::*) (const std::string &))
            &Tango::Database::delete_server_info)
        .def("get_server_class_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_server_class_list)
        .def("get_server_name_list", &Tango::Database::get_server_name_list)
        .def("get_instance_name_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_instance_name_list)
        .def("get_server_list",
            (Tango::DbDatum (Tango::Database::*) ())
            &Tango::Database::get_server_list)
        .def("get_server_list",
            (Tango::DbDatum (Tango::Database::*) (std::string &))
            &Tango::Database::get_server_list)
        .def("get_host_server_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_host_server_list)
        .def("get_device_class_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_device_class_list)
        .def("get_server_release", &Tango::Database::get_server_release)
        
        //
        // property methods
        //
        .def("_get_property",
            (void (Tango::Database::*) (std::string, Tango::DbData &))
            &Tango::Database::get_property)
        .def("_get_property_forced", &Tango::Database::get_property_forced)
        .def("_put_property", &Tango::Database::put_property)
        .def("_delete_property", &Tango::Database::delete_property)
        .def("get_property_history",
            (std::vector<Tango::DbHistory> (Tango::Database::*) (const std::string &, const std::string &))
            &Tango::Database::get_property_history)
        .def("get_object_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_object_list)
        .def("get_object_property_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &, const std::string &))
            &Tango::Database::get_object_property_list)
        .def("_get_device_property",
            (void (Tango::Database::*) (std::string, Tango::DbData &))
            &Tango::Database::get_device_property)
        .def("_put_device_property", &Tango::Database::put_device_property)
        .def("_delete_device_property", &Tango::Database::delete_device_property)
        .def("get_device_property_history",
            (std::vector<Tango::DbHistory>(Tango::Database::*) (const std::string &, const std::string &))
            &Tango::Database::get_device_property_history)
        .def("_get_device_property_list",
            (Tango::DbDatum (Tango::Database::*) (std::string &, std::string &))
            &Tango::Database::get_device_property_list)
        .def("_get_device_property_list", [](Tango::Database& self, const std::string &devname,
                const std::string &wildcard, std::vector<std::string> &d) {
            self.get_device_property_list(const_cast<std::string&>(devname), wildcard, d);
        })
        .def("_get_device_attribute_property",
            (void (Tango::Database::*) (std::string, Tango::DbData &))
            &Tango::Database::get_device_attribute_property)
        .def("_put_device_attribute_property",
            &Tango::Database::put_device_attribute_property)
        .def("_delete_device_attribute_property",
            &Tango::Database::delete_device_attribute_property)
        .def("get_device_attribute_property_history",
            (std::vector<Tango::DbHistory> (Tango::Database::*) (const std::string &, const std::string &, const std::string &))
            &Tango::Database::get_device_attribute_property_history)
        .def("_get_class_property",
            (void (Tango::Database::*) (std::string, Tango::DbData &))
            &Tango::Database::get_class_property)
        .def("_put_class_property", &Tango::Database::put_class_property)
        .def("_delete_class_property", &Tango::Database::delete_class_property)
        .def("get_class_property_history",
            (std::vector<Tango::DbHistory> (Tango::Database::*) (const std::string &, const std::string &))
            &Tango::Database::get_class_property_history)
        .def("get_class_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_class_list)
        .def("get_class_property_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_class_property_list)
        .def("_get_class_attribute_property",
            (void (Tango::Database::*) (std::string, Tango::DbData &))
            &Tango::Database::get_class_attribute_property)
        .def("_put_class_attribute_property",
            &Tango::Database::put_class_attribute_property)
        .def("_delete_class_attribute_property",
            &Tango::Database::delete_class_attribute_property)
        .def("get_class_attribute_property_history",
            (std::vector<Tango::DbHistory>(Tango::Database::*) (const std::string &, const std::string &, const std::string &))
            &Tango::Database::get_class_attribute_property_history)

        .def("get_class_attribute_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &, const std::string &))
            &Tango::Database::get_class_attribute_list)

        //
        // Attribute methods
        //
        .def("get_attribute_alias", [](Tango::Database& self, const std::string &alias) {
            std::string attrname;
            self.get_attribute_alias(alias, attrname);
            return attrname;
        })
        .def("get_attribute_alias_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            &Tango::Database::get_attribute_alias_list)
        .def("put_attribute_alias",
            (void (Tango::Database::*) (const std::string &, const std::string &))
            &Tango::Database::put_attribute_alias)
        .def("delete_attribute_alias",
            (void (Tango::Database::*) (const std::string &))
            &Tango::Database::delete_attribute_alias)

        //
        // event methods
        //
        .def("export_event", [](Tango::Database& self, const py::object &obj) {
            Tango::DevVarStringArray par;
//grm            convert2array(obj, par);
            self.export_event(&par);
        })
        .def("unexport_event",
            (void (Tango::Database::*) (const std::string &))
            &Tango::Database::unexport_event)

        //
        // alias methods
        //
        .def("get_device_from_alias", [](Tango::Database& self, const std::string &input) -> std::string {
            std::string output;
            self.get_device_from_alias(input, output);
            return output;
        })
        .def("get_alias_from_device", [](Tango::Database& self, const std::string &input) {
            std::string output;
            self.get_alias_from_device(input, output);
            return output;
        })
        .def("get_attribute_from_alias", [](Tango::Database& self, const std::string &input) {
            std::string output;
            self.get_attribute_from_alias(input, output);
            return output;
        })
        .def("get_alias_from_attribute", [](Tango::Database& self, const std::string &input) {
            std::string output;
            self.get_alias_from_attribute(input, output);
            return output;
        })
        .def(py::pickle(
            [](Tango::Database &p) { //__getstate__
                std::string& host = p.get_db_host();
                int port = p.get_db_port_num();
                return py::make_tuple(host, port);
            },
            [](py::tuple t) { //__setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");
                std::string host = t[0].cast<std::string>();
                int port = t[1].cast<int>();
                Tango::Database p = Tango::Database(host, port);
                return p;
            }
        ));
        ;
}

