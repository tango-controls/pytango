/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

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
    py::class_<Tango::Database>(m, "Database")
        .def(py::init<>())
        .def(py::init<const Tango::Database &>())

        .def(py::init([](const std::string& host, int port) {
//            return std::shared_ptr<Tango::Database>(
            return new Tango::Database(const_cast<std::string&>(host), port);
        }))

        .def(py::init([](const std::string& host, const std::string& port_str){
            std::istringstream port_stream(port_str);
            int port = 0;
            if(!(port_stream >> port)) {
                raise_(PyExc_TypeError, param_numb_or_str_numb);
            }
//            std::shared_ptr<Tango::Database>(
            return new Tango::Database(const_cast<std::string&>(host), port);
        }))

        .def(py::init([](const std::string& filename){
//            std::shared_ptr<Tango::Database>(
            return new Tango::Database(const_cast<std::string&>(filename));
        }))

        //
        // general methods
        //
        .def("get_db_host", [](Tango::Database& self) -> std::string {
            return self.get_db_host();
        })

        .def("get_db_port", [](Tango::Database& self) -> std::string {
            return self.get_db_port();
        })

        .def("get_db_port_num", [](Tango::Database& self) -> long {
            return self.get_db_port_num();
        })

        .def("get_from_env_var", [](Tango::Database& self) -> bool {
            return self.get_from_env_var();
        })

        .def("get_fqdn", [](Tango::Database& self) -> std::string {
            std::string fqdn_str;
            self.get_fqdn(fqdn_str);
            return fqdn_str;
        })

        .def("is_dbase_used", [](Tango::Database& self) -> bool {
            return self.is_dbase_used();
        })
        .def("get_dev_host", [](Tango::Database& self) -> std::string {
            return self.get_dev_host();
        })
        .def("get_dev_port", [](Tango::Database& self) -> std::string {
            return self.get_dev_port();
        })
        .def("dev_name", [](Tango::Database& self) -> std::string {
           Tango::Connection *conn = static_cast<Tango::Connection*>(&self);
           return conn->dev_name(); // C++ signature
        })
        .def("write_filedatabase", [](Tango::Database& self) -> void {
            self.write_filedatabase(); // C++ signature
        })
        .def("reread_filedatabase", [](Tango::Database& self) -> void {
            self.reread_filedatabase(); // C++ signature
        })
        .def("build_connection", [](Tango::Database& self) -> void {
            self.build_connection(); // C++ signature
        })
        .def("check_tango_host", [](Tango::Database& self, std::string& tango_host_env) -> void {
            self.check_tango_host(tango_host_env.c_str()); // C++ signature
        })
        .def("check_access_control", [](Tango::Database& self, std::string& dev_name) -> Tango::AccessControlType {
            return self.check_access_control(dev_name); // C++ signature
        })
        .def("is_control_access_checked", [](Tango::Database& self) -> bool {
            return self.is_control_access_checked(); // C++ signature
        })
        .def("set_access_checked", [](Tango::Database& self, bool checked) -> void {
            return self.set_access_checked(checked); // C++ signature
        })
        .def("get_access_except_errors", [](Tango::Database& self) -> Tango::DevErrorList& {
            return self.get_access_except_errors(); // C++ signature
        })
        .def("is_multi_tango_host", [](Tango::Database& self) -> bool {
            return self.is_multi_tango_host(); // C++ signature
        })
        .def("get_multi_host", [](Tango::Database& self) -> std::vector<std::string> {
            return self.get_multi_host(); // C++ signature
        })
        .def("get_multi_port", [](Tango::Database& self) -> std::vector<std::string> {
            return self.get_multi_port(); // C++ signature
        })
        .def("get_file_name", [](Tango::Database& self) -> std::string {
            return self.get_file_name();  // C++ signature
        })
        .def("get_info", [](Tango::Database& self) -> std::string {
            return self.get_info(); // C++ signature
        })
        .def("get_host_list", [](Tango::Database& self) -> Tango::DbDatum {
            return self.get_host_list();  // C++ signature
        })
        .def("get_host_list", [](Tango::Database& self, std::string& wildcard) -> Tango::DbDatum {
            return self.get_host_list();  // C++ signature
        })
        .def("get_services", [](Tango::Database& self, std::string& service_name, std::string& inst_name) -> Tango::DbDatum {
            return self.get_services(service_name, inst_name); // C++ signature
        })
        .def("get_device_service_list", [](Tango::Database& self, std::string& service_name) -> Tango::DbDatum {
            return self.get_device_service_list(service_name); // C++ signature
        })
        .def("register_service", [](Tango::Database& self, std::string& service_name,
                std::string& inst_name, std::string& dev_name) -> void {
            self.register_service(service_name, inst_name, dev_name); // C++ signature
        })
        .def("unregister_service", [](Tango::Database& self, std::string& service_name, std::string& inst_name) -> void {
            self.unregister_service(service_name, inst_name); // C++ signature
        })
        //
        // Device methods
        //
        .def("add_device", [](Tango::Database& self, Tango::DbDevInfo& dev_info) -> void {
            self.add_device(dev_info); // C++ signature
        })
        .def("delete_device", [](Tango::Database& self, std::string dev_name) -> void {
            self.delete_device(dev_name); // C++ signature
        })
        .def("import_device", [](Tango::Database& self, std::string& dev_name) -> Tango::DbDevImportInfo {
            return self.import_device(dev_name); // C++ signature
        })
        .def("export_device", [](Tango::Database& self, Tango::DbDevExportInfo &info) -> void {
            self.export_device(info); // C++ signature
        })
        .def("unexport_device", [](Tango::Database& self, std::string& dev_name) -> void {
            self.unexport_device(dev_name); // C++ signature
        })
        .def("get_device_info", [](Tango::Database& self, std::string& dev_name) -> Tango::DbDevFullInfo {
            return self.get_device_info(dev_name); // C++ signature
        })
        .def("get_class_for_device", [](Tango::Database& self, std::string& dev_name) -> std::string {
            return self.get_class_for_device(dev_name); // C++ signature
        })
        .def("get_class_inheritance_for_device", [](Tango::Database& self, std::string& dev_name) -> Tango::DbDatum {
            return self.get_class_inheritance_for_device(dev_name); // C++ signature
        })
        .def("get_device_name", [](Tango::Database& self, std::string& ds_name, std::string& class_name) -> Tango::DbDatum {
            return self.get_device_name(ds_name, class_name); // C++ signature
        })
        .def("get_device_exported", [](Tango::Database& self, std::string& filter) -> Tango::DbDatum {
            return self.get_device_exported(filter); // C++ signature
        })
        .def("get_device_domain", [](Tango::Database& self, std::string& wildcard) -> Tango::DbDatum {
            return self.get_device_domain(wildcard); // C++ signature
        })
        .def("get_device_family", [](Tango::Database& self, std::string& wildcard) -> Tango::DbDatum {
            return self.get_device_family(wildcard); // C++ signature
        })
        .def("get_device_member", [](Tango::Database& self, std::string& wildcard) -> Tango::DbDatum {
            return self.get_device_member(wildcard); // C++ signature
        })
        .def("get_device_class_list", [](Tango::Database& self, std::string& ds_name) -> Tango::DbDatum {
            return self.get_device_class_list(ds_name); // C++ signature
        })
        .def("get_device_exported_for_class", [](Tango::Database& self, std::string& class_name) -> Tango::DbDatum {
            return self.get_device_exported_for_class(class_name); // C++ signature
        })
        .def("get_object_list", [](Tango::Database& self, std::string& wildcard) -> Tango::DbDatum {
            return self.get_object_list(wildcard);  // Tango C++ signature
        })
        .def("get_object_property_list", [](Tango::Database& self, std::string& obj_name, std::string& wildcard) -> Tango::DbDatum {
            return self.get_object_property_list(obj_name, wildcard);  // Tango C++ signature
        })
        .def("get_class_list", [](Tango::Database& self, std::string& wildcard) -> Tango::DbDatum {
            return self.get_class_list(wildcard);  // Tango C++ signature
        })
        .def("get_class_property_list", [](Tango::Database& self, std::string& class_name) -> Tango::DbDatum {
            return self.get_class_property_list(class_name); // C++ signature
        })
        .def("get_class_attribute_list", [](Tango::Database& self, std::string& class_name, std::string& wildcard) -> Tango::DbDatum {
            return self.get_class_attribute_list(class_name, wildcard);  // Tango C++ signature
        })
        .def("get_class_pipe_list", [](Tango::Database& self, std::string& class_name, std::string& wildcard) -> Tango::DbDatum {
            return self.get_class_pipe_list(class_name, wildcard);  // Tango C++ signature
        })
        .def("get_device_alias_list", [](Tango::Database& self, std::string& filter) -> Tango::DbDatum {
            return self.get_device_alias_list(filter);  // Tango C++ signature
        })
        .def("get_attribute_alias_list", [](Tango::Database& self, std::string& filter) -> Tango::DbDatum {
            return self.get_attribute_alias_list(filter);  // Tango C++ signature
        })
        //
        // server methods
        //
        .def("_add_server", [](Tango::Database& self, std::string& ds_name, Tango::DbDevInfos& devs) -> void {
            self.add_server(ds_name, devs); // C++ signature
        })
        .def("delete_server", [](Tango::Database& self, std::string& ds_name) -> void {
            self.delete_server(ds_name); // C++ signature
        })
        .def("_export_server", [](Tango::Database& self, Tango::DbDevExportInfos &devs) -> void {
            self.export_server(devs); // C++ signature
        })
        .def("unexport_server", [](Tango::Database& self, std::string& ds_name) -> void {
            self.unexport_server(ds_name); // C++ signature
        })
        .def("rename_server", [](Tango::Database& self, std::string& old_ds_name, std::string& new_ds_name) -> void {
            self.rename_server(old_ds_name, new_ds_name); // C++ signature
        })
        .def("get_server_info", [](Tango::Database& self, std::string& name) -> Tango::DbServerInfo {
            return self.get_server_info(name); // C++ signature
        })
        .def("put_server_info", [](Tango::Database& self, Tango::DbServerInfo& info) -> void {
            self.put_server_info(info); // C++ signature
        })
        .def("delete_server_info", [](Tango::Database& self, std::string& info_name) -> void {
            self.delete_server_info(info_name); // C++ signature
        })
        .def("get_server_class_list", [](Tango::Database& self, std::string &ds_name) -> Tango::DbDatum {
            return self.get_server_class_list(ds_name); // C++ signature
        })
        .def("get_server_name_list", [](Tango::Database& self) -> Tango::DbDatum {
            return self.get_server_name_list(); // C++ signature
        })
        .def("get_instance_name_list", [](Tango::Database& self, std::string &ds_name) -> Tango::DbDatum {
            return self.get_instance_name_list(ds_name); // C++ signature
        })
        .def("get_server_list", [](Tango::Database& self) -> Tango::DbDatum {
            return self.get_server_list(); // C++ signature
        })
        .def("get_server_list", [](Tango::Database& self, std::string &wildcard) -> Tango::DbDatum {
            return self.get_server_list(wildcard); // C++ signature
        })
        .def("get_host_server_list", [](Tango::Database& self, std::string &host_name) -> Tango::DbDatum {
            return self.get_host_server_list(host_name); // C++ signature
        })
        .def("get_server_release", [](Tango::Database& self) -> int {
            self.get_server_release(); // C++ signature
        })
        //
        // property methods
        //
        .def("_get_property", [](Tango::Database& self, std::string obj_name, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
            self.get_property(obj_name, dbData); // C++ signature
            return dbData;
        })
        .def("_get_property_forced", [](Tango::Database& self, std::string obj_name, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
            self.get_property_forced(obj_name, dbData, NULL); // C++ signature
            return dbData;
        })
        .def("_put_property", [](Tango::Database& self, std::string obj_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.put_property(obj_name, dbData); // C++ signature
        })
        .def("_delete_property", [](Tango::Database& self, std::string obj_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.delete_property(obj_name, dbData); // C++ signature
        })
        .def("get_property_history", [](Tango::Database& self, std::string& obj_name, std::string& prop_name) -> std::vector<Tango::DbHistory> {
            return self.get_property_history(obj_name, prop_name);  // Tango C++ signature
        })
        .def("_get_device_property", [](Tango::Database& self, std::string& dev_name, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
            self.get_device_property(dev_name, dbData);  // Tango C++ signature
            return dbData;
        })
        .def("_put_device_property", [](Tango::Database& self, string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.put_device_property(dev_name, dbData);  // Tango C++ signature
        })
        .def("_delete_device_property", [](Tango::Database& self, string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.delete_device_property(dev_name, dbData);  // Tango C++ signature
        })
        .def("get_device_property_history", [](Tango::Database& self, std::string& dev_name, std::string& prop_name) -> std::vector<Tango::DbHistory> {
            return self.get_device_property_history(dev_name, prop_name);  // Tango C++ signature
        })
        .def("_get_device_property_list", [](Tango::Database& self, std::string& dev_name, std::string& wildcard) -> Tango::DbDatum {
            self.get_device_property_list(dev_name, wildcard);
        })
        .def("_get_device_property_list", [](Tango::Database& self, std::string& dev_name,
                std::string& wildcard, std::vector<std::string>& prop_list) -> std::vector<std::string> {
            self.get_device_property_list(dev_name, wildcard, prop_list);
            return prop_list;
        })
        .def("_get_device_attribute_property", [](Tango::Database& self, string dev_name, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
            self.get_device_attribute_property(dev_name, dbData); // C++ signature
            return dbData;
        })
        .def("_put_device_attribute_property", [](Tango::Database& self, string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.put_device_attribute_property(dev_name, dbData); // C++ signature
        })
        .def("_delete_device_attribute_property", [](Tango::Database& self, string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.delete_device_attribute_property(dev_name, dbData); // C++ signature
        })
        .def("get_device_attribute_property_history", [](Tango::Database& self, std::string& dev_name, std::string& prop_name, std::string& att_name) -> std::vector<Tango::DbHistory> {
            self.get_device_attribute_property_history(dev_name, prop_name, att_name);  // C++ signature
        })
        .def("get_device_attribute_list", [](Tango::Database& self, std::string& dev_name, std::vector<std::string>& att_list) -> std::vector<std::string> {
            self.get_device_attribute_list(dev_name, att_list); // C++ signature
        })
        .def("_get_class_property", [](Tango::Database& self, std::string& dev_class, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
            self.get_class_property(dev_class, dbData, NULL); // C++ signature
            return dbData;
        })
        .def("_put_class_property", [](Tango::Database& self, std::string& dev_class, std::vector<Tango::DbDatum>& dbData) -> void {
            self.put_class_property(dev_class, dbData); // C++ signature
        })
        .def("_delete_class_property", [](Tango::Database& self, std::string& dev_class, std::vector<Tango::DbDatum>& dbData) -> void {
            self.delete_class_property(dev_class, dbData); // C++ signature
        })
        .def("get_class_property_history", [](Tango::Database& self, std::string& class_name, std::string& prop_name) -> std::vector<Tango::DbHistory> {
            return self.get_class_property_history(class_name, prop_name); // C++ signature
        })
        .def("_get_class_attribute_property", [](Tango::Database& self, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
            self.get_class_attribute_property(class_name, dbData);  // C++ signature
            return dbData;
        })
        .def("_put_class_attribute_property", [](Tango::Database& self, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.put_class_attribute_property(class_name, dbData);  // C++ signature
        })
        .def("_delete_class_attribute_property", [](Tango::Database& self, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.delete_class_attribute_property(class_name, dbData);  // C++ signature
        })
        .def("get_class_attribute_property_history", [](Tango::Database& self, std::string& class_name, std::string& att_name,
                std::string& prop_name) -> std::vector<Tango::DbHistory> {
            return self.get_class_attribute_property_history(class_name, att_name, prop_name);  // C++ signature
        })
        //
        // event methods
        //
        .def("export_event", [](Tango::Database& self, std::vector<std::string>& event_data) {
            Tango::DevVarStringArray dvsa;
            dvsa.length(event_data.size());
            for (int i=0; i<event_data.size(); i++) {
                dvsa[i] = strdup(event_data[i].c_str());
            }
            self.export_event(&dvsa);
        })
        .def("unexport_event",[](Tango::Database& self, std::string &ev) -> void {
            self.unexport_event(ev); // C++ signature
        })
        //
        // pipe methods
        //
        .def("get_device_pipe_list", [](Tango::Database& self, std::string& dev_name, std::vector<std::string>& pipe_list) -> std::vector<std::string> {
            self.get_device_pipe_list(dev_name, pipe_list); // C++ signature
            return pipe_list;
        })
        .def("get_device_pipe_property", [](Tango::Database& self, std::string dev_name, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
            self.get_device_pipe_property(dev_name, dbData); // C++ signature
            return dbData;
        })
        .def("put_device_pipe_property", [](Tango::Database& self, std::string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.put_device_pipe_property(dev_name, dbData); // C++ signature
        })
        .def("delete_device_pipe_property", [](Tango::Database& self, std::string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.delete_device_pipe_property(dev_name, dbData); // C++ signature
        })
        .def("get_device_pipe_property_history", [](Tango::Database& self, std::string& dev_name,
               std::string& pipe_name, std::string& prop_name) -> std::vector<Tango::DbHistory> {
            return self.get_device_pipe_property_history(dev_name, pipe_name, prop_name); // C++ signature
        })
        .def("get_class_pipe_property", [](Tango::Database& self, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
            self.get_class_pipe_property(class_name, dbData); // C++ signature
            return dbData;
        })
        .def("put_class_pipe_property", [](Tango::Database& self, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.put_class_pipe_property(class_name, dbData); // C++ signature
        })
        .def("delete_class_pipe_property", [](Tango::Database& self, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> void {
            self.delete_class_pipe_property(class_name, dbData); // C++ signature
        })
        .def("get_class_pipe_property_history", [](Tango::Database& self, std::string& class_name,
                std::string& pipe_name, std::string& prop_name) -> std::vector<Tango::DbHistory> {
            return self.get_class_pipe_property_history(class_name, pipe_name, prop_name); // C++ signature
        })
        //
        // alias methods
        //
        .def("get_device_from_alias", [](Tango::Database& self, std::string& alias) -> std::string {
            std::string dev_name;
            self.get_device_from_alias(alias, dev_name); // C++ signature
            return std::move(dev_name);
        })
        .def("get_alias_from_device", [](Tango::Database& self, std::string& dev_name) -> std::string {
            std::string alias;
            self.get_alias_from_device(dev_name, alias);
            return std::move(alias);
        })
        .def("get_alias", [](Tango::Database& self, std::string& devname) -> std::string {
            std::string alias;
            self.get_alias(devname, alias); // C++ signature
            return std::move(alias);
        })
        .def("get_device_alias", [](Tango::Database& self, std::string& alias) -> std::string {
            std::string devname;
            self.get_device_alias(alias, devname); // C++ signature
            return std::move(devname);
        })
        .def("put_device_alias", [](Tango::Database& self, std::string& dev_name, std::string& dev_alias) -> void {
            self.put_device_alias(dev_name, dev_alias); // C++ signature
        })
        .def("delete_device_alias", [](Tango::Database& self, std::string& dev_alias) -> void {
            self.delete_device_alias(dev_alias); // C++ signature
        })
        .def("get_attribute_from_alias", [](Tango::Database& self, std::string& alias) -> std::string {
            std::string att_name;
            self.get_attribute_from_alias(alias, att_name); // C++ signature
            return std::move(att_name);
        })
        .def("get_alias_from_attribute", [](Tango::Database& self, std::string& att_name) -> std::string {
            std::string alias;
            self.get_alias_from_attribute(att_name, alias); // C++ signature
            return std::move(alias);
        })
        .def("get_attribute_alias", [](Tango::Database& self, const std::string& alias) -> std::string {
            std::string att_name;
            self.get_attribute_alias(alias, att_name); // C++ signature
            return std::move(att_name);
        })
        .def("put_attribute_alias", [](Tango::Database& self, std::string att_name, std::string& alias) -> void {
            self.put_attribute_alias(att_name, alias); // C++ signature
        })
        .def("delete_attribute_alias", [](Tango::Database& self, std::string& alias) -> void {
            self.delete_attribute_alias(alias); // C++ signature
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
