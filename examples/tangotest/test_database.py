# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2019 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

import socket
import pytest
from tango import Database
from tango import DbDatum
from tango import Release
from tango import AccessControlType

Hostname = socket.getfqdn()
Port = 10000
db = Database(Hostname, Port)
assert db.get_db_host() == Hostname
assert db.get_db_port() == str(Port)
assert db.get_db_port_num() == Port
assert db.get_from_env_var() == False
assert db.get_fqdn() == ""
assert db.is_dbase_used() == True
assert db.get_dev_host() == Hostname
assert db.get_dev_port() == str(Port)
assert db.dev_name() == "sys/database/2"
#db.write_filedatabase() -> void {
#db.reread_filedatabase() -> void {
#db.build_connection() -> void {
#db.check_tango_host(, std::string& tango_host_env) -> void {
assert db.check_access_control("sys/database/2") == AccessControlType.ACCESS_WRITE
assert db.is_control_access_checked() == False
#db.set_access_checked(, bool checked) -> void {
#db.get_access_except_errors() -> Tango::DevErrorList& {
assert db.is_multi_tango_host() == False
assert db.get_multi_host() == []
assert db.get_multi_port() == []
with pytest.raises(Exception):
    db.get_file_name()
#print(db.get_info())
db_datum= db.get_host_list()
assert db_datum.name == "host"
assert Hostname in db_datum.value_string
assert Hostname in db.get_host_list("*")
#db.get_services(, std::string& service_name, std::string& inst_name) -> Tango::DbDatum {
#db.get_device_service_list(, std::string& service_name) -> Tango::DbDatum {
#db.register_service(, std::string& service_name,
#db.unregister_service(, std::string& service_name, std::string& inst_name) -> void {
#db.add_device(, Tango::DbDevInfo& dev_info) -> void {
#db.delete_device(, std::string dev_name) -> void {
#db.import_device(, std::string& dev_name) -> Tango::DbDevImportInfo {
#db.export_device(, Tango::DbDevExportInfo &info) -> void {
#db.unexport_device(, std::string& dev_name) -> void {
info = db.get_device_info("sys/database/2")
assert info.class_name == "DataBase"
assert info.ds_full_name == "DataBaseds/2"
assert info.exported == True
assert info.ior[:4] == "IOR:"
assert info.name == "sys/database/2"
assert info.version == "5"
assert db.get_class_for_device("sys/database/2") == "DataBase"
db_datum = db.get_class_inheritance_for_device("sys/database/2")
assert db_datum.name == "class"
assert db_datum.value_string == ["DataBase"]
db_datum = db.get_class_inheritance_for_device("sys/tg_test/1")
assert db_datum.name == "class"
assert db_datum.value_string == ["TangoTest", "TANGO_BASE_CLASS"]
db_datum = db.get_device_name("DataBaseds","DataBase")
assert db_datum.name == "DataBaseds"
assert db_datum.value_string == []
db_datum = db.get_device_exported("sys/*")
assert db_datum.name == "sys/*"
assert db_datum.value_string == ["sys/database/2", "sys/tg_test/1"]
db_datum = db.get_device_domain("sy*")
assert db_datum.name == "sy*"
assert db_datum.value_string == ["sys"]
db_datum = db.get_device_family("*")
assert db_datum.name == "*"
assert "DataBaseds" in db_datum.value_string
db_datum = db.get_device_member("*")
assert db_datum.name == "*"
assert "test" in db_datum.value_string
db_datum = db.get_device_class_list("sys/tg_test/1")
assert db_datum.name == "server"
db_datum = db.get_device_exported_for_class("TangoTest")
assert db_datum.name == "device"
assert db_datum.value_string == ['sys/tg_test/1']
db_datum = db.get_object_list("*")
assert db_datum.name == "object"
print(db_datum.value_string)
assert db_datum.value_string == ['Astor', 'CtrlSystem']
db_datum = db.get_object_property_list("Astor", "R*")
assert db_datum.name == "object"
assert db_datum.value_string == ['RloginCmd', 'RloginUser']
db_datum = db.get_class_list("*")
assert db_datum.name == "class"
assert "DataBase" and "TangoTest" in db_datum.value_string
db_datum = db.get_class_property_list("DataBase")
assert db_datum.name == "class"
assert db_datum.value_string == ['AllowedAccessCmd']
db_datum = db.get_class_attribute_list("DataBase", "*")
assert db_datum.name == "class"
assert db_datum.value_string == []
db_datum = db.get_class_pipe_list("DataBase", "*")
assert db_datum.name == "class"
assert db_datum.value_string == []
db_datum = db.get_device_alias_list("D*")
assert db_datum.name == "D*"
assert db_datum.value_string == []
db_datum = db.get_attribute_alias_list("*")
assert db_datum.name == "*"
assert db_datum.value_string == []
#db._add_server(, std::string& ds_name, Tango::DbDevInfos& devs) -> void {
#db.delete_server(, std::string& ds_name) -> void {
#db._export_server(, Tango::DbDevExportInfos &devs) -> void {
#db.unexport_server(, std::string& ds_name) -> void {
#db.rename_server(, std::string& old_ds_name, std::string& new_ds_name) -> void {
#db.get_server_info(, std::string& name) -> Tango::DbServerInfo {
#db.put_server_info(, Tango::DbServerInfo& info) -> void {
#db.delete_server_info(, std::string& info_name) -> void {
db_datum = db.get_server_class_list("DataBaseds")
assert db_datum.name == "DataBaseds"
assert db_datum.value_string == []
db_datum = db.get_server_name_list()
assert db_datum.name == "server"
assert "DataBaseds" in db_datum.value_string
db_datum = db.get_instance_name_list("DataBaseds")
assert db_datum.name == "DataBaseds"
assert db_datum.value_string == ["2"]
db_datum = db.get_server_list()
assert db_datum.name == "server"
assert "DataBaseds/2" in db_datum.value_string
db_datum = db.get_server_list("Da*")
assert db_datum.name == "server"
assert db_datum.value_string == ["DataBaseds/2"]
db_datum = db.get_host_server_list("tcfidell11")
assert db_datum.name == "server"
assert "DataBaseds/2" in db_datum.value_string
assert db.get_server_release() == 400


db_datum = DbDatum("property-1")
db_datum.value_string = ["abcdef"]
db.put_property("DataBaseTest", db_datum)
db_dict = db.get_property("DataBaseTest", "property-1")
assert db_dict == {u'property-1': [u'abcdef']}
db.delete_property("DataBaseTest", DbDatum("property-1"))
db_dict = db.get_property("DataBaseTest", DbDatum("property-1"))
assert db_dict == {"property-1": []}
db_datum1 = DbDatum("property-2")
db_datum1.value_string = ["abcdef"]
db_datum2 = DbDatum("property-3")
db_datum2.value_string = ["ghijkl"]
db_data = [db_datum1, db_datum2]
db.put_property("DataBaseTest", db_data)
db_dict = db.get_property("DataBaseTest", [DbDatum("property-2"), DbDatum("property-3")])
assert db_dict == {u'property-2': [u'abcdef'], u'property-3': [u'ghijkl']}
db_dict = db.get_property_forced("DataBaseTest", db_data) 
assert db_dict == {u'property-2': [u'abcdef'], u'property-3': [u'ghijkl']}
db.delete_property("DataBaseTest", db_data)
db_dict = db.get_property("DataBaseTest", [DbDatum("property-2"), DbDatum("property-3")])
assert db_dict == {u'property-2': [], u'property-3': []}
db_datum = DbDatum("property-8")
db_datum.value_string = ["ABCDEF"]
db.put_property("DataBaseTest", {"property-8": db_datum})
db_dict = db.get_property("DataBaseTest", DbDatum("property-8"))
assert db_dict == {u'property-8': ["ABCDEF"]}
db.delete_property("DataBaseTest", DbDatum("property-8"))
db_dict = db.get_property("DataBaseTest", DbDatum("property-8"))
assert db_dict == {u'property-8': []}
db.put_property("DataBaseTest", {"property-4": ["qrstu","vwxyz"]})
db_dict = db.get_property("DataBaseTest", "property-4")
assert db_dict == {u'property-4': ["qrstu","vwxyz"]}
db.delete_property("DataBaseTest", "property-4")
db_dict = db.get_property("DataBaseTest", DbDatum("property-4"))
assert db_dict == {u'property-4': []}
db.put_property("DataBaseTest", {"property-5": 3.142})
db_dict = db.get_property("DataBaseTest", "property-5")
assert db_dict == {u'property-5': ["3.142"]}
db.delete_property("DataBaseTest", "property-5")
db_dict = db.get_property("DataBaseTest", DbDatum("property-5"))
assert db_dict == {u'property-5': []}

db_hist_data = db.get_property_history("DataBaseTest", "property-5")
assert len(db_hist_data) == 10
db_datum1 = DbDatum("property-2")
db_datum1.value_string = ["abcdef"]
db_datum2 = DbDatum("property-3")
db_datum2.value_string = ["ghijkl"]
db_data = [db_datum1, db_datum2]
db.put_device_property("sys/tg_test/1", db_data)
db_data = db.get_device_property("sys/tg_test/1", [DbDatum("property-2"), DbDatum("property-3")])
assert db_data == {u'property-2': [u'abcdef'], u'property-3': [u'ghijkl']}
#db._delete_device_property(, string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
#db_hist_data = db.get_device_property_history(dev_name, prop_name)
#db_datum = db._get_device_property_list(dev_name, "*")
#db._get_device_property_list(, std::string& dev_name,
#db_data = db._get_device_attribute_property(dev_name, dbData)
#db._put_device_attribute_property(, string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
#db._delete_device_attribute_property(, string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
#db_hist_data = db.get_device_attribute_property_history(dev_name, prop_name, att_name)
#db.get_device_attribute_list(, std::string& dev_name, std::vector<std::string>& att_list) -> std::vector<std::string> {
#db_data = db._get_class_property(dev_class, dbData)
#db._put_class_property(, std::string& dev_class, std::vector<Tango::DbDatum>& dbData) -> void {
#db._delete_class_property(, std::string& dev_class, std::vector<Tango::DbDatum>& dbData) -> void {
#db_hist_data = db.get_class_property_history(class_name, prop_name)
#db_data = db._get_class_attribute_property(class_name, dbData)
#db._put_class_attribute_property(, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> void {
#db._delete_class_attribute_property(, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> void {
#db.get_class_attribute_property_history(, std::string& class_name, std::string& att_name,
#db.export_event(, std::vector<std::string>& event_data) {
#db.unexport_event",[](Tango::Database& self, std::string &ev) -> void {
#db.get_device_pipe_list(, std::string& dev_name, std::vector<std::string>& pipe_list) -> std::vector<std::string> {
#db.get_device_pipe_property(, std::string dev_name, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
#db.put_device_pipe_property(, std::string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
#db.delete_device_pipe_property(, std::string dev_name, std::vector<Tango::DbDatum>& dbData) -> void {
#db.get_device_pipe_property_history(, std::string& dev_name,
#db.get_class_pipe_property(, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> std::vector<Tango::DbDatum> {
#db.put_class_pipe_property(, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> void {
#db.delete_class_pipe_property(, std::string class_name, std::vector<Tango::DbDatum>& dbData) -> void {
#db.get_class_pipe_property_history(, std::string& class_name,
#db.get_device_from_alias(, std::string& alias) -> std::string {
#db.get_alias_from_device(, std::string& dev_name) -> std::string {
#db.get_alias(, std::string& devname) -> std::string {
#db.get_device_alias(, std::string& alias) -> std::string {
#db.put_device_alias(, std::string& dev_name, std::string& dev_alias) -> void {
#db.delete_device_alias(, std::string& dev_alias) -> void {
#db.get_attribute_from_alias(, std::string& alias) -> std::string {
#db.get_alias_from_attribute(, std::string& att_name) -> std::string {
#db.get_attribute_alias(, const std::string& alias) -> std::string {
#db.put_attribute_alias(, std::string att_name, std::string& alias) -> void {
#db.delete_attribute_alias(, std::string& alias) -> void {

print("passed database tests")
