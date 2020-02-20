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
#from lib2to3.pgen2.tokenize import blank_re

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
#db.write_filedatabase()
#db.reread_filedatabase()
#db.build_connection()
#db.check_tango_host(std::string& tango_host_env)
assert db.check_access_control("sys/database/2") == AccessControlType.ACCESS_WRITE
assert db.is_control_access_checked() == False
db.set_access_checked(True)
assert db.is_control_access_checked() == True
db.set_access_checked(False)
#db.get_access_except_errors()
assert db.is_multi_tango_host() == False
assert db.get_multi_host() == []
assert db.get_multi_port() == []
with pytest.raises(Exception):
    db.get_file_name()
info = db.get_info()
assert info[:29] == "TANGO Database sys/database/2"
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
assert 'Astor' and 'CtrlSystem' in db_datum.value_string
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
# db._add_server(, std::string& ds_name, Tango::DbDevInfos& devs) -> void {
# db.delete_server(, std::string& ds_name) -> void {
# db._export_server(, Tango::DbDevExportInfos &devs) -> void {
# db.unexport_server(, std::string& ds_name) -> void {
# db.rename_server(, std::string& old_ds_name, std::string& new_ds_name) -> void {
db_info = db.get_server_info("sys/tg_test/1")
assert db_info.name == "sys/tg_test/1"
assert db_info.level == 0
assert db_info.mode == 0
assert db_info.host == " "
db_info.level = 1
db.put_server_info(db_info)
db_info = db.get_server_info("sys/tg_test/1")
assert db_info.level == 1
db_info.level = 0
db.put_server_info(db_info)
# db.delete_server_info(, std::string& info_name) -> void {
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
#
# test properties
#
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
#
# test device properties
#
db_datum1 = DbDatum("property-2")
db_datum1.value_string = ["abcdef"]
db_datum2 = DbDatum("property-3")
db_datum2.value_string = ["ghijkl"]
db_data = [db_datum1, db_datum2]
db.put_device_property("sys/tg_test/1", db_data)
db_data = db.get_device_property("sys/tg_test/1", [DbDatum("property-2"), DbDatum("property-3")])
assert db_data == {u'property-2': [u'abcdef'], u'property-3': [u'ghijkl']}
db._delete_device_property("sys/tg_test/1", [DbDatum("property-2"),DbDatum("property-3")])
db_data = db.get_device_property("sys/tg_test/1", [DbDatum("property-2"), DbDatum("property-3")])
assert db_data == {u'property-2': [], u'property-3': []}
db_hist_data = db.get_device_property_history("sys/tg_test/1", "sleep_period")
print(db_hist_data)
assert len(db_hist_data) >= 1
db_datum = db.get_device_property_list("sys/tg_test/1", "*")
assert db_datum.name == 'sys/tg_test/1'
assert 'sleep_period' in db_datum.value_string
db_property_data = []
db_data_list = db.get_device_property_list("sys/tg_test/1","*",db_property_data)
assert 'sleep_period' in db_data_list
#
# test device attribute properties
#
db_datum = DbDatum("float_scalar")
db_datum.value_string = ["2"]
db_datum_min = DbDatum("min")
db_datum_min.value_string = ["0.0"]
db_datum_max = DbDatum("max");
db_datum_max.value_string = ["4096.0"]
db_data = [db_datum, db_datum_min, db_datum_max]
db.put_device_attribute_property("sys/tg_test/1", db_data)
db.put_device_attribute_property("sys/tg_test/1", {"long_scalar": {"min": ["0"], "max": ["2048"]}})
db_dict = db.get_device_attribute_property("sys/tg_test/1", "float_scalar")
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}}
db_dict = db.get_device_attribute_property("sys/tg_test/1", ["float_scalar", "long_scalar"])
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_device_attribute_property("sys/tg_test/1", DbDatum("float_scalar"))
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}}
db_dict = db.get_device_attribute_property("sys/tg_test/1", [DbDatum("float_scalar"), DbDatum("long_scalar")])
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_device_attribute_property("sys/tg_test/1", {"float_scalar": "", "long_scalar": ""})
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_device_attribute_property("sys/tg_test/1", {"long_scalar": DbDatum("long_scalar")})
assert db_dict == {'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_data_list = db.get_device_attribute_list("sys/tg_test/1", [])
assert "long_scalar" and "float_scalar" in db_data_list
db.delete_device_attribute_property("sys/tg_test/1", [DbDatum("float_scalar"), DbDatum("max"), DbDatum("min")]) 
db.delete_device_attribute_property("sys/tg_test/1", {"long_scalar": ["max", "min"]}) 
db_dict = db.get_device_attribute_property("sys/tg_test/1", [DbDatum("float_scalar"), DbDatum("long_scalar")])
assert db_dict == {'float_scalar': {}, 'long_scalar': {}}
db_hist_data = db.get_device_attribute_property_history("sys/tg_test/1", "float_scalar", "max")
assert len(db_hist_data) > 1
#
# test class properties
#
db_datum = DbDatum("property-1")
db_datum.value_string = ["abcdef"]
db.put_class_property("TangoTest", db_datum)
db_dict = db.get_class_property("TangoTest", "property-1")
assert db_dict == {u'property-1': [u'abcdef']}
db.delete_class_property("TangoTest", DbDatum("property-1"))
db_dict = db.get_class_property("TangoTest", DbDatum("property-1"))
assert db_dict == {"property-1": []}
db_datum1 = DbDatum("property-2")
db_datum1.value_string = ["abcdef"]
db_datum2 = DbDatum("property-3")
db_datum2.value_string = ["ghijkl"]
db_data = [db_datum1, db_datum2]
db.put_class_property("TangoTest", db_data)
db_dict = db.get_class_property("TangoTest", [DbDatum("property-2"), DbDatum("property-3")])
assert db_dict == {u'property-2': [u'abcdef'], u'property-3': [u'ghijkl']}
db.delete_class_property("TangoTest", db_data)
db_dict = db.get_class_property("TangoTest", [DbDatum("property-2"), DbDatum("property-3")])
assert db_dict == {u'property-2': [], u'property-3': []}
db_datum = DbDatum("property-8")
db_datum.value_string = ["ABCDEF"]
db.put_class_property("TangoTest", {"property-8": db_datum})
db_dict = db.get_class_property("TangoTest", DbDatum("property-8"))
assert db_dict == {u'property-8': ["ABCDEF"]}
db.delete_class_property("TangoTest", DbDatum("property-8"))
db_dict = db.get_class_property("TangoTest", DbDatum("property-8"))
assert db_dict == {u'property-8': []}
db.put_class_property("TangoTest", {"property-4": ["qrstu","vwxyz"]})
db_dict = db.get_class_property("TangoTest", "property-4")
assert db_dict == {u'property-4': ["qrstu","vwxyz"]}
db.delete_class_property("TangoTest", "property-4")
db_dict = db.get_class_property("TangoTest", DbDatum("property-4"))
assert db_dict == {u'property-4': []}
db.put_class_property("TangoTest", {"property-5": 3.142})
db_dict = db.get_class_property("TangoTest", "property-5")
assert db_dict == {u'property-5': ["3.142"]}
db.delete_class_property("TangoTest", "property-5")
db_dict = db.get_class_property("TangoTest", DbDatum("property-5"))
assert db_dict == {u'property-5': []}
db_hist_data = db.get_class_property_history("TangoTest", "property-1")
assert len(db_hist_data) > 1
#
# test class attribute properties
#
db_datum = DbDatum("float_scalar")
db_datum.value_string = ["2"]
db_datum_min = DbDatum("min")
db_datum_min.value_string = ["0.0"]
db_datum_max = DbDatum("max");
db_datum_max.value_string = ["4096.0"]
db_data = [db_datum, db_datum_min, db_datum_max]
db.put_class_attribute_property("TangoTest", db_data)
db.put_class_attribute_property("TangoTest", {"long_scalar": {"min": ["0"], "max": ["2048"]}})
db_dict = db.get_class_attribute_property("TangoTest", "float_scalar")
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}}
db_dict = db.get_class_attribute_property("TangoTest", ["float_scalar", "long_scalar"])
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_class_attribute_property("TangoTest", DbDatum("float_scalar"))
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}}
db_dict = db.get_class_attribute_property("TangoTest", [DbDatum("float_scalar"), DbDatum("long_scalar")])
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_class_attribute_property("TangoTest", {"float_scalar": "", "long_scalar": ""})
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_class_attribute_property("TangoTest", {"long_scalar": DbDatum("long_scalar")})
assert db_dict == {'long_scalar': {'max': ['2048'], 'min': ['0']}}
db.delete_class_attribute_property("TangoTest", [DbDatum("float_scalar"), DbDatum("max"), DbDatum("min")]) 
db.delete_class_attribute_property("TangoTest", {"long_scalar": ["max", "min"]}) 
db_dict = db.get_class_attribute_property("TangoTest", [DbDatum("float_scalar"), DbDatum("long_scalar")])
assert db_dict == {'float_scalar': {}, 'long_scalar': {}}
db_hist_data = db.get_class_attribute_property_history("TangoTest", "long_scalar", "max")
assert len(db_hist_data) > 1

#db.export_event(, std::vector<std::string>& event_data) {
#db.unexport_event",[](Tango::Database& self, std::string &ev) -> void {
#
# test class attribute properties
#
db_datum = DbDatum("float_scalar")
db_datum.value_string = ["2"]
db_datum_min = DbDatum("min")
db_datum_min.value_string = ["0.0"]
db_datum_max = DbDatum("max");
db_datum_max.value_string = ["4096.0"]
db_data = [db_datum, db_datum_min, db_datum_max]
db.put_device_pipe_property("sys/tg_test/1", db_data)
db.put_device_pipe_property("sys/tg_test/1", {"long_scalar": {"min": ["0"], "max": ["2048"]}})
db_dict = db.get_device_pipe_property("sys/tg_test/1", "float_scalar")
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}}
db_dict = db.get_device_pipe_property("sys/tg_test/1", ["float_scalar", "long_scalar"])
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_device_pipe_property("sys/tg_test/1", DbDatum("float_scalar"))
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}}
db_dict = db.get_device_pipe_property("sys/tg_test/1", [DbDatum("float_scalar"), DbDatum("long_scalar")])
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_device_pipe_property("sys/tg_test/1", {"float_scalar": "", "long_scalar": ""})
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_device_pipe_property("sys/tg_test/1", {"long_scalar": DbDatum("long_scalar")})
assert db_dict == {'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_pipe_list = db.get_device_pipe_list("sys/tg_test/1", [])
assert db_pipe_list == ['float_scalar', 'long_scalar']
db.delete_device_pipe_property("sys/tg_test/1", [DbDatum("float_scalar"), DbDatum("max"), DbDatum("min")]) 
db.delete_device_pipe_property("sys/tg_test/1", {"long_scalar": ["max", "min"]}) 
db_dict = db.get_device_pipe_property("sys/tg_test/1", [DbDatum("float_scalar"), DbDatum("long_scalar")])
assert db_dict == {'float_scalar': {}, 'long_scalar': {}}
db_hist_data = db.get_device_pipe_property_history("sys/tg_test/1", "float_scalar", "max")
assert len(db_hist_data) > 1
#
# test pipe class properties
#
db_datum = DbDatum("float_scalar")
db_datum.value_string = ["2"]
db_datum_min = DbDatum("min")
db_datum_min.value_string = ["0.0"]
db_datum_max = DbDatum("max");
db_datum_max.value_string = ["4096.0"]
db_data = [db_datum, db_datum_min, db_datum_max]
db.put_class_pipe_property("TangoTest", db_data)
db.put_class_pipe_property("TangoTest", {"long_scalar": {"min": ["0"], "max": ["2048"]}})
db_dict = db.get_class_pipe_property("TangoTest", "float_scalar")
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}}
db_dict = db.get_class_pipe_property("TangoTest", ["float_scalar", "long_scalar"])
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_class_pipe_property("TangoTest", DbDatum("float_scalar"))
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}}
db_dict = db.get_class_pipe_property("TangoTest", [DbDatum("float_scalar"), DbDatum("long_scalar")])
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_class_pipe_property("TangoTest", {"float_scalar": "", "long_scalar": ""})
assert db_dict == {'float_scalar': {'max': ['4096.0'], 'min': ['0.0']}, 'long_scalar': {'max': ['2048'], 'min': ['0']}}
db_dict = db.get_class_pipe_property("TangoTest", {"long_scalar": DbDatum("long_scalar")})
assert db_dict == {'long_scalar': {'max': ['2048'], 'min': ['0']}}
db.delete_class_pipe_property("TangoTest", [DbDatum("float_scalar"), DbDatum("max"), DbDatum("min")]) 
db.delete_class_pipe_property("TangoTest", {"long_scalar": ["max", "min"]}) 
db_dict = db.get_class_pipe_property("TangoTest", [DbDatum("float_scalar"), DbDatum("long_scalar")])
assert db_dict == {'float_scalar': {}, 'long_scalar': {}}
db_hist_data = db.get_class_pipe_property_history("TangoTest", "long_scalar", "max")
assert len(db_hist_data) > 1
#
# test device alias
#
db.put_device_alias("sys/tg_test/1", "tg_test_alias")
assert db.get_device_from_alias("tg_test_alias") == "sys/tg_test/1"
assert db.get_alias_from_device("sys/tg_test/1") == "tg_test_alias"
assert db.get_alias("sys/tg_test/1") == "tg_test_alias"
assert db.get_device_alias("tg_test_alias") == "sys/tg_test/1"
db.delete_device_alias("tg_test_alias")
with pytest.raises(Exception):
    db.get_device_from_alias("tg_test_alias")
with pytest.raises(Exception):
    db.get_alias_from_device("sys/tg_test/1")
#
# test attribute alias
#
db.put_attribute_alias("sys/tg_test/1/long_scalar", "attr_test_alias")
assert db.get_attribute_from_alias("attr_test_alias") == "sys/tg_test/1/long_scalar"
assert db.get_alias_from_attribute("sys/tg_test/1/long_scalar") == "attr_test_alias"
db.get_attribute_alias("attr_test_alias") == "sys/tg_test/1/long_scalar"
db.delete_attribute_alias("attr_test_alias")
with pytest.raises(Exception):
    db.get_attribute_from_alias("attr_test_alias")
with pytest.raises(Exception):
    db.get_alias_from_attribute("sys/tg_test/1/long_scalar")

print("passed database tests")
