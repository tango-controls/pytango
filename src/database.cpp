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
#include <boost/python/return_value_policy.hpp>
#include <tango.h>
#include <string>

#include "defs.h"
#include "pytgutils.h"

using namespace boost::python;

extern const char *param_must_be_seq;
extern const char *unreachable_code;
extern const char *non_string_seq;

const char *param_numb_or_str_numb = "Second parameter must be an int or a "
                                     "string representing an int";

struct PyDatabase
{
    struct PickleSuite : pickle_suite
    {
        static tuple getinitargs(Tango::Database& self)
        {
            std::string& host = self.get_db_host();
            std::string& port = self.get_db_port();
            if (host.size() > 0 && port.size() > 0)
            {
                return make_tuple(host, port);
            }
            else
                return make_tuple();
        }
    };
    
    static inline boost::shared_ptr<Tango::Database>
    makeDatabase_host_port1(const std::string &host, int port)
    {
        return boost::shared_ptr<Tango::Database>
            (new Tango::Database(const_cast<std::string&>(host), port));
    }

    static inline boost::shared_ptr<Tango::Database>
    makeDatabase_host_port2(const std::string &host, const std::string &port_str)
    {
        std::istringstream port_stream(port_str);
        int port = 0;
        if(!(port_stream >> port))
        {
            raise_(PyExc_TypeError, param_numb_or_str_numb);
        }
        return boost::shared_ptr<Tango::Database>
            (new Tango::Database(const_cast<std::string&>(host), port));
    }

    static inline boost::shared_ptr<Tango::Database>
    makeDatabase_file(const std::string &filename)
    {
        return boost::shared_ptr<Tango::Database>
            (new Tango::Database(const_cast<std::string&>(filename)));
    }

    static inline boost::python::str
    get_device_alias(Tango::Database& self, const std::string &alias)
    {
        std::string devname;
        self.get_device_alias(alias, devname);
        return boost::python::str(devname);
    }

    static inline boost::python::str
    get_alias(Tango::Database& self, const std::string &devname)
    {
        std::string alias;
        self.get_alias(devname, alias);
        return boost::python::str(alias);
    }

    static inline void
    get_device_property_list2(Tango::Database& self, const std::string &devname,
                              const std::string &wildcard, StdStringVector &d)
    {
        self.get_device_property_list(const_cast<std::string&>(devname), wildcard, d);
    }

    static inline boost::python::str
    get_attribute_alias(Tango::Database& self, const std::string &alias)
    {
        std::string attrname;
        self.get_attribute_alias(alias, attrname);
        return boost::python::str(attrname);
    }

    static inline void
    export_event(Tango::Database& self, const boost::python::object &obj)
    {
        Tango::DevVarStringArray par;
        convert2array(obj, par);
        self.export_event(&par);
    }

    static inline boost::python::str dev_name(Tango::Database& self)
    {
        Tango::Connection *conn = static_cast<Tango::Connection *>(&self);
        return boost::python::str(conn->dev_name());
    }

    //static inline boost::python::str get_file_name(Tango::Database& self)
    //{
    //    return boost::python::str(self.get_file_name());
    //}
};

void export_database()
{
    // The following function declarations are necessary to be able to cast
    // the function parameters from string& to const string&, otherwise python
    // will not recognize the method calls

    Tango::DbDatum (Tango::Database::*get_host_list_)(std::string &) =
        &Tango::Database::get_host_list;
    Tango::DbDatum (Tango::Database::*get_services_)(std::string &, std::string &) =
        &Tango::Database::get_services;
    void (Tango::Database::*register_service_)(std::string &, std::string &, std::string &) =
        &Tango::Database::register_service;
    void (Tango::Database::*unregister_service_)(std::string &, std::string &) =
        &Tango::Database::unregister_service;
    Tango::DbDatum (Tango::Database::*get_device_name_)(std::string &, std::string &) =
        &Tango::Database::get_device_name;
    Tango::DbDatum (Tango::Database::*get_device_exported_)(std::string &) =
        &Tango::Database::get_device_exported;
    Tango::DbDatum (Tango::Database::*get_device_domain_)(std::string &) =
        &Tango::Database::get_device_domain;
    Tango::DbDatum (Tango::Database::*get_device_family_)(std::string &) =
        &Tango::Database::get_device_family;
    Tango::DbDatum (Tango::Database::*get_device_member_)(std::string &) =
        &Tango::Database::get_device_member;
    Tango::DbDatum (Tango::Database::*get_device_alias_list_)(std::string &) =
        &Tango::Database::get_device_alias_list;
    std::string (Tango::Database::*get_class_for_device_)(std::string &) =
        &Tango::Database::get_class_for_device;
    Tango::DbDatum (Tango::Database::*get_class_inheritance_for_device_)(std::string &) =
        &Tango::Database::get_class_inheritance_for_device;
    Tango::DbDatum (Tango::Database::*get_device_exported_for_class_)(std::string &) =
        &Tango::Database::get_device_exported_for_class;
    void (Tango::Database::*put_device_alias_)(std::string &, std::string &) =
        &Tango::Database::put_device_alias;
    void (Tango::Database::*delete_device_alias_)(std::string &) =
        &Tango::Database::delete_device_alias;
    void (Tango::Database::*add_server_)(std::string &, Tango::DbDevInfos &) =
        &Tango::Database::add_server;
    void (Tango::Database::*delete_server_)(std::string &) =
        &Tango::Database::delete_server;
    void (Tango::Database::*unexport_server_)(std::string &) =
        &Tango::Database::unexport_server;
    Tango::DbServerInfo (Tango::Database::*get_server_info_)(std::string &) =
        &Tango::Database::get_server_info;
    void (Tango::Database::*delete_server_info_)(std::string &) =
        &Tango::Database::delete_server_info;
    Tango::DbDatum (Tango::Database::*get_server_class_list_)(std::string &) =
        &Tango::Database::get_server_class_list;
    Tango::DbDatum (Tango::Database::*get_instance_name_list_)(std::string &) =
        &Tango::Database::get_instance_name_list;
    Tango::DbDatum (Tango::Database::*get_server_list_)(std::string &) =
        &Tango::Database::get_server_list;
    Tango::DbDatum (Tango::Database::*get_host_server_list_)(std::string &) =
        &Tango::Database::get_host_server_list;
    Tango::DbDatum (Tango::Database::*get_device_class_list_)(std::string &) =
        &Tango::Database::get_device_class_list;
    Tango::DbHistoryList (Tango::Database::*get_property_history_)(std::string &, std::string &) =
        &Tango::Database::get_property_history;
    Tango::DbDatum (Tango::Database::*get_object_list_)(std::string &) =
        &Tango::Database::get_object_list;
    Tango::DbDatum (Tango::Database::*get_object_property_list_)(std::string &, std::string &) =
        &Tango::Database::get_object_property_list;
    Tango::DbHistoryList (Tango::Database::*get_device_property_history_)(std::string &, std::string &) =
        &Tango::Database::get_device_property_history;
    Tango::DbDatum (Tango::Database::*get_device_property_list1_)(std::string &, std::string &) =
        &Tango::Database::get_device_property_list;
    Tango::DbHistoryList (Tango::Database::*get_device_attribute_property_history_)(std::string &, std::string &, std::string &) =
        &Tango::Database::get_device_attribute_property_history;
    Tango::DbHistoryList (Tango::Database::*get_class_property_history_)(std::string &, std::string &) =
        &Tango::Database::get_class_property_history;
    Tango::DbDatum (Tango::Database::*get_class_list_)(std::string &) =
        &Tango::Database::get_class_list;
    Tango::DbDatum (Tango::Database::*get_class_property_list_)(std::string &) =
        &Tango::Database::get_class_property_list;
    Tango::DbHistoryList (Tango::Database::*get_class_attribute_property_history_)(std::string &, std::string &, std::string &) =
        &Tango::Database::get_class_attribute_property_history;
    Tango::DbDatum (Tango::Database::*get_class_attribute_list_)(std::string &, std::string &) =
        &Tango::Database::get_class_attribute_list;

    Tango::DbDevImportInfo (Tango::Database::*import_device_)(std::string &) =
        &Tango::Database::import_device;
    
    Tango::DbDatum (Tango::Database::*get_attribute_alias_list_)(std::string &) =
        &Tango::Database::get_attribute_alias_list;
    void (Tango::Database::*put_attribute_alias_)(std::string &, std::string &) =
        &Tango::Database::put_attribute_alias;
    void (Tango::Database::*delete_attribute_alias_)(std::string &) =
        &Tango::Database::delete_attribute_alias;
    
    
    class_<Tango::Database, bases<Tango::Connection> > Database(
        "Database",
        init<>())
    ;

    Database
        .def("__init__", make_constructor(PyDatabase::makeDatabase_host_port1))
        .def("__init__", make_constructor(PyDatabase::makeDatabase_host_port2))
        .def("__init__", make_constructor(PyDatabase::makeDatabase_file))

        //
        // Pickle
        //
        .def_pickle(PyDatabase::PickleSuite())
        
        //
        // general methods
        //
        .def("dev_name", &PyDatabase::dev_name)
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
            return_internal_reference<1>())
        .def("is_multi_tango_host", &Tango::Database::is_multi_tango_host)
        
        //
        // General methods
        //

        .def("get_info",&Tango::Database::get_info)
        .def("get_host_list",
            (Tango::DbDatum (Tango::Database::*) ())
            &Tango::Database::get_host_list)
        .def("get_host_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_host_list_)
        .def("get_services",
            (Tango::DbDatum (Tango::Database::*) (const std::string &, const std::string &))
            get_services_)
        .def("register_service",
            (void (Tango::Database::*) (const std::string &, const std::string &, const std::string &))
            register_service_)
        .def("unregister_service",
            (void (Tango::Database::*) (const std::string &, const std::string &))
            unregister_service_)

        //
        // Device methods
        //

        .def("add_device", &Tango::Database::add_device)
        .def("delete_device", &Tango::Database::delete_device)
        .def("import_device", (Tango::DbDevImportInfo (Tango::Database::*) (const std::string &))
        import_device_)
        .def("export_device", &Tango::Database::export_device)
        .def("unexport_device", &Tango::Database::unexport_device)
        .def("get_device_name",
            (Tango::DbDatum (Tango::Database::*) (const string &, const string &))
            get_device_name_)
        .def("get_device_exported",
            (Tango::DbDatum (Tango::Database::*) (const string &))
            get_device_exported_)
        .def("get_device_domain",
            (Tango::DbDatum (Tango::Database::*) (const string &))
            get_device_domain_)
        .def("get_device_family",
            (Tango::DbDatum (Tango::Database::*) (const string &))
            get_device_family_)
        .def("get_device_member",
            (Tango::DbDatum (Tango::Database::*) (const string &))
            get_device_member_)
        .def("get_device_alias", &PyDatabase::get_device_alias)
        .def("get_alias", &PyDatabase::get_alias)
        .def("get_device_alias_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_device_alias_list_)
        .def("get_class_for_device",
            (std::string (Tango::Database::*) (const std::string &))
            get_class_for_device_)
        .def("get_class_inheritance_for_device",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_class_inheritance_for_device_)
        .def("get_device_exported_for_class",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_device_exported_for_class_)
        .def("put_device_alias",
            (void (Tango::Database::*) (const std::string &, const std::string &))
            put_device_alias_)
        .def("delete_device_alias",
            (void (Tango::Database::*) (const std::string &))
            delete_device_alias_)

        //
        // server methods
        //

        .def("_add_server",
            (void (Tango::Database::*) (const std::string &, Tango::DbDevInfos &))
            add_server_)
        .def("delete_server",
            (void (Tango::Database::*) (const std::string &))
            delete_server_)
        .def("_export_server", &Tango::Database::export_server)
        .def("unexport_server",
            (void (Tango::Database::*) (const std::string &))
            unexport_server_)
        .def("get_server_info",
            (Tango::DbServerInfo (Tango::Database::*) (const std::string &))
            get_server_info_)
        .def("put_server_info", &Tango::Database::put_server_info,
            ( arg_("self"), arg_("info") ))
        .def("delete_server_info",
            (void (Tango::Database::*) (const std::string &))
            delete_server_info_)
        .def("get_server_class_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_server_class_list_)
        .def("get_server_name_list", &Tango::Database::get_server_name_list)
        .def("get_instance_name_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_instance_name_list_)
        .def("get_server_list",
            (Tango::DbDatum (Tango::Database::*) ())
            &Tango::Database::get_server_list)
        .def("get_server_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_server_list_)
        .def("get_host_server_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_host_server_list_)
        .def("get_device_class_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_device_class_list_)

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
            (Tango::DbHistoryList (Tango::Database::*) (const std::string &, const std::string &))
            get_property_history_)
        .def("get_object_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_object_list_)
        .def("get_object_property_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &, const std::string &))
            get_object_property_list_)
        .def("_get_device_property",
            (void (Tango::Database::*) (std::string, Tango::DbData &))
            &Tango::Database::get_device_property)
        .def("_put_device_property", &Tango::Database::put_device_property)
        .def("_delete_device_property", &Tango::Database::delete_device_property)
        .def("get_device_property_history",
            (Tango::DbHistoryList (Tango::Database::*) (const std::string &, const std::string &))
            get_device_property_history_)
        .def("_get_device_property_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &, const std::string &))
            get_device_property_list1_)
        .def("_get_device_property_list", &PyDatabase::get_device_property_list2)
        .def("_get_device_attribute_property",
            (void (Tango::Database::*) (std::string, Tango::DbData &))
            &Tango::Database::get_device_attribute_property)
        .def("_put_device_attribute_property",
            &Tango::Database::put_device_attribute_property)
        .def("_delete_device_attribute_property",
            &Tango::Database::delete_device_attribute_property)
        .def("get_device_attribute_property_history",
            (Tango::DbHistoryList (Tango::Database::*) (const std::string &, const std::string &, const std::string &))
            get_device_attribute_property_history_)
        .def("_get_class_property",
            (void (Tango::Database::*) (std::string, Tango::DbData &))
            &Tango::Database::get_class_property)
        .def("_put_class_property", &Tango::Database::put_class_property)
        .def("_delete_class_property", &Tango::Database::delete_class_property)
        .def("get_class_property_history",
            (Tango::DbHistoryList (Tango::Database::*) (const std::string &, const std::string &))
            get_class_property_history_)
        .def("get_class_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_class_list_)
        .def("get_class_property_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_class_property_list_)
        .def("_get_class_attribute_property",
            (void (Tango::Database::*) (std::string, Tango::DbData &))
            &Tango::Database::get_class_attribute_property)
        .def("_put_class_attribute_property",
            &Tango::Database::put_class_attribute_property)
        .def("_delete_class_attribute_property",
            &Tango::Database::delete_class_attribute_property)
        .def("get_class_attribute_property_history",
            (Tango::DbHistoryList (Tango::Database::*) (const std::string &, const std::string &, const std::string &))
            get_class_attribute_property_history_)

        .def("get_class_attribute_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &, const std::string &))
            get_class_attribute_list_)

        //
        // Attribute methods
        //

        .def("get_attribute_alias", &PyDatabase::get_attribute_alias)
        .def("get_attribute_alias_list",
            (Tango::DbDatum (Tango::Database::*) (const std::string &))
            get_attribute_alias_list_)
        .def("put_attribute_alias",
            (void (Tango::Database::*) (const std::string &, const std::string &))
            put_attribute_alias_)
        .def("delete_attribute_alias",
            (void (Tango::Database::*) (const std::string &))
            delete_attribute_alias_)

        //
        // event methods
        //

        .def("export_event", &PyDatabase::export_event)
        .def("unexport_event",
            (void (Tango::Database::*) (const std::string &))
            &Tango::Database::unexport_event)
        ;
}

