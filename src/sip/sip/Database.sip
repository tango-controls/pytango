/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

class Database : Connection {

%TypeHeaderCode
#include <tango.h>
using namespace Tango;
%End

private:
	virtual std::string get_corba_name(bool);
	virtual std::string build_corba_name();
	virtual int get_lock_ctr();
	virtual void set_lock_ctr(int);
    virtual std::string dev_name();

public :
	Database();
	Database(std::string &host, int port);
	Database(std::string &file);

	Database(const Database &);
////	Database & operator=(const Database &);

	void write_filedatabase();
	void reread_filedatabase();
	void write_event_channel_ior_filedatabase(std::string &);
	void build_connection ();
	void post_reconnection();


////	inline Device_var &get_dbase() { return device;}
	void check_tango_host(const char *);
	AccessControlType check_access_control(std::string &);
	bool is_control_access_checked();
	void set_access_checked(bool val);

//	void set_tango_utils(Tango::Util *ptr);
	int get_server_release();

//	DevErrorList &get_access_except_errors();
	void clear_access_except_errors();
	bool is_command_allowed(std::string &,std::string &);

	bool is_multi_tango_host();
	std::vector<std::string> &get_multi_host();
	std::vector<std::string> &get_multi_port();

	const std::string &get_file_name();

//#ifdef _TG_WINDOWS_
//	Database(CORBA::ORB *orb,std::string &,std::string &);
//	long get_tango_host_from_reg(char **,std::string &,std::string &);
//#endif

//
// general methods
//

	std::string get_info();

	DbDatum get_host_list();
	DbDatum get_host_list(std::string &);
	DbDatum get_services(std::string &,std::string &);
	void register_service(std::string &,std::string &,std::string &);
	void unregister_service(std::string &,std::string &);
//	CORBA::Any *fill_server_cache(std::string &,std::string &);

/*
//
// device methods
//

	void add_device(DbDevInfo&);
	void delete_device(std::string);
	DbDevImportInfo import_device(std::string &);
	void export_device(DbDevExportInfo &);
	void unexport_device(std::string);

	DbDatum get_device_name(std::string &, std::string &,DbServerCache *dsc);
	DbDatum get_device_name(std::string &, std::string &);
	DbDatum get_device_exported(std::string &);
	DbDatum get_device_domain(std::string &);
	DbDatum get_device_family(std::string &);
	DbDatum get_device_member(std::string &);
	void get_device_alias(std::string,std::string &);
	void get_alias(std::string,std::string &);
	DbDatum get_device_alias_list(std::string &);
	std::string get_class_for_device(std::string &);
	DbDatum get_class_inheritance_for_device(std::string &);
	DbDatum get_device_exported_for_class(std::string &);
	void put_device_alias(std::string &,std::string &);
	void delete_device_alias(std::string &);

//
// server methods
//
	void add_server(std::string &, DbDevInfos&);
	void delete_server(std::string &);
	void export_server(DbDevExportInfos &);
	void unexport_server(std::string &);

	DbServerInfo get_server_info(std::string &);
	void put_server_info(DbServerInfo &);
	void delete_server_info(std::string &);
	DbDatum get_server_class_list(std::string &);
	DbDatum get_server_name_list();
	DbDatum get_instance_name_list(std::string &);
	DbDatum get_server_list();
	DbDatum get_server_list(std::string &);
	DbDatum get_host_server_list(std::string &);
	DbDatum get_device_class_list(std::string &);

//
// property methods
//

	void get_property(std::string, DbData &,DbServerCache *dsc);
	void get_property(std::string st, DbData &db) {get_property(st,db,NULL);}
	void get_property_forced(std::string, DbData &,DbServerCache *dsc = NULL);
	void put_property(std::string, DbData &);
	void delete_property(std::string, DbData &);
	vector<DbHistory> get_property_history(std::string &,std::string &);
	DbDatum get_object_list(std::string &);
	DbDatum get_object_property_list(std::string &,std::string &);

	void get_device_property(std::string, DbData &, DbServerCache *dsc);
	void get_device_property(std::string st, DbData &db) {get_device_property(st,db,NULL);}
	void put_device_property(std::string, DbData &);
	void delete_device_property(std::string, DbData &);
	vector<DbHistory> get_device_property_history(std::string &,std::string &);
	DbDatum get_device_property_list(std::string &,std::string &);
	void get_device_property_list(std::string &,const std::string &,vector<std::string> &,DbServerCache *dsc = NULL);

	void get_device_attribute_property(std::string, DbData &, DbServerCache *dsc);
	void get_device_attribute_property(std::string st, DbData &db) {get_device_attribute_property(st,db,NULL);}
	void put_device_attribute_property(std::string, DbData &);
	void delete_device_attribute_property(std::string, DbData &);
	void delete_all_device_attribute_property(std::string, DbData &);
	vector<DbHistory> get_device_attribute_property_history(std::string &,std::string &,std::string &);

	void get_class_property(std::string, DbData &, DbServerCache *dsc);
	void get_class_property(std::string st,DbData &db) {get_class_property(st,db,NULL);}
	void put_class_property(std::string, DbData &);
	void delete_class_property(std::string, DbData &);
	vector<DbHistory> get_class_property_history(std::string &,std::string &);
	DbDatum get_class_list(std::string &);
	DbDatum get_class_property_list(std::string &);

	void get_class_attribute_property(std::string, DbData &, DbServerCache *dsc);
	void get_class_attribute_property(std::string st,DbData &db) {get_class_attribute_property(st,db,NULL);}
	void put_class_attribute_property(std::string, DbData &);
	void delete_class_attribute_property(std::string, DbData &);
	vector<DbHistory> get_class_attribute_property_history(std::string &,std::string &,std::string &);
	DbDatum get_class_attribute_list(std::string &,std::string &);


// attribute methods

	void get_attribute_alias(std::string, std::string&);
	DbDatum get_attribute_alias_list(std::string &);
	void put_attribute_alias(std::string &,std::string &);
	void delete_attribute_alias(std::string &);

// event methods

	void export_event(DevVarstd::stringArray *);
	void unexport_event(std::string &);
	CORBA::Any *import_event(std::string &);
*/
};

