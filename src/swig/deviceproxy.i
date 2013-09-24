/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
   
   PyTango is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   PyTango is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

 %{
 #include <tango.h>
 %}

namespace Tango
{

class DeviceProxy: public Connection
{
public:
	%newobject black_box;
	%newobject command_list_query;
	%newobject get_attribute_list;
	%newobject get_attribute_config;
	%newobject get_attribute_ex;
	%newobject attribute_list_query;
	%newobject attribute_list_query_ex;
	%newobject polling_status;

	DeviceProxy();
	DeviceProxy(std::string &name, CORBA::ORB *orb=NULL);
	DeviceProxy(std::string &name, bool ch_access, CORBA::ORB *orb=NULL);
	DeviceProxy(const char *, bool ch_access, CORBA::ORB *orb=NULL);
	DeviceProxy(const char *, CORBA::ORB *orb=NULL);

	DeviceProxy(const DeviceProxy &);
	DeviceProxy & operator=(const DeviceProxy &);
	virtual ~DeviceProxy();

//
// general methods
//

	virtual Tango::DeviceInfo const &info();
	virtual inline std::string dev_name() { return device_name; }
	virtual void parse_name(std::string &);
	virtual Tango::Database *get_device_db();

	virtual std::string status();
	virtual Tango::DevState state();
	virtual std::string adm_name();
	virtual std::string description();
	virtual std::string name();
	virtual std::string alias();

	int get_tango_lib_version();

	virtual int ping();
	virtual std::vector<std::string> *black_box(int);
//
// device methods
//
	virtual Tango::CommandInfo command_query(std::string);
	virtual Tango::CommandInfoList *command_list_query();

	virtual Tango::DbDevImportInfo import_info();
//
// property methods
//
	virtual void get_property(std::string&, Tango::DbData&);
	virtual void get_property(std::vector<std::string>&, Tango::DbData&);
	virtual void get_property(Tango::DbData&);
	virtual void put_property(Tango::DbData&);
	virtual void delete_property(std::string&);
	virtual void delete_property(std::vector<std::string>&);
	virtual void delete_property(Tango::DbData&);
	virtual void get_property_list(const std::string &,std::vector<std::string> &);
//
// attribute methods
//
	virtual std::vector<std::string> *get_attribute_list(); /* MEMORY LEAK */

	virtual Tango::AttributeInfoList *get_attribute_config(std::vector<std::string>&);
	virtual Tango::AttributeInfoListEx *get_attribute_config_ex(std::vector<std::string>&);
	virtual Tango::AttributeInfoEx get_attribute_config(const std::string &);

	virtual Tango::AttributeInfoEx attribute_query(std::string name);
	virtual Tango::AttributeInfoList *attribute_list_query();
	virtual Tango::AttributeInfoListEx *attribute_list_query_ex();

	virtual void set_attribute_config(Tango::AttributeInfoList &);
	virtual void set_attribute_config(Tango::AttributeInfoListEx &);

	virtual Tango::DeviceAttribute read_attribute(std::string&);
	virtual Tango::DeviceAttribute read_attribute(const char *at);
	void read_attribute(const char *, Tango::DeviceAttribute &);
	void read_attribute(std::string &at, Tango::DeviceAttribute &da);
	virtual std::vector<Tango::DeviceAttribute> *read_attributes(std::vector<std::string>&);

	virtual void write_attribute(Tango::DeviceAttribute&);
	virtual void write_attributes(std::vector<Tango::DeviceAttribute>&);

	virtual Tango::DeviceAttribute write_read_attribute(Tango::DeviceAttribute &);

//
// history methods
//
	virtual std::vector<Tango::DeviceDataHistory> *command_history(std::string &,int);
	virtual std::vector<Tango::DeviceDataHistory> *command_history(const char *na,int n);

	virtual std::vector<Tango::DeviceAttributeHistory> *attribute_history(std::string &,int);
	virtual std::vector<Tango::DeviceAttributeHistory> *attribute_history(const char *na,int n);
//
// Polling administration methods
//
	virtual std::vector<std::string> *polling_status();

	virtual void poll_command(std::string &, int);
	virtual void poll_command(const char *na, int per);
	virtual void poll_attribute(std::string &, int);
	virtual void poll_attribute(const char *na, int per);

	virtual int get_command_poll_period(std::string &);
	virtual int get_command_poll_period(const char *na)
			{std::string tmp(na);return get_command_poll_period(tmp);}
	virtual int get_attribute_poll_period(std::string &);
	virtual int get_attribute_poll_period(const char *na)
			{std::string tmp(na);return get_attribute_poll_period(tmp);}

	virtual bool is_command_polled(std::string &);
	virtual bool is_command_polled(const char *na);
	virtual bool is_attribute_polled(std::string &);
	virtual bool is_attribute_polled(const char *na);

	virtual void stop_poll_command(std::string &);
	virtual void stop_poll_command(const char *na);
	virtual void stop_poll_attribute(std::string &);
	virtual void stop_poll_attribute(const char *na);
//
// Asynchronous methods
//
	virtual long read_attribute_asynch(const char *na);
	virtual long read_attribute_asynch(std::string &att_name);
	virtual long read_attributes_asynch(std::vector <std::string> &);

	virtual std::vector<Tango::DeviceAttribute> *read_attributes_reply(long);
	virtual std::vector<Tango::DeviceAttribute> *read_attributes_reply(long,long);
	virtual Tango::DeviceAttribute *read_attribute_reply(long);
	virtual Tango::DeviceAttribute *read_attribute_reply(long,long);

	virtual long write_attribute_asynch(Tango::DeviceAttribute &);
	virtual long write_attributes_asynch(std::vector<Tango::DeviceAttribute> &);

	virtual void write_attributes_reply(long);
	virtual void write_attributes_reply(long,long);
	virtual void write_attribute_reply(long id) {write_attributes_reply(id);}
	virtual void write_attribute_reply(long to,long id) {write_attributes_reply(to,id);}

	virtual long pending_asynch_call(Tango::asyn_req_type req)
			{if (req == POLLING)return pasyn_ctr;
			else if (req==CALL_BACK) return pasyn_cb_ctr;
			else return (pasyn_ctr + pasyn_cb_ctr);}

	virtual void read_attributes_asynch(std::vector<std::string> &, Tango::CallBack &);
	virtual void read_attribute_asynch(const char *na, Tango::CallBack &cb);
	virtual void read_attribute_asynch(std::string &, Tango::CallBack &);

	virtual void write_attribute_asynch(Tango::DeviceAttribute &, Tango::CallBack &);
	virtual void write_attributes_asynch(std::vector<Tango::DeviceAttribute> &, Tango::CallBack &);
//
// Logging administration methods
//
#ifdef TANGO_HAS_LOG4TANGO
	virtual void add_logging_target(const std::string &target_type_name);
	virtual void add_logging_target(const char *target_type_name)
			{add_logging_target(std::string(target_type_name));}

	virtual void remove_logging_target(const std::string &target_type_name);
	virtual void remove_logging_target(const char *target_type_name)
			{remove_logging_target(std::string(target_type_name));}

	virtual std::vector<std::string> get_logging_target (void);
	virtual int get_logging_level (void);
	virtual void set_logging_level (int level);
#endif // TANGO_HAS_LOG4TANGO
//
// Event methods
//
	virtual int subscribe_event(const std::string &attr_name, Tango::EventType event, Tango::CallBack *,
	                   const std::vector<std::string> &filters);  // For compatibility with Tango < 8
	virtual int subscribe_event(const std::string &attr_name, Tango::EventType event, Tango::CallBack *,
	                   const std::vector<std::string> &filters, bool stateless); // For compatibility with Tango < 8
	virtual int subscribe_event(const std::string &attr_name, Tango::EventType event, int event_queue_size,
	                   const std::vector<std::string> &filters, bool stateless = false); // For compatibility with Tango < 8

	virtual int subscribe_event(const std::string &attr_name, Tango::EventType event, Tango::CallBack *);
	virtual int subscribe_event(const std::string &attr_name, Tango::EventType event, Tango::CallBack *,bool stateless);
	virtual int subscribe_event(const std::string &attr_name, Tango::EventType event, int event_queue_size,bool stateless = false);

	virtual void unsubscribe_event(int event_id);
//
// Methods to access data in event queues
//
	virtual void get_events (int event_id, Tango::EventDataList &event_list);
	virtual void get_events (int event_id, Tango::AttrConfEventDataList &event_list);
	virtual void get_events (int event_id, Tango::DataReadyEventDataList &event_list);
	virtual void get_events (int event_id, Tango::CallBack *cb);
	virtual int  event_queue_size(int event_id);
	virtual Tango::TimeVal get_last_event_date(int event_id);
	virtual bool is_event_queue_empty(int event_id);

//
// Locking methods
//
	virtual void lock(int lock_validity=DEFAULT_LOCK_VALIDITY);
	virtual void unlock(bool force=false);
	virtual std::string locking_status();
	virtual bool is_locked();
	virtual bool is_locked_by_me();
	virtual bool get_locker(Tango::LockerInfo &);
};

}