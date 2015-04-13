/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

class DeviceProxy : Connection {

%TypeHeaderCode
#include <tango.h>
using namespace Tango;
%End

protected:
    virtual std::string get_corba_name(bool);
    virtual std::string build_corba_name();
    virtual int get_lock_ctr();
    virtual void set_lock_ctr(int);

public:
    DeviceProxy();
    DeviceProxy(const char *, bool ch_access);
    DeviceProxy(const char *);

    DeviceProxy(const DeviceProxy &);
//    DeviceProxy & operator=(const DeviceProxy &);

//
// general methods
//

    virtual const DeviceInfo & info();
    virtual std::string dev_name();
    virtual void parse_name(std::string &);
    virtual Database *get_device_db();

    virtual std::string status();
    virtual DevState state();
    virtual std::string adm_name();
    virtual std::string description();
    virtual std::string name();
    virtual std::string alias();

    virtual int ping();

    virtual std::vector<std::string>* black_box(int);

//
// device methods
//
    virtual CommandInfo command_query(std::string);
    virtual CommandInfoList* command_list_query();
    

    virtual DbDevImportInfo import_info();
//
// property methods
//
    virtual void get_property(std::string& /In/, DbData& /Out/);
    virtual void get_property(std::vector<std::string> /In/, DbData& /Out/);
    virtual void get_property(DbData&);
    virtual void put_property(DbData& /In/);
    virtual void delete_property(std::string& /In/);
    virtual void delete_property(std::vector<std::string>& /In/);
    virtual void delete_property(DbData& /In/);
    virtual void get_property_list(const std::string &, std::vector<std::string> & /Out/);

//
// attribute methods
//
    virtual std::vector<std::string> *get_attribute_list();

    virtual AttributeInfoList *get_attribute_config(std::vector<std::string>&);
    virtual AttributeInfoListEx *get_attribute_config_ex(std::vector<std::string>&);

    virtual AttributeInfoEx get_attribute_config(const std::string &);

    virtual AttributeInfoEx attribute_query(std::string name);

    virtual AttributeInfoList *attribute_list_query();
    virtual AttributeInfoListEx *attribute_list_query_ex();

    virtual void set_attribute_config(AttributeInfoList &);
    virtual void set_attribute_config(AttributeInfoListEx &);
/*
    virtual DeviceAttribute read_attribute(std::string&);
    virtual DeviceAttribute read_attribute(const char *at);
    void read_attribute(const char *,DeviceAttribute &);
    void read_attribute(std::string &at,DeviceAttribute &da);
    virtual std::vector<DeviceAttribute> *read_attributes(std::vector<std::string>&);

    virtual void write_attribute(DeviceAttribute&);
    virtual void write_attributes(std::vector<DeviceAttribute>&);

    virtual DeviceAttribute write_read_attribute(DeviceAttribute &);

//
// history methods
//
    virtual std::vector<DeviceDataHistory> *command_history(std::string &,int);
    virtual std::vector<DeviceDataHistory> *command_history(const char *na,int n)
            {std::string str(na);return command_history(str,n);}

    virtual std::vector<DeviceAttributeHistory> *attribute_history(std::string &,int);
    virtual std::vector<DeviceAttributeHistory> *attribute_history(const char *na,int n)
            {std::string str(na);return attribute_history(str,n);}
//
// Polling administration methods
//
    virtual std::vector<std::string> *polling_status();

    virtual void poll_command(std::string &, int);
    virtual void poll_command(const char *na, int per) {std::string tmp(na);poll_command(tmp,per);}
    virtual void poll_attribute(std::string &, int);
    virtual void poll_attribute(const char *na, int per) {std::string tmp(na);poll_attribute(tmp,per);}

    virtual int get_command_poll_period(std::string &);
    virtual int get_command_poll_period(const char *na)
            {std::string tmp(na);return get_command_poll_period(tmp);}
    virtual int get_attribute_poll_period(std::string &);
    virtual int get_attribute_poll_period(const char *na)
            {std::string tmp(na);return get_attribute_poll_period(tmp);}

    virtual bool is_command_polled(std::string &);
    virtual bool is_command_polled(const char *na) {std::string tmp(na);return is_command_polled(tmp);}
    virtual bool is_attribute_polled(std::string &);
    virtual bool is_attribute_polled(const char *na) {std::string tmp(na);return is_attribute_polled(tmp);}

    virtual void stop_poll_command(std::string &);
    virtual void stop_poll_command(const char *na) {std::string tmp(na);stop_poll_command(tmp);}
    virtual void stop_poll_attribute(std::string &);
    virtual void stop_poll_attribute(const char *na) {std::string tmp(na);stop_poll_attribute(tmp);}
//
// Asynchronous methods
//
    virtual long read_attribute_asynch(const char *na) {std::string tmp(na);return read_attribute_asynch(tmp);}
    virtual long read_attribute_asynch(std::string &att_name);
    virtual long read_attributes_asynch(std::vector <std::string> &);

    virtual std::vector<DeviceAttribute> *read_attributes_reply(long);
    virtual std::vector<DeviceAttribute> *read_attributes_reply(long,long);
    virtual DeviceAttribute *read_attribute_reply(long);
    virtual DeviceAttribute *read_attribute_reply(long,long);

    virtual long write_attribute_asynch(DeviceAttribute &);
    virtual long write_attributes_asynch(std::vector<DeviceAttribute> &);

    virtual void write_attributes_reply(long);
    virtual void write_attributes_reply(long,long);
    virtual void write_attribute_reply(long id) {write_attributes_reply(id);}
    virtual void write_attribute_reply(long to,long id) {write_attributes_reply(to,id);}

    virtual long pending_asynch_call(asyn_req_type req)
            {if (req == POLLING)return pasyn_ctr;
            else if (req==CALL_BACK) return pasyn_cb_ctr;
            else return (pasyn_ctr + pasyn_cb_ctr);}

    virtual void read_attributes_asynch(std::vector<std::string> &,CallBack &);
    virtual void read_attribute_asynch(const char *na,CallBack &cb) {std::string tmp(na);read_attribute_asynch(tmp,cb);}
    virtual void read_attribute_asynch(std::string &,CallBack &);

    virtual void write_attribute_asynch(DeviceAttribute &,CallBack &);
    virtual void write_attributes_asynch(std::vector<DeviceAttribute> &,CallBack &);
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
    virtual int subscribe_event(const std::string &attr_name, EventType event, CallBack *,
                       const std::vector<std::string> &filters);  // For compatibility with Tango < 8
    virtual int subscribe_event(const std::string &attr_name, EventType event, CallBack *,
                       const std::vector<std::string> &filters, bool stateless); // For compatibility with Tango < 8
    virtual int subscribe_event(const std::string &attr_name, EventType event, int event_queue_size,
                       const std::vector<std::string> &filters, bool stateless = false); // For compatibility with Tango < 8

    virtual int subscribe_event(const std::string &attr_name, EventType event, CallBack *);
    virtual int subscribe_event(const std::string &attr_name, EventType event, CallBack *,bool stateless);
    virtual int subscribe_event(const std::string &attr_name, EventType event, int event_queue_size,bool stateless = false);

    virtual void unsubscribe_event(int event_id);
//
// Methods to access data in event queues
//
    virtual void get_events (int event_id, EventDataList &event_list);
    virtual void get_events (int event_id, AttrConfEventDataList &event_list);
    virtual void get_events (int event_id, DataReadyEventDataList &event_list);
    virtual void get_events (int event_id, CallBack *cb);
    virtual int  event_queue_size(int event_id);
    virtual TimeVal get_last_event_date(int event_id);
    virtual bool is_event_queue_empty(int event_id);

//
// Locking methods
//
    virtual void lock(int lock_validity=DEFAULT_LOCK_VALIDITY);
    virtual void unlock(bool force=false);
    virtual std::string locking_status();
    virtual bool is_locked();
    virtual bool is_locked_by_me();
    virtual bool get_locker(LockerInfo &);
*/
};

