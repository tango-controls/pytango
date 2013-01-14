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

#include "precompiled_header.hpp"
#include "device_attribute.h"
#include "callback.h"
#include "defs.h"
#include "pytgutils.h"

extern const char *param_must_be_seq;
extern const char *unreachable_code;
extern const char *non_string_seq;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(lock_overloads, Tango::DeviceProxy::lock, 0, 1);
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(unlock_overloads, Tango::DeviceProxy::unlock, 0, 1);

namespace PyDeviceProxy
{
    struct PickleSuite : bopy::pickle_suite
    {
        static bopy::tuple getinitargs(Tango::DeviceProxy& self)
        {
            std::string ret = self.get_db_host() + ":" + self.get_db_port() +
                              "/" + self.dev_name();
            return bopy::make_tuple(ret);
        }
    };
    
    static inline Tango::DevState state(Tango::DeviceProxy& self)
    {
        AutoPythonAllowThreads guard;
        return self.state();
    }

    static inline std::string status(Tango::DeviceProxy& self)
    {
        AutoPythonAllowThreads guard;
        return self.status();
    }

    static inline int ping(Tango::DeviceProxy& self)
    {
        AutoPythonAllowThreads guard;
        return self.ping();
    }

    static inline void pylist_to_devattrs(Tango::DeviceProxy& self,
        bopy::object &py_list, std::vector<Tango::DeviceAttribute> &dev_attrs)
    {
        std::vector<std::string> attr_names;
        std::vector<bopy::object> py_values;
        long size = len(py_list);

        // Fill attr_names and py_values
        for (long n = 0; n < size; ++n) {
            bopy::object tup = py_list[n];
            std::string attr_name = bopy::extract<std::string>(tup[0]);
            attr_names.push_back(attr_name);
            py_values.push_back(tup[1]);
        }

        // Get attr_info for all the attr_names
        unique_pointer<Tango::AttributeInfoListEx> attr_infos;
        {
            AutoPythonAllowThreads guard;
            attr_infos.reset(self.get_attribute_config_ex(attr_names));
        }

        // Now prepare dev_attrs with attr_infos and py_values
        dev_attrs.resize(size);
        for (long n = 0; n < size; ++n) {
            PyDeviceAttribute::reset(dev_attrs[n], (*attr_infos)[n], py_values[n]);
        }
    }

    static inline bopy::object read_attribute(Tango::DeviceProxy& self, const string & attr_name, PyTango::ExtractAs extract_as)
    {
        // Even if there's an exception in convert_to_python, the
        // DeviceAttribute will be deleted there, so we don't need to worry.
        Tango::DeviceAttribute* dev_attr = 0;
        {
            AutoPythonAllowThreads guard;
            dev_attr = \
                    new Tango::DeviceAttribute(self.read_attribute(attr_name.c_str()));
        }
        return PyDeviceAttribute::convert_to_python(dev_attr, self, extract_as);
    }

    static inline bopy::object read_attributes(Tango::DeviceProxy& self, bopy::object py_attr_names, PyTango::ExtractAs extract_as)
    {
        CSequenceFromPython<StdStringVector> attr_names(py_attr_names);

        PyDeviceAttribute::AutoDevAttrVector dev_attr_vec;
        {
            AutoPythonAllowThreads guard;
            dev_attr_vec.reset(self.read_attributes(*attr_names));
        }

        return PyDeviceAttribute::convert_to_python(dev_attr_vec, self, extract_as);
    }

    static inline void write_attribute(Tango::DeviceProxy& self, const Tango::AttributeInfo & attr_info, bopy::object py_value)
    {
        Tango::DeviceAttribute da;
        PyDeviceAttribute::reset(da, attr_info, py_value);
        AutoPythonAllowThreads guard;
        self.write_attribute(da);
    }

    static inline void write_attribute(Tango::DeviceProxy& self, const string & attr_name, bopy::object py_value)
    {
        Tango::DeviceAttribute dev_attr;
        PyDeviceAttribute::reset(dev_attr, attr_name, self, py_value);
        {
            AutoPythonAllowThreads guard;
            self.write_attribute(dev_attr);
        }
    }

    static inline void write_attributes(Tango::DeviceProxy& self, bopy::object py_list)
    {
        std::vector<Tango::DeviceAttribute> dev_attrs;
        pylist_to_devattrs(self, py_list, dev_attrs);

        AutoPythonAllowThreads guard;
        self.write_attributes(dev_attrs);
    }

    static inline bopy::object write_read_attribute(Tango::DeviceProxy& self, const string & attr_name, bopy::object py_value, PyTango::ExtractAs extract_as)
    {
        Tango::DeviceAttribute w_dev_attr;
        unique_pointer<Tango::DeviceAttribute> r_dev_attr;

        // Prepare dev_attr structure
        PyDeviceAttribute::reset(w_dev_attr, attr_name, self, py_value);

        // Do the actual write_read_attribute thing...
        {
            AutoPythonAllowThreads guard;
            r_dev_attr.reset(new Tango::DeviceAttribute(self.write_read_attribute(w_dev_attr)));
        }

        // Convert the result back to python
        return PyDeviceAttribute::convert_to_python(r_dev_attr.release(), self, extract_as);
    }

    static inline bopy::object
    command_history(Tango::DeviceProxy& self, const std::string& cmd_name, int depth)
    {
        std::vector<Tango::DeviceDataHistory>* device_data_hist = NULL;
        bopy::list ret;
        {
            AutoPythonAllowThreads guard;
            device_data_hist =
                self.command_history(const_cast<std::string&>(cmd_name), depth);
        }
        vector<Tango::DeviceDataHistory>::iterator it = device_data_hist->begin();
        for(;it != device_data_hist->end(); ++it)
        {
            Tango::DeviceDataHistory& hist = *it;
            ret.append(hist);
        }
        delete device_data_hist;
        return ret;
    }

    static inline bopy::object
            attribute_history(Tango::DeviceProxy& self, const std::string & attr_name, int depth, PyTango::ExtractAs extract_as)
    {
        unique_pointer< vector<Tango::DeviceAttributeHistory> > att_hist;
        {
            AutoPythonAllowThreads guard;
            att_hist.reset(self.attribute_history(const_cast<std::string&>(attr_name), depth));
        }
        return PyDeviceAttribute::convert_to_python(att_hist, self, extract_as);
    }


    static inline long read_attributes_asynch(Tango::DeviceProxy& self, bopy::object py_attr_names)
    {
        CSequenceFromPython<StdStringVector> attr_names(py_attr_names);

        AutoPythonAllowThreads guard;
        return self.read_attributes_asynch(*attr_names);
    }



    static inline bopy::object read_attributes_reply(Tango::DeviceProxy& self, long id, PyTango::ExtractAs extract_as)
    {
        PyDeviceAttribute::AutoDevAttrVector dev_attr_vec;
        {
            AutoPythonAllowThreads guard;
            dev_attr_vec.reset(self.read_attributes_reply(id));
        }
        return PyDeviceAttribute::convert_to_python(dev_attr_vec, self, extract_as);
    }

    static inline bopy::object read_attributes_reply(Tango::DeviceProxy& self, long id, long timeout, PyTango::ExtractAs extract_as)
    {
        PyDeviceAttribute::AutoDevAttrVector dev_attr_vec;
        {
            AutoPythonAllowThreads guard;
            dev_attr_vec.reset(self.read_attributes_reply(id, timeout));
        }
        return PyDeviceAttribute::convert_to_python(dev_attr_vec, self, extract_as);
    }

    static inline long write_attributes_asynch(Tango::DeviceProxy& self, bopy::object py_list)
    {
        std::vector<Tango::DeviceAttribute> dev_attrs;
        pylist_to_devattrs(self, py_list, dev_attrs);

        AutoPythonAllowThreads guard;
        return self.write_attributes_asynch(dev_attrs);
    }

    static inline void write_attributes_reply(Tango::DeviceProxy& self, long id, long timestamp)
    {
        AutoPythonAllowThreads guard;
        self.write_attributes_reply(id, timestamp);
    }

    static inline void write_attributes_reply(Tango::DeviceProxy& self, long id)
    {
        AutoPythonAllowThreads guard;
        self.write_attributes_reply(id);
    }

    static inline void read_attributes_asynch(bopy::object py_self, bopy::object py_attr_names, bopy::object py_cb, PyTango::ExtractAs extract_as)
    {
        Tango::DeviceProxy* self = bopy::extract<Tango::DeviceProxy*>(py_self);
        CSequenceFromPython<StdStringVector> attr_names(py_attr_names);

        PyCallBackAutoDie* cb = bopy::extract<PyCallBackAutoDie*>(py_cb);
        cb->set_autokill_references(py_cb, py_self);
        cb->set_extract_as(extract_as);

        try {
            AutoPythonAllowThreads guard;
            self->read_attributes_asynch(*attr_names, *cb);
        } catch (...) {
            cb->unset_autokill_references();
            throw;
        }
    }

    static inline void write_attributes_asynch(bopy::object py_self, bopy::object py_list, bopy::object py_cb)
    {
        Tango::DeviceProxy* self = bopy::extract<Tango::DeviceProxy*>(py_self);
        std::vector<Tango::DeviceAttribute> dev_attrs;
        pylist_to_devattrs(*self, py_list, dev_attrs);

        PyCallBackAutoDie* cb = bopy::extract<PyCallBackAutoDie*>(py_cb);
        cb->set_autokill_references(py_cb, py_self);

        try {
            AutoPythonAllowThreads guard;
            self->write_attributes_asynch(dev_attrs, *cb);
        } catch (...) {
            cb->unset_autokill_references();
            throw;
        }
    }

    static int subscribe_event(
            bopy::object py_self,
            const string &attr_name,
            Tango::EventType event,
            bopy::object py_cb_or_queuesize,
            bopy::object &py_filters,
            bool stateless,
            PyTango::ExtractAs extract_as )
    {
        Tango::DeviceProxy& self = bopy::extract<Tango::DeviceProxy&>(py_self);
        CSequenceFromPython<StdStringVector> filters(py_filters);

        PyCallBackPushEvent* cb = 0;
        int event_queue_size = 0;
        if (bopy::extract<PyCallBackPushEvent&>(py_cb_or_queuesize).check()) {
            cb = bopy::extract<PyCallBackPushEvent*>(py_cb_or_queuesize);

            cb->set_device(py_self);
            cb->set_extract_as(extract_as);

            AutoPythonAllowThreads guard;
            return self.subscribe_event(attr_name, event, cb, *filters, stateless);
        } else {
            event_queue_size = bopy::extract<int>(py_cb_or_queuesize);
            AutoPythonAllowThreads guard;
            return self.subscribe_event(attr_name, event, event_queue_size,
                                        *filters, stateless);
        }
    }

    static void unsubscribe_event(Tango::DeviceProxy& self, int event)
    {
        // If the callback is running, unsubscribe_event will lock
        // until it finishes. So we MUST release GIL to avoid a deadlock
        AutoPythonAllowThreads guard;
        self.unsubscribe_event(event);
    }

    template<typename ED, typename EDList>
    static bopy::object
    get_events__aux(bopy::object py_self, int event_id,
                    PyTango::ExtractAs extract_as=PyTango::ExtractAsNumpy)
    {
        Tango::DeviceProxy &self = bopy::extract<Tango::DeviceProxy&>(py_self);

        EDList event_list;
        self.get_events(event_id, event_list);

        bopy::list r;

        for (size_t i=0; i < event_list.size(); ++i) {
            ED* event_data = event_list[i];

            bopy::object py_ev(bopy::handle<>(
                bopy::to_python_indirect<
                    ED*, bopy::detail::make_owning_holder>()(event_data)));

            // EventDataList deletes EventData's on destructor, so once
            // we are handling it somewhere else (as an bopy::object) we must
            // unset the reference
            event_list[i] = 0;

            PyCallBackPushEvent::fill_py_event(event_data, py_ev, py_self, extract_as);

            r.append(py_ev);
        }
        return r;
    }

    static void
    get_events__callback(bopy::object py_self, int event_id,
                         PyCallBackPushEvent *cb, PyTango::ExtractAs extract_as)
    {
        Tango::DeviceProxy& self = bopy::extract<Tango::DeviceProxy&>(py_self);

        cb->set_device(py_self);
        cb->set_extract_as(extract_as);

        self.get_events(event_id, cb);
    }

    static bopy::object
    get_events__attr_conf(bopy::object py_self, int event_id)
    {
        return get_events__aux<Tango::AttrConfEventData, Tango::AttrConfEventDataList>
                                                (py_self, event_id);
    }

    static bopy::object
    get_events__data(bopy::object py_self, int event_id, PyTango::ExtractAs extract_as)
    {
        return get_events__aux<Tango::EventData, Tango::EventDataList>
                                                (py_self, event_id, extract_as);
    }

    static bopy::object
    get_events__data_ready(bopy::object py_self, int event_id)
    {
        return get_events__aux<Tango::DataReadyEventData, Tango::DataReadyEventDataList>
                                                (py_self, event_id);
    }
};

void export_device_proxy()
{
    // The following function declarations are necessary to be able to cast
    // the function parameters from string& to const string&, otherwise python
    // will not recognize the method calls

    void (Tango::DeviceProxy::*get_property_)(std::string &, Tango::DbData &) =
        &Tango::DeviceProxy::get_property;

    void (Tango::DeviceProxy::*delete_property_)(std::string &) =
        &Tango::DeviceProxy::delete_property;

    bopy::class_<Tango::DeviceProxy, bopy::bases<Tango::Connection> >
        DeviceProxy("DeviceProxy", bopy::init<>())
    ;

    DeviceProxy
        .def(bopy::init<const char *>())
        .def(bopy::init<const char *, bool>())
        .def(bopy::init<const Tango::DeviceProxy &>())

        //
        // Pickle
        //
        .def_pickle(PyDeviceProxy::PickleSuite())

        //
        // general methods
        //
        .def("dev_name", &Tango::DeviceProxy::dev_name)
        
        .def("info", &Tango::DeviceProxy::info,
            ( arg_("self") ),
            bopy::return_internal_reference<1>() )

        .def("get_device_db", &Tango::DeviceProxy::get_device_db,
            bopy::return_value_policy<bopy::reference_existing_object>())

        .def("status", &PyDeviceProxy::status,
            ( arg_("self") ) )

        .def("state", &PyDeviceProxy::state,
            ( arg_("self") ) )

        .def("adm_name", &Tango::DeviceProxy::adm_name,
            ( arg_("self") ) )

        .def("description", &Tango::DeviceProxy::description,
            ( arg_("self") ) )

        .def("name", &Tango::DeviceProxy::name,
            ( arg_("self") ) )

        .def("alias", &Tango::DeviceProxy::alias,
            ( arg_("self") ) )

        .def("ping", &PyDeviceProxy::ping,
            ( arg_("self") ) )
            

        .def("black_box", &Tango::DeviceProxy::black_box,
            ( arg_("self"), arg_("n") ),
            bopy::return_value_policy<bopy::manage_new_object>() )

        //
        // device methods
        //

        .def("command_query", &Tango::DeviceProxy::command_query,
            ( arg_("self"), arg_("command") ) )

        .def("command_list_query", &Tango::DeviceProxy::command_list_query,
            ( arg_("self") ),
            bopy::return_value_policy<bopy::manage_new_object>() )

        .def("import_info", &Tango::DeviceProxy::import_info,
            ( arg_("self") ) )

        //
        // property methods
        //
        .def("_get_property",
            (void (Tango::DeviceProxy::*) (const std::string &, Tango::DbData &))
            get_property_,
            ( arg_("self"), arg_("propname"), arg_("propdata") ) )

        .def("_get_property",
            (void (Tango::DeviceProxy::*) (std::vector<std::string>&, Tango::DbData &))
            &Tango::DeviceProxy::get_property,
            ( arg_("self"), arg_("propnames"), arg_("propdata") ) )

        .def("_get_property",
            (void (Tango::DeviceProxy::*) (Tango::DbData &))
            &Tango::DeviceProxy::get_property,
            ( arg_("self"), arg_("propdata") ) )

        .def("_put_property", &Tango::DeviceProxy::put_property,
            ( arg_("self"), arg_("propdata") ) )

        .def("_delete_property", (void (Tango::DeviceProxy::*) (const std::string &))
            delete_property_,
            ( arg_("self"), arg_("propname") ) )

        .def("_delete_property", (void (Tango::DeviceProxy::*) (StdStringVector &))
            &Tango::DeviceProxy::delete_property,
            ( arg_("self"), arg_("propnames") ) )

        .def("_delete_property", (void (Tango::DeviceProxy::*) (Tango::DbData &))
            &Tango::DeviceProxy::delete_property,
            ( arg_("self"), arg_("propdata") ) )

        .def("_get_property_list", &Tango::DeviceProxy::get_property_list,
            ( arg_("self"), arg_("filter"), arg_("array") ) )

        //
        // attribute methods
        //

        .def("get_attribute_list", &Tango::DeviceProxy::get_attribute_list,
            ( arg_("self") ),
            bopy::return_value_policy<bopy::manage_new_object>() )

        .def("_get_attribute_config",
            (Tango::AttributeInfoList* (Tango::DeviceProxy::*)(StdStringVector &))
            &Tango::DeviceProxy::get_attribute_config,
            ( arg_("self"), arg_("attr_names") ),
            bopy::return_value_policy<bopy::manage_new_object>() )

        .def("_get_attribute_config",
            (Tango::AttributeInfoEx (Tango::DeviceProxy::*)(const std::string&))
            &Tango::DeviceProxy::get_attribute_config,
            ( arg_("self"), arg_("attr_name") ) )

        .def("_get_attribute_config_ex",
            &Tango::DeviceProxy::get_attribute_config_ex,
            ( arg_("self"), arg_("attr_names") ),
            bopy::return_value_policy<bopy::manage_new_object>() )

        .def("attribute_query", &Tango::DeviceProxy::attribute_query,
            ( arg_("self"), arg_("attr_name") ) )

        .def("attribute_list_query", &Tango::DeviceProxy::attribute_list_query,
            ( arg_("self") ),
            bopy::return_value_policy<bopy::manage_new_object>() )

        .def("attribute_list_query_ex",
            &Tango::DeviceProxy::attribute_list_query_ex,
            ( arg_("self") ),
            bopy::return_value_policy<bopy::manage_new_object>() )

        .def("_set_attribute_config",
            (void (Tango::DeviceProxy::*)(Tango::AttributeInfoList &))
            &Tango::DeviceProxy::set_attribute_config,
            ( arg_("self"), arg_("seq") ) )

        .def("_set_attribute_config",
            (void (Tango::DeviceProxy::*)(Tango::AttributeInfoListEx &))
            &Tango::DeviceProxy::set_attribute_config,
            ( arg_("self"), arg_("seq") ) )

        .def("_read_attribute",
            &PyDeviceProxy::read_attribute,
            ( arg_("self"), arg_("attr_name"), arg_("extract_as")=PyTango::ExtractAsNumpy ) )

        .def("read_attributes",
            &PyDeviceProxy::read_attributes,
            ( arg_("self"), arg_("attr_names"), arg_("extract_as")=PyTango::ExtractAsNumpy ) )

        .def("write_attribute",
            (void (*)(Tango::DeviceProxy&, const string &, bopy::object ))
            &PyDeviceProxy::write_attribute,
            ( arg_("self"), arg_("attr_name"), arg_("value") ) )

        .def("write_attribute",
            (void (*)(Tango::DeviceProxy&, const Tango::AttributeInfo &, bopy::object ))
            &PyDeviceProxy::write_attribute,
            ( arg_("self"), arg_("attr_info"), arg_("value") ) )

        .def("write_attributes",
            &PyDeviceProxy::write_attributes,
            ( arg_("self"), arg_("name_val") ) )

        .def("_write_read_attribute",
            &PyDeviceProxy::write_read_attribute,
            ( arg_("self"), arg_("attr_name"), arg_("value"), arg_("extract_as")=PyTango::ExtractAsNumpy ) )

        //
        // history methods
        //

        .def("command_history", &PyDeviceProxy::command_history,
            (arg_("self"), arg_("cmd_name"), arg_("depth")))

        .def("attribute_history", &PyDeviceProxy::attribute_history,
            (   arg_("self"),
                arg_("attr_name"),
                arg_("depth"),
                arg_("extract_as")=PyTango::ExtractAsNumpy ) )

        //
        // Polling administration methods
        //

        .def("polling_status", &Tango::DeviceProxy::polling_status,
            ( arg_("self") ),
            bopy::return_value_policy<bopy::manage_new_object>() )

        .def("poll_command",
            (void (Tango::DeviceProxy::*)(const char *, int)) &Tango::DeviceProxy::poll_command,
            ( arg_("self"), arg_("cmd_name"), arg_("period") ) )

        .def("poll_attribute",
            (void (Tango::DeviceProxy::*)(const char *, int)) &Tango::DeviceProxy::poll_attribute,
            ( arg_("self"), arg_("attr_name"), arg_("period") ) )

        .def("get_command_poll_period",
            (int (Tango::DeviceProxy::*)(const char *)) &Tango::DeviceProxy::get_command_poll_period,
            ( arg_("self"), arg_("cmd_name") ) )

        .def("get_attribute_poll_period",
            (int (Tango::DeviceProxy::*)(const char *)) &Tango::DeviceProxy::get_attribute_poll_period,
            ( arg_("self"), arg_("attr_name") ) )

        .def("is_command_polled",
            (bool (Tango::DeviceProxy::*)(const char *)) &Tango::DeviceProxy::is_command_polled,
            ( arg_("self"), arg_("cmd_name") ) )

        .def("is_attribute_polled",
            (bool (Tango::DeviceProxy::*)(const char *)) &Tango::DeviceProxy::is_attribute_polled,
            ( arg_("self"), arg_("attr_name") ) )

        .def("stop_poll_command",
            (void (Tango::DeviceProxy::*)(const char *)) &Tango::DeviceProxy::stop_poll_command,
            ( arg_("self"), arg_("cmd_name") ) )

        .def("stop_poll_attribute",
            (void (Tango::DeviceProxy::*)(const char *)) &Tango::DeviceProxy::stop_poll_attribute,
            ( arg_("self"), arg_("attr_name") ) )

        //
        // Asynchronous methods
        //
        .def("__read_attributes_asynch",
            (long (*) (Tango::DeviceProxy &, bopy::object))
            &PyDeviceProxy::read_attributes_asynch,
            ( arg_("self"), arg_("attr_names") ) )

        .def("read_attributes_reply",
            (bopy::object (*) (Tango::DeviceProxy &, long, PyTango::ExtractAs))
            &PyDeviceProxy::read_attributes_reply,
            ( arg_("self"), arg_("id"), arg_("extract_as")=PyTango::ExtractAsNumpy ) )

        .def("read_attributes_reply",
            (bopy::object (*) (Tango::DeviceProxy &, long, long, PyTango::ExtractAs))
            &PyDeviceProxy::read_attributes_reply,
            ( arg_("self"), arg_("id"), arg_("timeout"), arg_("extract_as")=PyTango::ExtractAsNumpy ) )

        .def("pending_asynch_call",
            &Tango::DeviceProxy::pending_asynch_call,
            ( arg_("self"), arg_("req_type") ) )

        .def("__write_attributes_asynch",
            (long (*) (Tango::DeviceProxy &, bopy::object))
            &PyDeviceProxy::write_attributes_asynch,
            ( arg_("self"), arg_("values") ) )

        .def("write_attributes_reply",
            (void (*) (Tango::DeviceProxy &, long))
            &PyDeviceProxy::write_attributes_reply,
            ( arg_("self"), arg_("id") ) )

        .def("write_attributes_reply",
            (void (*) (Tango::DeviceProxy &, long, long))
            &PyDeviceProxy::write_attributes_reply,
            ( arg_("self"), arg_("id"), arg_("timeout") ) )

        .def("__read_attributes_asynch",
            (void (*) (bopy::object, bopy::object, bopy::object, PyTango::ExtractAs))
            &PyDeviceProxy::read_attributes_asynch,
            ( arg_("self"), arg_("attr_names"), arg_("callback"), arg_("extract_as")=PyTango::ExtractAsNumpy ) )

        .def("__write_attributes_asynch",
            (void (*) (bopy::object, bopy::object, bopy::object))
            &PyDeviceProxy::write_attributes_asynch,
            ( arg_("self"), arg_("values"), arg_("callback") ) )
        
        //
        // Logging administration methods
        //

        .def("add_logging_target",
            (void (Tango::DeviceProxy::*)(const std::string &)) &Tango::DeviceProxy::add_logging_target,
            ( arg_("self"), arg_("target_type_target_name") ) )

        .def("remove_logging_target",
            (void (Tango::DeviceProxy::*)(const std::string &)) &Tango::DeviceProxy::remove_logging_target,
            ( arg_("self"), arg_("target_type_target_name") ) )

        .def("get_logging_target", &Tango::DeviceProxy::get_logging_target,
            ( arg_("self") ) )

        .def("get_logging_level", &Tango::DeviceProxy::get_logging_level,
            ( arg_("self") ) )

        .def("set_logging_level", &Tango::DeviceProxy::set_logging_level,
            ( arg_("self"), arg_("level") ))

        //
        // Event methods
        //

        .def("__subscribe_event", &PyDeviceProxy::subscribe_event,
            (   arg_("self"),
                arg_("attr_name"),
                arg_("event"),
                arg_("cb_or_queuesize"),
                arg_("filters")=bopy::list(),
                arg_("stateless")=false,
                arg_("extract_as")=PyTango::ExtractAsNumpy )
            )

        .def("__unsubscribe_event", &PyDeviceProxy::unsubscribe_event )

        .def("__get_callback_events", PyDeviceProxy::get_events__callback,
            ( arg_("self"), arg_("event_id"), arg_("callback"), arg_("extract_as")=PyTango::ExtractAsNumpy) )

        .def("__get_attr_conf_events", PyDeviceProxy::get_events__attr_conf,
            ( arg_("self"), arg_("event_id")) )

        .def("__get_data_events", PyDeviceProxy::get_events__data,
            ( arg_("self"), arg_("event_id"), arg_("extract_as")=PyTango::ExtractAsNumpy ))

        .def("__get_data_ready_events", PyDeviceProxy::get_events__data_ready,
            ( arg_("self"), arg_("event_id")) )

        // methods to access data in event queues
        .def("event_queue_size", &Tango::DeviceProxy::event_queue_size,
            ( arg_("self"), arg_("event_id") ) )

        .def("get_last_event_date", &Tango::DeviceProxy::get_last_event_date,
            ( arg_("self"), arg_("event_id") ) )

        .def("is_event_queue_empty", &Tango::DeviceProxy::is_event_queue_empty,
            ( arg_("self"), arg_("event_id") ) )

        //
        // Locking methods
        //
        .def("lock", &Tango::DeviceProxy::lock,
            lock_overloads( ( arg_("lock_validity") )))

        .def("unlock", &Tango::DeviceProxy::unlock,
            unlock_overloads( arg_("force")))

        .def("locking_status", &Tango::DeviceProxy::locking_status,
            ( arg_("self") ))

        .def("is_locked", &Tango::DeviceProxy::is_locked,
            ( arg_("self") ))

        .def("is_locked_by_me", &Tango::DeviceProxy::is_locked_by_me,
            ( arg_("self") ))

        .def("get_locker", &Tango::DeviceProxy::get_locker,
            ( arg_("self"), arg_("lockinfo") ))

        /// This is to be used by the python layer of this api...
        //.setattr("__subscribed_events", bopy::dict())
        ;
}
