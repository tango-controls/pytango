/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "defs.h"
#include "to_py.h"
#include "pyutils.h"

using namespace boost::python;

object to_py(const Tango::AttributeAlarm &attr_alarm)
{
    PYTANGO_MOD
    object py_attr_alarm = pytango.attr("AttributeAlarm")();
    
    py_attr_alarm.attr("min_alarm") = str(attr_alarm.min_alarm.in());
    py_attr_alarm.attr("max_alarm") = str(attr_alarm.max_alarm.in());
    py_attr_alarm.attr("min_warning") = str(attr_alarm.min_warning.in());
    py_attr_alarm.attr("max_warning") = str(attr_alarm.max_warning.in());
    py_attr_alarm.attr("delta_t") = str(attr_alarm.delta_t.in());
    py_attr_alarm.attr("delta_val") = str(attr_alarm.delta_val.in());
    py_attr_alarm.attr("extensions") = 
        CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(attr_alarm.extensions);
    
    return py_attr_alarm;
}

object to_py(const Tango::ChangeEventProp &change_event_prop)
{
    PYTANGO_MOD
    object py_change_event_prop = pytango.attr("ChangeEventProp")();
    
    py_change_event_prop.attr("rel_change") = str(change_event_prop.rel_change.in());
    py_change_event_prop.attr("abs_change") = str(change_event_prop.abs_change.in());
    py_change_event_prop.attr("extensions") = 
        CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(change_event_prop.extensions);
    
    return py_change_event_prop;
}

object to_py(const Tango::PeriodicEventProp &periodic_event_prop)
{
    PYTANGO_MOD
    object py_periodic_event_prop = pytango.attr("PeriodicEventProp")();
    
    py_periodic_event_prop.attr("period") = str(periodic_event_prop.period.in());
    py_periodic_event_prop.attr("extensions") = 
        CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(periodic_event_prop.extensions);
    
    return py_periodic_event_prop;
}

object to_py(const Tango::ArchiveEventProp &archive_event_prop)
{
    PYTANGO_MOD
    object py_archive_event_prop = pytango.attr("ArchiveEventProp")();
    
    py_archive_event_prop.attr("rel_change") = str(archive_event_prop.rel_change.in());
    py_archive_event_prop.attr("abs_change") = str(archive_event_prop.abs_change.in());
    py_archive_event_prop.attr("period") = str(archive_event_prop.period.in());
    py_archive_event_prop.attr("extensions") = 
        CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(archive_event_prop.extensions);
    
    return py_archive_event_prop;
}

object to_py(const Tango::EventProperties &event_props)
{
    PYTANGO_MOD
    object py_event_props = pytango.attr("EventProperties")();
    
    py_event_props.attr("ch_event") = to_py(event_props.ch_event);
    py_event_props.attr("per_event") = to_py(event_props.per_event);
    py_event_props.attr("arch_event") = to_py(event_props.arch_event);
    return py_event_props;
}

object to_py(const Tango::AttributeConfig &attr_conf, object py_attr_conf)
{
    if(py_attr_conf.ptr() == Py_None)
    {
        PYTANGO_MOD
        py_attr_conf = pytango.attr("AttributeConfig")();
    }
    
    py_attr_conf.attr("name") = str(attr_conf.name.in());
    py_attr_conf.attr("writable") = attr_conf.writable;
    py_attr_conf.attr("data_format") = attr_conf.data_format;
    py_attr_conf.attr("data_type") = attr_conf.data_type;
    py_attr_conf.attr("max_dim_x") = attr_conf.max_dim_x;
    py_attr_conf.attr("max_dim_y") = attr_conf.max_dim_y;
    py_attr_conf.attr("description") = str(attr_conf.description.in());
    py_attr_conf.attr("label") = str(attr_conf.label.in());
    py_attr_conf.attr("unit") = str(attr_conf.unit.in());
    py_attr_conf.attr("standard_unit") = str(attr_conf.standard_unit.in());
    py_attr_conf.attr("display_unit") = str(attr_conf.display_unit.in());
    py_attr_conf.attr("format") = str(attr_conf.format.in());
    py_attr_conf.attr("min_value") = str(attr_conf.min_value.in());
    py_attr_conf.attr("max_value") = str(attr_conf.max_value.in());
    py_attr_conf.attr("min_alarm") = str(attr_conf.min_alarm.in());
    py_attr_conf.attr("max_alarm") = str(attr_conf.max_alarm.in());
    py_attr_conf.attr("writable_attr_name") = str(attr_conf.writable_attr_name.in());
    py_attr_conf.attr("extensions") = 
        CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(attr_conf.extensions);
    
    return py_attr_conf;
}

object to_py(const Tango::AttributeConfig_2 &attr_conf, object py_attr_conf)
{
    if(py_attr_conf.ptr() == Py_None)
    {
        PYTANGO_MOD
        py_attr_conf = pytango.attr("AttributeConfig_2")();
    }
    
    py_attr_conf.attr("name") = str(attr_conf.name.in());
    py_attr_conf.attr("writable") = attr_conf.writable;
    py_attr_conf.attr("data_format") = attr_conf.data_format;
    py_attr_conf.attr("data_type") = attr_conf.data_type;
    py_attr_conf.attr("max_dim_x") = attr_conf.max_dim_x;
    py_attr_conf.attr("max_dim_y") = attr_conf.max_dim_y;
    py_attr_conf.attr("description") = str(attr_conf.description.in());
    py_attr_conf.attr("label") = str(attr_conf.label.in());
    py_attr_conf.attr("unit") = str(attr_conf.unit.in());
    py_attr_conf.attr("standard_unit") = str(attr_conf.standard_unit.in());
    py_attr_conf.attr("display_unit") = str(attr_conf.display_unit.in());
    py_attr_conf.attr("format") = str(attr_conf.format.in());
    py_attr_conf.attr("min_value") = str(attr_conf.min_value.in());
    py_attr_conf.attr("max_value") = str(attr_conf.max_value.in());
    py_attr_conf.attr("min_alarm") = str(attr_conf.min_alarm.in());
    py_attr_conf.attr("max_alarm") = str(attr_conf.max_alarm.in());
    py_attr_conf.attr("writable_attr_name") = str(attr_conf.writable_attr_name.in());
    py_attr_conf.attr("level") = attr_conf.level;
    py_attr_conf.attr("extensions") = CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(attr_conf.extensions);
    
    return py_attr_conf;
}

object to_py(const Tango::AttributeConfig_3 &attr_conf, object py_attr_conf)
{
    if(py_attr_conf.ptr() == Py_None)
    {
        PYTANGO_MOD
        py_attr_conf = pytango.attr("AttributeConfig_3")();
    }
    
    py_attr_conf.attr("name") = str(attr_conf.name.in());
    py_attr_conf.attr("writable") = attr_conf.writable;
    py_attr_conf.attr("data_format") = attr_conf.data_format;
    py_attr_conf.attr("data_type") = attr_conf.data_type;
    py_attr_conf.attr("max_dim_x") = attr_conf.max_dim_x;
    py_attr_conf.attr("max_dim_y") = attr_conf.max_dim_y;
    py_attr_conf.attr("description") = str(attr_conf.description.in());
    py_attr_conf.attr("label") = str(attr_conf.label.in());
    py_attr_conf.attr("unit") = str(attr_conf.unit.in());
    py_attr_conf.attr("standard_unit") = str(attr_conf.standard_unit.in());
    py_attr_conf.attr("display_unit") = str(attr_conf.display_unit.in());
    py_attr_conf.attr("format") = str(attr_conf.format.in());
    py_attr_conf.attr("min_value") = str(attr_conf.min_value.in());
    py_attr_conf.attr("max_value") = str(attr_conf.max_value.in());
    py_attr_conf.attr("writable_attr_name") = str(attr_conf.writable_attr_name.in());
    py_attr_conf.attr("level") = attr_conf.level;
    py_attr_conf.attr("att_alarm") = to_py(attr_conf.att_alarm);
    py_attr_conf.attr("event_prop") = to_py(attr_conf.event_prop);
    py_attr_conf.attr("extensions") = CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(attr_conf.extensions);
    py_attr_conf.attr("sys_extensions") = CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(attr_conf.sys_extensions);
    return py_attr_conf;
}

object to_py(const Tango::AttributeConfig_5 &attr_conf, object py_attr_conf)
{
    if(py_attr_conf.ptr() == Py_None)
    {
        PYTANGO_MOD
        py_attr_conf = pytango.attr("AttributeConfig_5")();
    }

    py_attr_conf.attr("name") = str(attr_conf.name.in());
    py_attr_conf.attr("writable") = attr_conf.writable;
    py_attr_conf.attr("data_format") = attr_conf.data_format;
    py_attr_conf.attr("data_type") = attr_conf.data_type;
    py_attr_conf.attr("memorized") = attr_conf.memorized;
    py_attr_conf.attr("mem_init") = attr_conf.mem_init;
    py_attr_conf.attr("max_dim_x") = attr_conf.max_dim_x;
    py_attr_conf.attr("max_dim_y") = attr_conf.max_dim_y;
    py_attr_conf.attr("description") = str(attr_conf.description.in());
    py_attr_conf.attr("label") = str(attr_conf.label.in());
    py_attr_conf.attr("unit") = str(attr_conf.unit.in());
    py_attr_conf.attr("standard_unit") = str(attr_conf.standard_unit.in());
    py_attr_conf.attr("display_unit") = str(attr_conf.display_unit.in());
    py_attr_conf.attr("format") = str(attr_conf.format.in());
    py_attr_conf.attr("min_value") = str(attr_conf.min_value.in());
    py_attr_conf.attr("max_value") = str(attr_conf.max_value.in());
    py_attr_conf.attr("writable_attr_name") = str(attr_conf.writable_attr_name.in());
    py_attr_conf.attr("level") = attr_conf.level;
    py_attr_conf.attr("root_attr_name") = str(attr_conf.root_attr_name.in());
    py_attr_conf.attr("enum_labels") = CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(attr_conf.enum_labels);
    py_attr_conf.attr("att_alarm") = to_py(attr_conf.att_alarm);
    py_attr_conf.attr("event_prop") = to_py(attr_conf.event_prop);
    py_attr_conf.attr("extensions") = CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(attr_conf.extensions);
    py_attr_conf.attr("sys_extensions") = CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(attr_conf.sys_extensions);
    return py_attr_conf;
}
boost::python::list to_py(const Tango::AttributeConfigList &attr_conf_list)
{
    boost::python::list py_attr_conf_list;
    boost::python::object none;
    for(unsigned long index = 0; index < attr_conf_list.length(); ++index)
    {
        const Tango::AttributeConfig &attr_conf = attr_conf_list[index];
        py_attr_conf_list.append(to_py(attr_conf, none));
    }
    return py_attr_conf_list;
}

boost::python::list to_py(const Tango::AttributeConfigList_2 &attr_conf_list)
{
    boost::python::list py_attr_conf_list;
    boost::python::object none;
    for(unsigned long index = 0; index < attr_conf_list.length(); ++index)
    {
        const Tango::AttributeConfig_2 &attr_conf = attr_conf_list[index];
        py_attr_conf_list.append(to_py(attr_conf, none));
    }
    return py_attr_conf_list;
}

boost::python::list to_py(const Tango::AttributeConfigList_3 &attr_conf_list)
{
    boost::python::list py_attr_conf_list;
    boost::python::object none;
    for(unsigned long index = 0; index < attr_conf_list.length(); ++index)
    {
        const Tango::AttributeConfig_3 &attr_conf = attr_conf_list[index];
        py_attr_conf_list.append(to_py(attr_conf, none));
    }
    return py_attr_conf_list;
}

boost::python::list to_py(const Tango::AttributeConfigList_5 &attr_conf_list)
{
    boost::python::list py_attr_conf_list;
    boost::python::object none;
    for(unsigned long index = 0; index < attr_conf_list.length(); ++index)
    {
        const Tango::AttributeConfig_5 &attr_conf = attr_conf_list[index];
        py_attr_conf_list.append(to_py(attr_conf, none));
    }
    return py_attr_conf_list;
}

object to_py(const Tango::PipeConfig &pipe_conf, object py_pipe_conf)
{
    if(py_pipe_conf.ptr() == Py_None)
    {
        PYTANGO_MOD
        py_pipe_conf = pytango.attr("PipeConfig")();
    }

    py_pipe_conf.attr("name") = str(pipe_conf.name.in());
    py_pipe_conf.attr("description") = str(pipe_conf.description.in());
    py_pipe_conf.attr("label") = str(pipe_conf.label.in());
    py_pipe_conf.attr("level") = pipe_conf.level;
    py_pipe_conf.attr("writable") = pipe_conf.writable;
    py_pipe_conf.attr("extensions") = CORBA_sequence_to_list<Tango::DevVarStringArray>::to_list(pipe_conf.extensions);
    return py_pipe_conf;
}

boost::python::list to_py(const Tango::PipeConfigList &pipe_conf_list)
{
    boost::python::list py_pipe_conf_list;
    boost::python::object none;
    for(unsigned long index = 0; index < pipe_conf_list.length(); ++index)
    {
        const Tango::PipeConfig &pipe_conf = pipe_conf_list[index];
        py_pipe_conf_list.append(to_py(pipe_conf, none));
    }
    return py_pipe_conf_list;
}
