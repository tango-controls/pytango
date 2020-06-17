/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include <map>
#include "defs.h"

/// Tango expects an object for callbacks derived from Tango::CallBack.
/// For read_attribute_asynch, write_attribute_asynch, the callback object
/// should be available until the callback is run. Then it can disappear.
/// Also if we forget about a DeviceProxy we don't need the callback anymore.
/// For event subscription however, the requirements are different. The C++
/// callback can be called way after the original DeviceProxy has disappered.
/// So for this case, the callback should live forever. As we don't want it,
/// we implemented the deletion of the callback in the DeviceProxy destructor
/// itself, after performing an unsubscribe.
/// @todo this is for cmd_ended, attr_read and attr_written. push_event are not done!
class PyCallBackAutoDie : public Tango::CallBack , public boost::python::wrapper<Tango::CallBack>
{
public:
    PyCallBackAutoDie() : m_self(0), m_weak_parent(0), m_extract_as(PyTango::ExtractAsNumpy) {}
    virtual ~PyCallBackAutoDie();

    //! It is the PyCallBackAutoDie object itself, as seen from python
    PyObject* m_self;
    //! The object that will call this callback, so we can
    //! monitor if it disappears, we are not needed anymore.
    PyObject* m_weak_parent;

    PyTango::ExtractAs m_extract_as;

    static std::map<PyObject*, PyObject*> s_weak2ob;
    static boost::python::object py_on_callback_parent_fades;

    static void on_callback_parent_fades(PyObject* weakobj);
    static void init();

    void set_autokill_references(boost::python::object &py_self, boost::python::object &py_parent);
    void unset_autokill_references();

    void set_extract_as(PyTango::ExtractAs extract_as)
    {   this->m_extract_as = extract_as; }

    boost::python::object get_override(const char* name)
    { return boost::python::wrapper<Tango::CallBack>::get_override(name); }
    
    virtual void cmd_ended(Tango::CmdDoneEvent * ev);
    virtual void attr_read(Tango::AttrReadEvent *ev);
    virtual void attr_written(Tango::AttrWrittenEvent *ev);
//     virtual void push_event(Tango::EventData *ev);
//     virtual void push_event(Tango::AttrConfEventData *ev);
//     virtual void push_event(Tango::DataReadyEventData *ev);
};


class PyCallBackPushEvent : public Tango::CallBack , public boost::python::wrapper<Tango::CallBack>
{
public:
    PyCallBackPushEvent() : m_weak_device(0), m_extract_as(PyTango::ExtractAsNumpy)
    {}
    virtual ~PyCallBackPushEvent();

    //! The object that will call this callback (DeviceProxy), so we can
    //! monitor if it disappears, we are not needed anymore.
    PyObject* m_weak_device;
    PyTango::ExtractAs m_extract_as;

    void set_device(boost::python::object &py_device);

    void set_extract_as(PyTango::ExtractAs extract_as)
    {   this->m_extract_as = extract_as; }

    boost::python::object get_override(const char* name);
    
//     virtual void cmd_ended(Tango::CmdDoneEvent * ev);
//     virtual void attr_read(Tango::AttrReadEvent *ev);
//     virtual void attr_written(Tango::AttrWrittenEvent *ev);
    virtual void push_event(Tango::EventData *ev);
    virtual void push_event(Tango::AttrConfEventData *ev);
    virtual void push_event(Tango::DataReadyEventData *ev);
    virtual void push_event(Tango::PipeEventData *ev);
    virtual void push_event(Tango::DevIntrChangeEventData *ev);

    static void fill_py_event(Tango::EventData* ev, boost::python::object & py_ev, boost::python::object py_device, PyTango::ExtractAs extract_as);
    static void fill_py_event(Tango::AttrConfEventData* ev, boost::python::object & py_ev, boost::python::object py_device, PyTango::ExtractAs extract_as);
    static void fill_py_event(Tango::DataReadyEventData* ev, boost::python::object & py_ev, boost::python::object py_device, PyTango::ExtractAs extract_as);
    static void fill_py_event(Tango::PipeEventData* ev, boost::python::object & py_ev, boost::python::object py_device, PyTango::ExtractAs extract_as);
    static void fill_py_event(Tango::DevIntrChangeEventData* ev, boost::python::object & py_ev, boost::python::object py_device, PyTango::ExtractAs extract_as);
};
