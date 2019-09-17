/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include <map>
#include "defs.h"

namespace py = pybind11;

// Tango expects an object for callbacks derived from Tango::CallBack.
// For read_attribute_asynch, write_attribute_asynch, the callback object
// should be available until the callback is run. Then it can disappear.
// Also if we forget about a DeviceProxy we don't need the callback anymore.
// For event subscription however, the requirements are different. The C++
// callback can be called way after the original DeviceProxy has disappeared.
// So for this case, the callback should live forever. As we don't want it,
// we implemented the deletion of the callback in the DeviceProxy destructor
// itself, after performing an unsubscribe.

class __attribute__ ((visibility("hidden"))) CallBackAutoDie : public Tango::CallBack
{
public:
    CallBackAutoDie();
    virtual ~CallBackAutoDie();


//    std::unique_ptr<PyObject> m_cbk;

    //    void set_extract_as();
    void set_callback(py::object callback);
    void set_weak_parent(py::object parent);

    //! It is the CallBackAutoDie object itself, as seen from python
//    PyObject* m_self;
    //! The object that will call this callback, so we can
    //! monitor if it disappears, we are not needed anymore.
    PyObject* m_weak_parent;

    py::object m_callback;

    static std::map<int, CallBackAutoDie*> s_weak2ob;
    static py::object py_on_callback_parent_fades;
    static void on_callback_parent_fades(PyObject* weakobj);
    static void init();

    void set_autokill_references(CallBackAutoDie* cb, Tango::Connection& py_parent);
//    void unset_autokill_references();

    virtual void cmd_ended(Tango::CmdDoneEvent* ev);
    virtual void attr_read(Tango::AttrReadEvent* ev);
    virtual void attr_written(Tango::AttrWrittenEvent* ev);
};

class __attribute__ ((visibility("hidden"))) CallBackPushEvent : public Tango::CallBack
{
public:
    CallBackPushEvent();
    virtual ~CallBackPushEvent();

    void set_device(Tango::DeviceProxy& dp);

    virtual void push_event(Tango::EventData* ev);
    virtual void push_event(Tango::AttrConfEventData* ev);
    virtual void push_event(Tango::DataReadyEventData* ev);
    virtual void push_event(Tango::PipeEventData* ev);
    virtual void push_event(Tango::DevIntrChangeEventData* ev);

    static void fill_py_event(Tango::EventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device);
    static void fill_py_event(Tango::AttrConfEventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device);
    static void fill_py_event(Tango::DataReadyEventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device);
    static void fill_py_event(Tango::PipeEventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device);
    static void fill_py_event(Tango::DevIntrChangeEventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device);

    Tango::DeviceProxy m_device;
    py::object m_callback;
};

