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
#include <tango/tango.h>
#include "pytgutils.h"
#include "callback.h"
#include "device_attribute.h"
#include "exception.h"

using namespace boost::python;

struct PyCmdDoneEvent {
    object device;
    object cmd_name;
    object argout;
    object argout_raw;
    object err;
    object errors;
    object ext;
};

struct PyAttrReadEvent {
    object device;
    object attr_names;
    object argout;
    object err;
    object errors;
    object ext;
};

struct PyAttrWrittenEvent {
    object device;
    object attr_names;
    object err;
    object errors;
    object ext;
};


static void copy_most_fields(PyCallBackAutoDie* self, const Tango::CmdDoneEvent* ev, PyCmdDoneEvent* py_ev)
{
    // py_ev->device
    py_ev->cmd_name = object(ev->cmd_name);
    py_ev->argout_raw = object(ev->argout);
    py_ev->err = object(ev->err);
    py_ev->errors = object(ev->errors);
    // py_ev->ext = object(ev->ext);
}

static void copy_most_fields(PyCallBackAutoDie* self, const Tango::AttrReadEvent* ev, PyAttrReadEvent* py_ev)
{
    // py_ev->device
    py_ev->attr_names = object(ev->attr_names);

    PyDeviceAttribute::AutoDevAttrVector dev_attr_vec(ev->argout);
    py_ev->argout = PyDeviceAttribute::convert_to_python( \
                            dev_attr_vec, *ev->device, self->m_extract_as);

    py_ev->err = object(ev->err);
    py_ev->errors = object(ev->errors);
    // py_ev->ext = object(ev->ext);
}

static void copy_most_fields(PyCallBackAutoDie* self, const Tango::AttrWrittenEvent* ev, PyAttrWrittenEvent* py_ev)
{
    // py_ev->device
    py_ev->attr_names = object(ev->attr_names);
    py_ev->err = object(ev->err);
    py_ev->errors = object(ev->errors);
    // py_ev->ext = object(ev->ext);
}

/*static*/ object PyCallBackAutoDie::py_on_callback_parent_fades;
/*static*/ std::map<PyObject*, PyObject*> PyCallBackAutoDie::s_weak2ob;

PyCallBackAutoDie::~PyCallBackAutoDie()
{
    if (this->m_weak_parent) {
        PyCallBackAutoDie::s_weak2ob.erase(this->m_weak_parent);
        boost::python::xdecref(this->m_weak_parent);
    }
}



/*static*/ void PyCallBackAutoDie::init()
{
    object py_scope = boost::python::scope();

    def ("__on_callback_parent_fades", on_callback_parent_fades);
    PyCallBackAutoDie::py_on_callback_parent_fades = py_scope.attr("__on_callback_parent_fades");
}

void PyCallBackAutoDie::on_callback_parent_fades(PyObject* weakobj)
{
    PyObject* ob = PyCallBackAutoDie::s_weak2ob[weakobj];

    if (!ob)
        return;

//     while (ob->ob_refcnt)
    boost::python::xdecref(ob);
}

void PyCallBackAutoDie::set_autokill_references(object &py_self, object &py_parent)
{
    if (m_self == 0)
        m_self = py_self.ptr();

    assert(m_self == py_self.ptr());

    PyObject* recb = PyCallBackAutoDie::py_on_callback_parent_fades.ptr();
    this->m_weak_parent = PyWeakref_NewRef(py_parent.ptr(), recb);

    if (!this->m_weak_parent)
        throw_error_already_set();

    boost::python::incref(this->m_self);
    PyCallBackAutoDie::s_weak2ob[this->m_weak_parent] = py_self.ptr();
}

void PyCallBackAutoDie::unset_autokill_references()
{
    boost::python::decref(m_self);
}


template<typename OriginalT, typename CopyT>
static void _run_virtual_once(PyCallBackAutoDie* self, OriginalT * ev, const char* virt_fn_name)
{
    AutoPythonGIL gil;

    try {
        CopyT* py_ev = new CopyT();
        object py_value = object( handle<>(
                    to_python_indirect<
                        CopyT*,
                        detail::make_owning_holder>()(py_ev) ) );

        // - py_ev->device = object(ev->device); No, we use m_weak_parent
        // so we get exactly the same python object...
        if (self->m_weak_parent) {
            PyObject* parent = PyWeakref_GET_OBJECT(self->m_weak_parent);
            if (parent && parent != Py_None) {
                py_ev->device = object(handle<>(borrowed(parent)));
            }
        }

        copy_most_fields(self, ev, py_ev);

        self->get_override(virt_fn_name)(py_value);
    } catch (...) {
        self->unset_autokill_references();
        /// @todo yes, I want the exception to go to Tango and then wathever. But it will make a memory leak bcos tangoc++ is not handling exceptions properly!!  (proxy_asyn_cb.cpp, void Connection::Cb_ReadAttr_Request(CORBA::Request_ptr req,Tango::CallBack *cb_ptr))
        /// and the same for cmd_ended, attr_read & attr_written
        /// @bug See previous todo. If TangoC++ is fixed, it'll become a bug:
        delete ev;
        /// @todo or maybe it's just that I am not supposed to re-throw the exception? (still a bug in tangoc++). Then also get rid of the "delete ev" line!
        throw;
    }
    self->unset_autokill_references();
};

/*virtual*/ void PyCallBackAutoDie::cmd_ended(Tango::CmdDoneEvent * ev)
{
    _run_virtual_once<Tango::CmdDoneEvent, PyCmdDoneEvent>(this, ev, "cmd_ended");
};

/*virtual*/ void PyCallBackAutoDie::attr_read(Tango::AttrReadEvent *ev)
{
    _run_virtual_once<Tango::AttrReadEvent, PyAttrReadEvent>(this, ev, "attr_read");
};

/*virtual*/ void PyCallBackAutoDie::attr_written(Tango::AttrWrittenEvent *ev)
{
    _run_virtual_once<Tango::AttrWrittenEvent, PyAttrWrittenEvent>(this, ev, "attr_written");
};



PyCallBackPushEvent::~PyCallBackPushEvent()
{
    boost::python::xdecref(this->m_weak_device);
}

void PyCallBackPushEvent::set_device(object &py_device)
{
    this->m_weak_device = PyWeakref_NewRef(py_device.ptr(), 0);

    if (!this->m_weak_device)
        throw_error_already_set();
}


namespace {

    template<typename OriginalT>
    void copy_device(OriginalT* ev, object py_ev, object py_device)
    {
        if (py_device.ptr() != Py_None)
            py_ev.attr("device") = py_device;
        else
            py_ev.attr("device") = object(ev->device);
    }

    template<typename OriginalT>
    static void _push_event(PyCallBackPushEvent* self, OriginalT * ev)
    {
        // If the event is received after python dies but before the process
        // finishes then discard the event
        if (!Py_IsInitialized())
        {
            cout4 << "Tango event (" << ev->event << " for " 
                  << ev->attr_name << ") received for after python shutdown. "
                  << "Event will be ignored" << std::endl;
            return;
        }
        
        AutoPythonGIL gil;

        // Make a copy of ev in python
        // (the original will be deleted by TangoC++ on return)
        object py_ev(ev);
        OriginalT* ev_copy = extract<OriginalT*>(py_ev);

        // If possible, reuse the original DeviceProxy
        object py_device;
        if (self->m_weak_device) {
            PyObject* py_c_device = PyWeakref_GET_OBJECT(self->m_weak_device);
            if (py_c_device && py_c_device != Py_None)
                py_device = object(handle<>(borrowed(py_c_device)));
        }

        try
        {
            PyCallBackPushEvent::fill_py_event(ev_copy, py_ev, py_device, self->m_extract_as);
        }
        SAFE_CATCH_REPORT("PyCallBackPushEvent::fill_py_event")

        try
        {
            self->get_override("push_event")(py_ev);
        }
        SAFE_CATCH_INFORM("push_event")
    };
}



void PyCallBackPushEvent::fill_py_event(Tango::EventData* ev, object & py_ev, object py_device, PyTango::ExtractAs extract_as)
{
    copy_device(ev, py_ev, py_device);
    /// @todo on error extracting, we could save the error in DeviceData
    /// instead of throwing it...?
    // Save a copy of attr_value, so we can still access it after
    // the execution of the callback (Tango will delete the original!)
    // I originally was 'stealing' the reference to TangoC++: I got
    // attr_value and set it to 0... But now TangoC++ is not deleting
    // attr_value pointer but its own copy, so my efforts are useless.
    if (ev->attr_value)
    {
        Tango::DeviceAttribute *attr = new Tango::DeviceAttribute(*ev->attr_value);
        py_ev.attr("attr_value") = PyDeviceAttribute::convert_to_python(attr, *ev->device, extract_as);
    }
    // ev->attr_value = 0; // Do not delete, python will.
}


void PyCallBackPushEvent::fill_py_event(Tango::AttrConfEventData* ev, object & py_ev, object py_device, PyTango::ExtractAs extract_as)
{
    copy_device(ev, py_ev, py_device);

    if (ev->attr_conf) {
        py_ev.attr("attr_conf") = *ev->attr_conf;
    }
}

void PyCallBackPushEvent::fill_py_event(Tango::DataReadyEventData* ev, object & py_ev, object py_device, PyTango::ExtractAs extract_as)
{
    copy_device(ev, py_ev, py_device);
}



/*virtual*/ void PyCallBackPushEvent::push_event(Tango::EventData *ev)
{
    _push_event(this, ev);
}

/*virtual*/ void PyCallBackPushEvent::push_event(Tango::AttrConfEventData *ev)
{
    _push_event(this, ev);
}

/*virtual*/ void PyCallBackPushEvent::push_event(Tango::DataReadyEventData *ev)
{
    _push_event(this, ev);
}

void export_callback()
{
    PyCallBackAutoDie::init();

    /// @todo move somewhere else, another file i tal...

    class_<PyCmdDoneEvent> CmdDoneEvent("CmdDoneEvent", no_init);
    CmdDoneEvent
            .def_readonly("device", &PyCmdDoneEvent::device)
            .def_readonly("cmd_name", &PyCmdDoneEvent::cmd_name)
            .def_readonly("argout_raw", &PyCmdDoneEvent::argout_raw)
            .def_readonly("argout", &PyCmdDoneEvent::argout)
            .def_readonly("err", &PyCmdDoneEvent::err)
            .def_readonly("errors", &PyCmdDoneEvent::errors)
            .def_readonly("ext", &PyCmdDoneEvent::ext)
    ;

    class_<PyAttrReadEvent> AttrReadEvent("AttrReadEvent", no_init);
    AttrReadEvent
            .def_readonly("device", &PyAttrReadEvent::device)
            .def_readonly("attr_names", &PyAttrReadEvent::attr_names)
            .def_readonly("argout", &PyAttrReadEvent::argout)
            .def_readonly("err", &PyAttrReadEvent::err)
            .def_readonly("errors", &PyAttrReadEvent::errors)
            .def_readonly("ext", &PyAttrReadEvent::ext)
    ;

    class_<PyAttrWrittenEvent> AttrWrittenEvent( "AttrWrittenEvent", no_init);
    AttrWrittenEvent
            .def_readonly("device", &PyAttrWrittenEvent::device)
            .def_readonly("attr_names", &PyAttrWrittenEvent::attr_names)
            .def_readonly("err", &PyAttrWrittenEvent::err)
            .def_readonly("errors", &PyAttrWrittenEvent::errors)
            .def_readonly("ext", &PyAttrWrittenEvent::ext)
    ;

    class_<PyCallBackAutoDie, boost::noncopyable> CallBackAutoDie(
        "__CallBackAutoDie",
        "INTERNAL CLASS - DO NOT USE IT",
        init<>())
    ;

    CallBackAutoDie
        .def("cmd_ended", &PyCallBackAutoDie::cmd_ended,
            "This method is defined as being empty and must be overloaded by the user when the asynchronous callback model is used. This is the method which will be executed when the server reply from a command_inout is received in both push and pull sub-mode.")
        .def("attr_read", &PyCallBackAutoDie::attr_read,
            "This method is defined as being empty and must be overloaded by the user when the asynchronous callback model is used. This is the method which will be executed when the server reply from a read_attribute(s) is received in both push and pull sub-mode.")
        .def("attr_written", &PyCallBackAutoDie::attr_written,
            "This method is defined as being empty and must be overloaded by the user when the asynchronous callback model is used. This is the method which will be executed when the server reply from a write_attribute(s) is received in both push and pull sub-mode. ")
    ;

    class_<PyCallBackPushEvent, boost::noncopyable> CallBackPushEvent(
        "__CallBackPushEvent",
        "INTERNAL CLASS - DO NOT USE IT",
        init<>())
    ;

    CallBackPushEvent
        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::EventData*))&PyCallBackAutoDie::push_event,
            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send event(s) to the client. ")
        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::AttrConfEventData*))&PyCallBackAutoDie::push_event,
            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send attribute configuration change event(s) to the client. ")
        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::DataReadyEventData*))&PyCallBackAutoDie::push_event,
            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send attribute data ready event(s) to the client. ")
    ;
}
