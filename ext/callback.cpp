/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>
#include <callback.h>

namespace py = pybind11;

//struct __attribute__ ((visibility("hidden"))) PyCmdDoneEvent {
//    py::object device;
//    py::object cmd_name;
//    py::object argout;
//    py::object argout_raw;
//    py::object err;
//    py::object errors;
//    py::object ext;
//};
//
//struct __attribute__ ((visibility("hidden"))) PyAttrReadEvent {
//    py::object device;
//    py::object attr_names;
//    py::object argout;
//    py::object err;
//    py::object errors;
//    py::object ext;
//};
//
//struct __attribute__ ((visibility("hidden"))) PyAttrWrittenEvent {
//    py::object device;
//    py::object attr_names;
//    py::object err;
//    py::object errors;
//    py::object ext;
//};
//

//static void copy_most_fields(PyCallBackAutoDie* self, const Tango::CmdDoneEvent* ev, PyCmdDoneEvent* py_ev)
//{
//    // py_ev->device
//    py_ev->cmd_name = py::object(ev->cmd_name);
//    py_ev->argout_raw = py::object(ev->argout);
//    py_ev->err = py::object(ev->err);
//    py_ev->errors = py::object(ev->errors);
//    // py_ev->ext = py::object(ev->ext);
//}
//
//static void copy_most_fields(PyCallBackAutoDie* self, const Tango::AttrReadEvent* ev, PyAttrReadEvent* py_ev)
//{
//    // py_ev->device
//    py_ev->attr_names = py::object(ev->attr_names);
//
//    PyDeviceAttribute::AutoDevAttrVector dev_attr_vec(ev->argout);
//    py_ev->argout = PyDeviceAttribute::convert_to_python(dev_attr_vec, *ev->device, self->m_extract_as);
//
//    py_ev->err = py::object(ev->err);
//    py_ev->errors = py::object(ev->errors);
//    // py_ev->ext = py::object(ev->ext);
//}
//
//static void copy_most_fields(PyCallBackAutoDie* self, const Tango::AttrWrittenEvent* ev, PyAttrWrittenEvent* py_ev)
//{
//    // py_ev->device
//    py_ev->attr_names = py::object(ev->attr_names);
//    py_ev->err = py::object(ev->err);
//    py_ev->errors = py::object(ev->errors);
//    // py_ev->ext = py::object(ev->ext);
//}
//
///*static*/ py::object PyCallBackAutoDie::py_on_callback_parent_fades;
///*static*/ std::map<PyObject*, PyObject*> PyCallBackAutoDie::s_weak2ob;
//
//PyCallBackAutoDie::~PyCallBackAutoDie()
//{
//    if (this->m_weak_parent) {
//        PyCallBackAutoDie::s_weak2ob.erase(this->m_weak_parent);
//        py::xdecref(this->m_weak_parent);
//    }
//}



///*static*/ void PyCallBackAutoDie::init()
//{
//    py::object py_scope = py::scope();
//
//    def ("__on_callback_parent_fades", on_callback_parent_fades);
//    PyCallBackAutoDie::py_on_callback_parent_fades = py_scope.attr("__on_callback_parent_fades");
//}

//void PyCallBackAutoDie::on_callback_parent_fades(PyObject* weakobj)
//{
//    PyObject* ob = PyCallBackAutoDie::s_weak2ob[weakobj];
//
//    if (!ob)
//        return;
//
////     while (ob->ob_refcnt)
//    py::xdecref(ob);
//}
//
//void PyCallBackAutoDie::set_autokill_references(py::object &py_self, py::object &py_parent)
//{
//    if (m_self == 0)
//        m_self = py_self.ptr();
//
//    assert(m_self == py_self.ptr());
//
//    PyObject* recb = PyCallBackAutoDie::py_on_callback_parent_fades.ptr();
//    this->m_weak_parent = PyWeakref_NewRef(py_parent.ptr(), recb);
//
//    if (!this->m_weak_parent)
//        throw_error_already_set();
//
//    py::incref(this->m_self);
//    PyCallBackAutoDie::s_weak2ob[this->m_weak_parent] = py_self.ptr();
//}
//
//void PyCallBackAutoDie::unset_autokill_references()
//{
//    py::decref(m_self);
//}
//
//
//template<typename OriginalT, typename CopyT>
//static void _run_virtual_once(PyCallBackAutoDie* self, OriginalT * ev, const char* virt_fn_name)
//{
//    AutoPythonGIL gil;
//
//    try {
//        CopyT* py_ev = new CopyT();
//        py::object py_value = py::object( handle<>(
//                    to_python_indirect<
//                        CopyT*,
//                        detail::make_owning_holder>()(py_ev) ) );
//
//        // - py_ev->device = py::object(ev->device); No, we use m_weak_parent
//        // so we get exactly the same python py::object...
//        if (self->m_weak_parent) {
//            PyObject* parent = PyWeakref_GET_OBJECT(self->m_weak_parent);
//            if (parent && parent != Py_None) {
//                py_ev->device = py::object(handle<>(borrowed(parent)));
//            }
//        }
//
//        copy_most_fields(self, ev, py_ev);
//
//        self->get_override(virt_fn_name)(py_value);
//    } catch (...) {
//        self->unset_autokill_references();
//        /// @todo yes, I want the exception to go to Tango and then wathever. But it will make a memory leak bcos tangoc++ is not handling exceptions properly!!  (proxy_asyn_cb.cpp, void Connection::Cb_ReadAttr_Request(CORBA::Request_ptr req,Tango::CallBack *cb_ptr))
//        /// and the same for cmd_ended, attr_read & attr_written
//        /// @bug See previous todo. If TangoC++ is fixed, it'll become a bug:
//        delete ev;
//        /// @todo or maybe it's just that I am not supposed to re-throw the exception? (still a bug in tangoc++). Then also get rid of the "delete ev" line!
//        throw;
//    }
//    self->unset_autokill_references();
//};
//
///*virtual*/ void PyCallBackAutoDie::cmd_ended(Tango::CmdDoneEvent * ev)
//{
//    _run_virtual_once<Tango::CmdDoneEvent, PyCmdDoneEvent>(this, ev, "cmd_ended");
//};
//
///*virtual*/ void PyCallBackAutoDie::attr_read(Tango::AttrReadEvent *ev)
//{
//    _run_virtual_once<Tango::AttrReadEvent, PyAttrReadEvent>(this, ev, "attr_read");
//};
//
///*virtual*/ void PyCallBackAutoDie::attr_written(Tango::AttrWrittenEvent *ev)
//{
//    _run_virtual_once<Tango::AttrWrittenEvent, PyAttrWrittenEvent>(this, ev, "attr_written");
//};



PyCallBackPushEvent::~PyCallBackPushEvent()
{
//    Py_XDECREF(this->m_weak_device);
}

//void PyCallBackPushEvent::set_device(py::object &py_device)
//{
//    this->m_weak_device = PyWeakref_NewRef(py_device.ptr(), 0);
//
//    if (!this->m_weak_device)
//        throw_error_already_set();
//}


//namespace {
//
//    template<typename OriginalT>
//    void copy_device(OriginalT* ev, py::object py_ev, py::object py_device)
//    {
//        if (py_device.ptr() != Py_None)
//            py_ev.attr("device") = py_device;
//        else
//            py_ev.attr("device") = py::object(ev->device);
//    }
//
//    template<typename OriginalT>
//    static void _push_event(PyCallBackPushEvent* self, OriginalT * ev)
//    {
//    	// If the event is received after python dies but before the process
//        // finishes then discard the event
//        if (!Py_IsInitialized())
//        {
//            cout4 << "Tango event (" << ev->event << " for "
////                  << ev->attr_name << ") received for after python shutdown. "
//                  << "Event will be ignored" << std::endl;
//            return;
//        }
//
//        AutoPythonGIL gil;
//
//        // Make a copy of ev in python
//        // (the original will be deleted by TangoC++ on return)
//        py::object py_ev(ev);
//        OriginalT* ev_copy = extract<OriginalT*>(py_ev);
//
//        // If possible, reuse the original DeviceProxy
//        py::object py_device;
//        if (self->m_weak_device) {
//            PyObject* py_c_device = PyWeakref_GET_OBJECT(self->m_weak_device);
//            if (py_c_device && py_c_device != Py_None) {
//               py_device = py::object(handle<>(borrowed(py_c_device)));
//            }
//        }
//
//        try
//        {
//            PyCallBackPushEvent::fill_py_event(ev_copy, py_ev, py_device, self->m_extract_as);
//        }
//        SAFE_CATCH_REPORT("PyCallBackPushEvent::fill_py_event")
//
//        try
//        {
//            self->get_override("push_event")(py_ev);
//        }
//        SAFE_CATCH_INFORM("push_event")
//    };
//}
//
//
//py::object PyCallBackPushEvent::get_override(const char* name)
//{
//    return py::wrapper<Tango::CallBack>::get_override(name);
//}
//
//
//void PyCallBackPushEvent::fill_py_event(Tango::EventData* ev, py::object & py_ev, py::object py_device, PyTango::ExtractAs extract_as)
//{
//    copy_device(ev, py_ev, py_device);
//    /// @todo on error extracting, we could save the error in DeviceData
//    /// instead of throwing it...?
//    // Save a copy of attr_value, so we can still access it after
//    // the execution of the callback (Tango will delete the original!)
//    // I originally was 'stealing' the reference to TangoC++: I got
//    // attr_value and set it to 0... But now TangoC++ is not deleting
//    // attr_value pointer but its own copy, so my efforts are useless.
//    if (ev->attr_value)
//    {
//#ifdef PYTANGO_HAS_UNIQUE_PTR
//        Tango::DeviceAttribute *attr = new Tango::DeviceAttribute;
//	(*attr) = std::move(*ev->attr_value);
//#else
//	Tango::DeviceAttribute *attr = new Tango::DeviceAttribute(*ev->attr_value);
//#endif
//        py_ev.attr("attr_value") = PyDeviceAttribute::convert_to_python(attr, *ev->device, extract_as);
//    }
//    // ev->attr_value = 0; // Do not delete, python will.
//}
//
//
//void PyCallBackPushEvent::fill_py_event(Tango::AttrConfEventData* ev, object & py_ev, object py_device, PyTango::ExtractAs extract_as)
//{
//    copy_device(ev, py_ev, py_device);
//
//    if (ev->attr_conf) {
//        py_ev.attr("attr_conf") = *ev->attr_conf;
//    }
//}
//
//void PyCallBackPushEvent::fill_py_event(Tango::DataReadyEventData* ev, object & py_ev, object py_device, PyTango::ExtractAs extract_as)
//{
//    copy_device(ev, py_ev, py_device);
//}
//
//void PyCallBackPushEvent::fill_py_event(Tango::PipeEventData* ev, object & py_ev, object py_device, PyTango::ExtractAs extract_as)
//{
//    copy_device(ev, py_ev, py_device);
//    if (ev->pipe_value) {
//#ifdef PYTANGO_HAS_UNIQUE_PTR
//        Tango::DevicePipe *pipe_value = new Tango::DevicePipe;
//        (*pipe_value) = std::move(*ev->pipe_value);
//#else
//        Tango::DevicePipe *pipe_value = new Tango::DevicePipe(*ev->pipe_value);
//#endif
//        py_ev.attr("pipe_value") = PyTango::DevicePipe::convert_to_python(pipe_value, extract_as);
//    }
//}
//
///*virtual*/ void PyCallBackPushEvent::push_event(Tango::EventData *ev)
//{
//    _push_event(this, ev);
//}
//
///*virtual*/ void PyCallBackPushEvent::push_event(Tango::AttrConfEventData *ev)
//{
//    _push_event(this, ev);
//}
//
///*virtual*/ void PyCallBackPushEvent::push_event(Tango::DataReadyEventData *ev)
//{
//    _push_event(this, ev);
//}
//
///*virtual*/ void PyCallBackPushEvent::push_event(Tango::PipeEventData *ev)
//{
//    _push_event(this, ev);
//}

void export_callback(py::module &m)
{
//    PyCallBackAutoDie::init();
//
//    /// @todo move somewhere else, another file i tal...
//
//    py::class_<PyCmdDoneEvent>(m, "CmdDoneEvent")
//        .def_readonly("device", &PyCmdDoneEvent::device)
//        .def_readonly("cmd_name", &PyCmdDoneEvent::cmd_name)
//        .def_readonly("argout_raw", &PyCmdDoneEvent::argout_raw)
//        .def_readonly("err", &PyCmdDoneEvent::err)
//        .def_readonly("errors", &PyCmdDoneEvent::errors)
//        .def_readonly("ext", &PyCmdDoneEvent::ext)
//        .def_readwrite("argout", &PyCmdDoneEvent::argout)
//    ;
//
//    py::class_<PyAttrReadEvent>(m, "AttrReadEvent")
//        .def_readonly("device", &PyAttrReadEvent::device)
//        .def_readonly("attr_names", &PyAttrReadEvent::attr_names)
//        .def_readonly("argout", &PyAttrReadEvent::argout)
//        .def_readonly("err", &PyAttrReadEvent::err)
//        .def_readonly("errors", &PyAttrReadEvent::errors)
//        .def_readonly("ext", &PyAttrReadEvent::ext)
//    ;
//
//    py::class_<PyAttrWrittenEvent>(m, "AttrWrittenEvent")
//        .def_readonly("device", &PyAttrWrittenEvent::device)
//        .def_readonly("attr_names", &PyAttrWrittenEvent::attr_names)
//        .def_readonly("err", &PyAttrWrittenEvent::err)
//        .def_readonly("errors", &PyAttrWrittenEvent::errors)
//        .def_readonly("ext", &PyAttrWrittenEvent::ext)
//    ;
//
////      boost::noncopyable
//        py::class_<PyCallBackAutoDie>(m, "__CallBackAutoDie",
//                "INTERNAL CLASS - DO NOT USE IT")
//        .def(py::init<>())
//        .def("cmd_ended", &PyCallBackAutoDie::cmd_ended,
//            "This method is defined as being empty and must be overloaded by the user when the asynchronous callback model is used. This is the method which will be executed when the server reply from a command_inout is received in both push and pull sub-mode.")
//        .def("attr_read", &PyCallBackAutoDie::attr_read,
//            "This method is defined as being empty and must be overloaded by the user when the asynchronous callback model is used. This is the method which will be executed when the server reply from a read_attribute(s) is received in both push and pull sub-mode.")
//        .def("attr_written", &PyCallBackAutoDie::attr_written,
//            "This method is defined as being empty and must be overloaded by the user when the asynchronous callback model is used. This is the method which will be executed when the server reply from a write_attribute(s) is received in both push and pull sub-mode. ")
//    ;
//        boost::noncopyable
    py::class_<PyCallBackPushEvent>(m,"CallBackPushEvent", "INTERNAL CLASS - DO NOT USE IT")
        .def(py::init<>())
        .def("push_event", [](Tango::PipeEventData& ev) {
//            _push_event(this, ev);
            py::print(ev);
        })
//        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::EventData*))&PyCallBackAutoDie::push_event,
//            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send event(s) to the client. ")
//        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::AttrConfEventData*))&PyCallBackAutoDie::push_event,
//            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send attribute configuration change event(s) to the client. ")
//        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::DataReadyEventData*))&PyCallBackAutoDie::push_event,
//            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send attribute data ready event(s) to the client. ")
//        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::PipeEventData*))&PyCallBackAutoDie::push_event,
//            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send pipe event(s) to the client. ")
    ;
}
