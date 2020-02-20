/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <memory>
#include <tango.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <callback.h>
#include <pytgutils.h>
#include <exception.h>
#include <device_attribute.h>
#include <device_pipe.h>

namespace py = pybind11;

//struct __attribute__ ((visibility("hidden"))) PyCmdDoneEvent {
class PyCmdDoneEvent {
public:
    py::object device;
    py::object cmd_name;
    py::object argout;
    py::object argout_raw;
    py::object err;
    py::object errors;
    py::object ext;
};

//struct __attribute__ ((visibility("hidden"))) PyAttrReadEvent {
class PyAttrReadEvent {
public:
    py::object device;
    py::object attr_names;
    py::object argout;
    py::object err;
    py::object errors;
    py::object ext;
};

//struct __attribute__ ((visibility("hidden"))) PyAttrWrittenEvent {
class PyAttrWrittenEvent {
public:
    py::object device;
    py::object attr_names;
    py::object err;
    py::object errors;
    py::object ext;
};

static void copy_most_fields(PyCallBackAutoDie& self, Tango::CmdDoneEvent* ev, PyCmdDoneEvent* py_ev)
{
    // py_ev->device
    py_ev->cmd_name = py::cast(ev->cmd_name);
    py_ev->argout_raw = py::cast(ev->argout);
    py_ev->err = py::cast(ev->err);
    py_ev->errors = py::cast(ev->errors);
    // py_ev->ext = py::object(ev->ext);
}

static void copy_most_fields(PyCallBackAutoDie& self, Tango::AttrReadEvent* ev, PyAttrReadEvent* py_ev)
{
    // py_ev->device
    py_ev->attr_names = py::cast(ev->attr_names);

//    PyDeviceAttribute::AutoDevAttrVector dev_attr_vec(ev->argout);
//    py_ev->argout = PyDeviceAttribute::convert_to_python(dev_attr_vec, *ev->device);

    py_ev->err = py::cast(ev->err);
    py_ev->errors = py::cast(ev->errors);
    // py_ev->ext = py::object(ev->ext);
}

static void copy_most_fields(PyCallBackAutoDie& self, const Tango::AttrWrittenEvent* ev, PyAttrWrittenEvent* py_ev)
{
    // py_ev->device
    py_ev->attr_names = py::cast(ev->attr_names);
    py_ev->err = py::cast(ev->err);
    py_ev->errors = py::cast(ev->errors);
    // py_ev->ext = py::object(ev->ext);
}

/*static*/ py::object PyCallBackAutoDie::py_on_callback_parent_fades;
/*static*/ std::map<py::object, py::object> PyCallBackAutoDie::s_weak2ob;

PyCallBackAutoDie::~PyCallBackAutoDie() {
    cerr << "PyCallBackAutoDie destructor" << endl;
    if (this->m_weak_parent) {
        PyCallBackAutoDie::s_weak2ob.erase(this->m_weak_parent);
//        boost::python::xdecref(this->m_weak_parent);
    }
}

/*static*/ void PyCallBackAutoDie::init()
{
//    py::object py_scope = py::scope();

//    def ("__on_callback_parent_fades", on_callback_parent_fades);
//    PyCallBackAutoDie::py_on_callback_parent_fades = py_scope.attr("__on_callback_parent_fades");
}

void PyCallBackAutoDie::on_callback_parent_fades(py::object& weakobj)
{
    py::object ob = PyCallBackAutoDie::s_weak2ob[weakobj];

    if (!ob)
        return;

//     while (ob->ob_refcnt)
//    boost::python::xdecref(ob);
}

void PyCallBackAutoDie::set_autokill_references(py::object& py_self, py::object& py_parent)
{
    if (m_self.is(py::none()))
        m_self = py_self;

    assert(m_self.is(py_self));

    py::object recb = PyCallBackAutoDie::py_on_callback_parent_fades;
    this->m_weak_parent = py::reinterpret_borrow<py::object>(PyWeakref_NewRef(py_parent.ptr(), recb.ptr()));

    if (!this->m_weak_parent)
        throw py::error_already_set();

//    boost::python::incref(this->m_self);
    PyCallBackAutoDie::s_weak2ob[this->m_weak_parent] = py_self;
}

void PyCallBackAutoDie::unset_autokill_references()
{
//    boost::python::decref(m_self);
}

template<typename OriginalT, typename CopyT>
static void _run_virtual_once(PyCallBackAutoDie* self, OriginalT * ev, const char* virt_fn_name)
{
    AutoPythonGILEnsure gil;

//    try {
//        CopyT* py_ev = new CopyT();
//       py::object py_value =py::object( handle<>(
//                    to_python_indirect<
//                        CopyT*,
//                        detail::make_owning_holder>()(py_ev) ) );
//
//        // - py_ev->device =py::object(ev->device); No, we use m_weak_parent
//        // so we get exactly the same python object...
//        if (self->m_weak_parent) {
//            PyObject* parent = PyWeakref_GET_OBJECT(self->m_weak_parent);
//            if (parent && parent != Py_None) {
//                py_ev->device =py::object(handle<>(borrowed(parent)));
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
};

void PyCallBackAutoDie::cmd_ended(Tango::CmdDoneEvent* ev)
{
    cerr << "cmd_ended run_virtual_once" << endl;
//    _run_virtual_once<Tango::CmdDoneEvent, PyCmdDoneEvent>(this, ev, "cmd_ended");
};


/*virtual*/
void PyCallBackAutoDie::attr_read(Tango::AttrReadEvent* ev)
{
    py::print("attr read _run_virtual_once");
//    _run_virtual_once<Tango::AttrReadEvent, PyAttrReadEvent>(this, ev, "attr_read");
};

/*virtual*/
void PyCallBackAutoDie::attr_written(Tango::AttrWrittenEvent* ev)
{
    py::print("attr written _run_virtual_once");
//    _run_virtual_once<Tango::AttrWrittenEvent, PyAttrWrittenEvent>(this, ev, "attr_written");
};

PyCallBackPushEvent::~PyCallBackPushEvent()
{
//    boost::python::xdecref(this->m_weak_device);
}

void PyCallBackPushEvent::set_device(Tango::DeviceProxy& dp)
{
    py::object py_device = py::cast(dp);
    this->m_weak_device = py::reinterpret_borrow<py::object>(PyWeakref_NewRef(py_device.ptr(), py::none().ptr()));

    if (!this->m_weak_device)
        throw py::error_already_set();
}

namespace {

    template<typename OriginalT>
    void copy_device(OriginalT* ev, py::object& py_ev, Tango::DeviceProxy& dp)
    {
        py::object py_device = py::cast(dp);
        if (!py_device.is(py::none()))
            py_ev.attr("device") = py_device;
        else
            py_ev.attr("device") = py::cast(ev->device);
    }

    template<typename OriginalT>
    static void _push_event(PyCallBackPushEvent* self, OriginalT* ev)
    {
    	// If the event is received after python dies but before the process
        // finishes then discard the event
        if (!Py_IsInitialized())
        {
            cout4 << "Tango event (" << ev->event <<
                  ") received for after python shutdown. "
                  << "Event will be ignored" << std::endl;
            return;
        }

        AutoPythonGILEnsure gil;

        // Make a copy of ev in python
        // (the original will be deleted by TangoC++ on return)
        py::object py_ev(ev);
        OriginalT* ev_copy = py_ev.cast<OriginalT*>();

        // If possible, reuse the original DeviceProxy
        py::object py_device;
        if (self->m_weak_device) {
            py_device = py::reinterpret_borrow<py::object>(PyWeakref_GET_OBJECT(self->m_weak_device.ptr()));
        }

        try
        {
            PyCallBackPushEvent::fill_py_event(ev_copy, py_ev, py_device);
        }
        catch(py::error_already_set &eas)
        {
            std::cerr << "PyTango generated an unexpected python exception in "
                      << "PyCallBackPushEvent::fill_py_eventmeth_name." << std::endl
                      << "Please report this bug to PyTango with the following report:"
                      << std::endl;
//            PyErr_Print();
        }
        catch(Tango::DevFailed &df)
        {
            std::cerr << "PyTango generated a DevFailed exception in "
                      << "PyCallBackPushEvent::fill_py_eventmeth_name." << std::endl
                      << "Please report this bug to PyTango with the following report:"
                      << std::endl;
            Tango::Except::print_exception(df);
        }
        catch(...)
        {
            std::cerr << "PyTango generated an unknown exception in "
                      << "PyCallBackPushEvent::fill_py_event." << std::endl
                      << "Please report this bug to PyTango." << std::endl;
        }
        try
        {
            std::cerr << "should push event here" <<std::endl;
//            self->attr("push_event")(py_ev);
        }
        catch(py::error_already_set& eas)
        {
            std::cerr << "push_event generated the following python exception:" << std::endl;
//            PyErr_Print();
        }
        catch(Tango::DevFailed &df)
        {
            std::cerr << "push_event generated the following DevFailed exception:" << std::endl;
            Tango::Except::print_exception(df);
        }
        catch(...)
        {
            std::cerr << "push_event generated an unknown exception." << std::endl; \
        }
    };
}

void PyCallBackPushEvent::fill_py_event(Tango::EventData* ev, py::object& py_ev, Tango::DeviceProxy& dp)
{
    copy_device(ev, py_ev, dp);
    if (ev->attr_value)
    {
        Tango::DeviceAttribute *attr = new Tango::DeviceAttribute;
        (*attr) = std::move(*ev->attr_value);
        py_ev.attr("attr_value") = PyDeviceAttribute::convert_to_python(attr, *ev->device);
    }
    // ev->attr_value = 0; // Do not delete, python will.
}


void PyCallBackPushEvent::fill_py_event(Tango::AttrConfEventData* ev, py::object& py_ev, Tango::DeviceProxy& dp)
{
    copy_device(ev, py_ev, dp);

    if (ev->attr_conf) {
        py_ev.attr("attr_conf") = *ev->attr_conf;
    }
}

void PyCallBackPushEvent::fill_py_event(Tango::DataReadyEventData* ev, py::object& py_ev, Tango::DeviceProxy& dp)
{
    copy_device(ev, py_ev, dp);
}

void PyCallBackPushEvent::fill_py_event(Tango::PipeEventData* ev, py::object& py_ev, Tango::DeviceProxy& dp)
{
    copy_device(ev, py_ev, dp);
    if (ev->pipe_value) {
        Tango::DevicePipe *pipe_value = new Tango::DevicePipe;
        (*pipe_value) = std::move(*ev->pipe_value);
        py_ev.attr("pipe_value") = PyTango::DevicePipe::convert_to_python(pipe_value);
    }
}

void PyCallBackPushEvent::fill_py_event(Tango::DevIntrChangeEventData* ev, py::object& py_ev, Tango::DeviceProxy& dp)
{
    copy_device(ev, py_ev, dp);
    py_ev.attr("cmd_list") = ev->cmd_list;
    py_ev.attr("att_list") = ev->att_list;
}


void PyCallBackPushEvent::push_event(Tango::EventData *ev)
{
//    _push_event(this, ev);
}

void PyCallBackPushEvent::push_event(Tango::AttrConfEventData *ev)
{
//    _push_event(this, ev);
}

void PyCallBackPushEvent::push_event(Tango::DataReadyEventData *ev)
{
//    _push_event(this, ev);
}

void PyCallBackPushEvent::push_event(Tango::PipeEventData *ev)
{
    Tango::PipeEventData event_data = *ev;
    py::object py_ev = py::cast(event_data);
    AutoPythonGILEnsure gil;
//    m_callback(py_ev);
//    _push_event(this, ev);
}

void PyCallBackPushEvent::push_event(Tango::DevIntrChangeEventData *ev)
{
    Tango::DevIntrChangeEventData event_data = *ev;
    py::object py_ev = py::cast(event_data);
    AutoPythonGILEnsure gil;
//    m_callback(py_ev);
//    _push_event(this, ev);
}

void export_callback(py::module &m)
{
    PyCallBackAutoDie::init();

    py::class_<PyCmdDoneEvent>(m, "CmdDoneEvent")
        .def_readonly("device", &PyCmdDoneEvent::device)
        .def_readonly("cmd_name", &PyCmdDoneEvent::cmd_name)
        .def_readonly("argout_raw", &PyCmdDoneEvent::argout_raw)
        .def_readonly("err", &PyCmdDoneEvent::err)
        .def_readonly("errors", &PyCmdDoneEvent::errors)
        .def_readonly("ext", &PyCmdDoneEvent::ext)
        .def_readwrite("argout", &PyCmdDoneEvent::argout)
    ;

    py::class_<PyAttrReadEvent>(m, "AttrReadEvent")
        .def_readonly("device", &PyAttrReadEvent::device)
        .def_readonly("attr_names", &PyAttrReadEvent::attr_names)
        .def_readonly("argout", &PyAttrReadEvent::argout)
        .def_readonly("err", &PyAttrReadEvent::err)
        .def_readonly("errors", &PyAttrReadEvent::errors)
        .def_readonly("ext", &PyAttrReadEvent::ext)
    ;

    py::class_<PyAttrWrittenEvent>(m, "AttrWrittenEvent")
        .def_readonly("ext", &PyAttrWrittenEvent::ext)
        .def_readonly("attr_names", &PyAttrWrittenEvent::attr_names)
        .def_readonly("err", &PyAttrWrittenEvent::err)
        .def_readonly("errors", &PyAttrWrittenEvent::errors)
        .def_readonly("ext", &PyAttrWrittenEvent::ext)
    ;

//    py::class_<CallBackAutoDie>(m, "__CallBackAutoDie", py::dynamic_attr(), "INTERNAL CLASS - DO NOT USE IT")
    py::class_<PyCallBackAutoDie>(m, "__CallBackAutoDie", "INTERNAL CLASS - DO NOT USE IT")
        .def(py::init<>())
//        .def("set_callback", [&](CallBackAutoDie& self, py::object& callback) {
////            std::unique_ptr<PyObject> cbk(callback.ptr());
////            self.m_cbk = std::move(cbk);
//            self.m_callback = callback;
//            cout << "setting callback" << endl;
//            m.attr("free_on_callback") = self;
//        })
//        .def("set_weak_parent", [](CallBackAutoDie& self, py::object parent) {
//            cout << ".def set weak parent" << endl;
//            self.m_weak_parent = parent.ptr();
//            cerr << "ref count" << static_cast<int>(Py_REFCNT(self.m_weak_parent)) << endl;
//            Py_INCREF(self.m_weak_parent);
//            cerr << "ref count" << static_cast<int>(Py_REFCNT(self.m_weak_parent)) << endl;
//            cout << ".def set weak parent" << endl;
//        })
//        .def("set_autokill_references", [](CallBackAutoDie& self, py::object parent) {
//            cout << "set autokill refs" << endl;
////            CallBackAutoDie::s_weak2ob[parent.ptr()] = self.m_callback.ptr();
//        })
//        .def("cmd_ended", &PyCallBackAutoDie::cmd_ended,
//            "This method is defined as being empty and must be overloaded by the user when the asynchronous callback model is used. This is the method which will be executed when the server reply from a command_inout is received in both push and pull sub-mode.")
//        .def("attr_read", &PyCallBackAutoDie::attr_read,
//            "This method is defined as being empty and must be overloaded by the user when the asynchronous callback model is used. This is the method which will be executed when the server reply from a read_attribute(s) is received in both push and pull sub-mode.")
//        .def("attr_written", &PyCallBackAutoDie::attr_written,
//            "This method is defined as being empty and must be overloaded by the user when the asynchronous callback model is used. This is the method which will be executed when the server reply from a write_attribute(s) is received in both push and pull sub-mode. ")
    ;

    py::class_<PyCallBackPushEvent>(m,"__CallBackPushEvent", py::dynamic_attr(), "INTERNAL CLASS - DO NOT USE IT")
        .def(py::init<>())
//        .def("device", [](PyCallBackPushEvent& self, Tango::DeviceProxy& dp) {
//            self.m_device = dp;
//        })
//        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::EventData*))&PyCallBackAutoDie::push_event,
//            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send event(s) to the client. ")
//        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::AttrConfEventData*))&PyCallBackAutoDie::push_event,
//            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send attribute configuration change event(s) to the client. ")
//        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::DataReadyEventData*))&PyCallBackAutoDie::push_event,
//            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send attribute data ready event(s) to the client. ")
//        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::PipeEventData*))&PyCallBackAutoDie::push_event,
//            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send pipe event(s) to the client. ")
//        .def("push_event", (void (PyCallBackAutoDie::*)(Tango::DevIntrChangeEventData*))&PyCallBackAutoDie::push_event,
//            "This method is defined as being empty and must be overloaded by the user when events are used. This is the method which will be executed when the server send device interface change event(s) to the client. ")
    ;
}
