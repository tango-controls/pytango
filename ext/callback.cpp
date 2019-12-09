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

struct __attribute__ ((visibility("hidden"))) PyCmdDoneEvent {
    py::object device;
    py::object cmd_name;
    py::object argout;
    py::object argout_raw;
    py::object err;
    py::object errors;
    py::object ext;
};

struct __attribute__ ((visibility("hidden"))) PyAttrReadEvent {
    py::object device;
    py::object attr_names;
    py::object argout;
    py::object err;
    py::object errors;
    py::object ext;
};

struct __attribute__ ((visibility("hidden"))) PyAttrWrittenEvent {
    py::object device;
    py::object attr_names;
    py::object err;
    py::object errors;
    py::object ext;
};

static void copy_most_fields(CallBackAutoDie* self, Tango::CmdDoneEvent* ev, PyCmdDoneEvent* py_ev)
{
//    // py_ev->device
//    py_ev->cmd_name =py::object(ev->cmd_name);
//    py_ev->argout_raw =py::object(ev->argout);
//    py_ev->err =py::object(ev->err);
//    py_ev->errors =py::object(ev->errors);
//    // py_ev->ext =py::object(ev->ext);
}

static void copy_most_fields(CallBackAutoDie* self, Tango::AttrReadEvent* ev, PyAttrReadEvent* py_ev)
{
//    // py_ev->device
//    py_ev->attr_names =py::object(ev->attr_names);
//
//    PyDeviceAttribute::AutoDevAttrVector dev_attr_vec(ev->argout);
//    py_ev->argout = PyDeviceAttribute::convert_to_python(dev_attr_vec, *ev->device);
//
//    py_ev->err =py::object(ev->err);
//    py_ev->errors =py::object(ev->errors);
//    // py_ev->ext =py::object(ev->ext);
}

static void copy_most_fields(CallBackAutoDie* self, const Tango::AttrWrittenEvent* ev, PyAttrWrittenEvent* py_ev)
{
//    // py_ev->device
//    py_ev->attr_names =py::object(ev->attr_names);
//    py_ev->err =py::object(ev->err);
//    py_ev->errors =py::object(ev->errors);
//    // py_ev->ext =py::object(ev->ext);
}

//CallBackAutoDie::CallBackAutoDie(): m_self(0), m_weak_parent(0)) {}
CallBackAutoDie::CallBackAutoDie() {
    cerr << "CallBackAutoDie constructor" << endl;
}

CallBackAutoDie::~CallBackAutoDie() {
    cerr << "CallBackAutoDie destructor" << endl;
//    if (this->m_weak_parent) {
//        CallBackAutoDie::s_weak2ob.erase(this->m_weak_parent);
//        boost::python::xdecref(this->m_weak_parent);
//    }
}

void CallBackAutoDie::set_callback(py::object callback)
{
    cout << "setting callback in func" << endl;
    m_callback = callback;
//    std::unique_ptr<PyObject> cbk(callback.ptr());
//    m_cbk = std::move(cbk);
}

void CallBackAutoDie::set_weak_parent(py::object parent)
{
    cout << "setting weak parent" << endl;
    m_weak_parent = parent.ptr();
    cerr << "ref count" << static_cast<int>(Py_REFCNT(m_weak_parent)) << endl;
    Py_INCREF(m_weak_parent);
    cerr << "ref count" << static_cast<int>(Py_REFCNT(m_weak_parent)) << endl;
}

/*static*/
//py::object CallBackAutoDie::py_on_callback_parent_fades;
/*static*/
//std::map<PyObject*, PyObject*> CallBackAutoDie::s_weak2ob;



/*static*/ void CallBackAutoDie::init()
{
//   py::object py_scope = py::scope();
//    def ("__on_callback_parent_fades", on_callback_parent_fades);
//    CallBackAutoDie::py_on_callback_parent_fades = py_scope.attr("__on_callback_parent_fades");
}

void CallBackAutoDie::on_callback_parent_fades(PyObject* weakobj)
{
    cerr << "on_callback_parent_fades" << endl;
//    PyObject* ob = CallBackAutoDie::s_weak2ob[weakobj];
//
//    if (!ob)
//        return;
//
////     while (ob->ob_refcnt)
//    boost::python::xdecref(ob);
}

void CallBackAutoDie::set_autokill_references(CallBackAutoDie* cb, Tango::Connection& parent)
{
    cout << "set autokill refs " << &parent << endl;
    std::string index = "parent";
}
//{
//    if (m_self == 0)
//        m_self = py_self.ptr();
//
//    assert(m_self == py_self.ptr());
//
//    PyObject* recb = CallBackAutoDie::py_on_callback_parent_fades.ptr();
//    this->m_weak_parent = PyWeakref_NewRef(py_parent.ptr(), recb);
//
//    if (!this->m_weak_parent)
//        throw_error_already_set();
//
//    boost::python::incref(this->m_self);
//    CallBackAutoDie::s_weak2ob[this->m_weak_parent] = py_self.ptr();
//}

//void CallBackAutoDie::unset_autokill_references()
//{
//    boost::python::decref(m_self);
//}


//template<typename OriginalT, typename CopyT>
//static void _run_virtual_once(CallBackAutoDie* self, OriginalT * ev, const char* virt_fn_name)
//{
//    AutoPythonGIL gil;
//
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
//};

void CallBackAutoDie::cmd_ended(Tango::CmdDoneEvent* ev)
{
    cerr << "CallBackAutoDie::cmd_ended " << endl;
    py::gil_scoped_acquire acquire;
//    PyObject* cbk = m_cbk.release();
//    PyObject_CallObject(cbk, nullptr);
    py::gil_scoped_release release;
//    _run_virtual_once<Tango::CmdDoneEvent, PyCmdDoneEvent>(this, ev, "cmd_ended");
};


/*virtual*/
void CallBackAutoDie::attr_read(Tango::AttrReadEvent* ev)
{
    py::print("attr read _run_virtual_once");
//    _run_virtual_once<Tango::AttrReadEvent, PyAttrReadEvent>(this, ev, "attr_read");
};

/*virtual*/
void CallBackAutoDie::attr_written(Tango::AttrWrittenEvent* ev)
{
    py::print("attr written _run_virtual_once");
//    _run_virtual_once<Tango::AttrWrittenEvent, PyAttrWrittenEvent>(this, ev, "attr_written");
};

//
// CallBackPushEvent
//
CallBackPushEvent::CallBackPushEvent() {}
CallBackPushEvent::~CallBackPushEvent() {}

void CallBackPushEvent::set_device(Tango::DeviceProxy& dp)
{
    m_device = dp;
}


void CallBackPushEvent::fill_py_event(Tango::EventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device)
{
    if (ev->attr_value)
    {
        py_ev.attr("device") = py_device;
        py::object py_attr = PyDeviceAttribute::convert_to_python(ev->attr_value);
        py_ev.attr("attr_value") = py_attr.ptr();
    }
}


void CallBackPushEvent::fill_py_event(Tango::AttrConfEventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device)
{
    py_ev.attr("device") = py_device;
    if (ev->attr_conf) {
        py_ev.attr("attr_conf") = *ev->attr_conf;
    }
}

void CallBackPushEvent::fill_py_event(Tango::DataReadyEventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device)
{
    py_ev.attr("device") = py_device;
}

void CallBackPushEvent::fill_py_event(Tango::PipeEventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device)
{
    py_ev.attr("device") = py_device;
    if (ev->pipe_value) {
        Tango::DevicePipe *pipe_value = new Tango::DevicePipe;
        (*pipe_value) = std::move(*ev->pipe_value);
        py_ev.attr("pipe_value") = PyTango::DevicePipe::convert_to_python(pipe_value);
    }
}

void CallBackPushEvent::fill_py_event(Tango::DevIntrChangeEventData* ev, py::object& py_ev, Tango::DeviceProxy& py_device)
{
    py_ev.attr("device") = py_device;
    py_ev.attr("cmd_list") = ev->cmd_list;
    py_ev.attr("att_list") = ev->att_list;
}

/*virtual*/
void CallBackPushEvent::push_event(Tango::EventData *ev)
{
    Tango::EventData tg_ed = *ev;
    py::object py_ev = py::cast(tg_ed);
    fill_py_event(ev, py_ev, m_device);
    py::gil_scoped_acquire acquire;
    m_callback(py_ev);
    py::gil_scoped_release release;
}

/*virtual*/
void CallBackPushEvent::push_event(Tango::AttrConfEventData *ev)
{
    Tango::AttrConfEventData tg_ed = *ev;
    py::object py_ev = py::cast(tg_ed);

    py::gil_scoped_acquire acquire;
    m_callback(py_ev);
    py::gil_scoped_release release;
}

/*virtual*/
void CallBackPushEvent::push_event(Tango::DataReadyEventData *ev)
{
    Tango::DataReadyEventData tg_ed = *ev;
    py::object py_ev = py::cast(tg_ed);

    py::gil_scoped_acquire acquire;
    m_callback(py_ev);
    py::gil_scoped_release release;
}

/*virtual*/
void CallBackPushEvent::push_event(Tango::PipeEventData *ev)
{
    Tango::PipeEventData tg_ed = *ev;
    py::object py_ev = py::cast(tg_ed);

    py::gil_scoped_acquire acquire;
    m_callback(py_ev);
    py::gil_scoped_release release;
}

/*virtual*/
void CallBackPushEvent::push_event(Tango::DevIntrChangeEventData *ev)
{
    Tango::DevIntrChangeEventData tg_ed = *ev;
    py::object py_ev = py::cast(tg_ed);

    py::gil_scoped_acquire acquire;
    m_callback(py_ev);
    py::gil_scoped_release release;
}

void export_callback(py::module &m)
{
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
    py::class_<CallBackAutoDie, std::shared_ptr<CallBackAutoDie>>(m, "__CallBackAutoDie", "INTERNAL CLASS - DO NOT USE IT")
        .def(py::init<>())
        .def("set_callback", [&](CallBackAutoDie& self, py::object& callback) {
//            std::unique_ptr<PyObject> cbk(callback.ptr());
//            self.m_cbk = std::move(cbk);
            self.m_callback = callback;
            cout << "setting callback" << endl;
            m.attr("free_on_callback") = self;
        })
        .def("set_weak_parent", [](CallBackAutoDie& self, py::object parent) {
            cout << ".def set weak parent" << endl;
            self.m_weak_parent = parent.ptr();
            cerr << "ref count" << static_cast<int>(Py_REFCNT(self.m_weak_parent)) << endl;
            Py_INCREF(self.m_weak_parent);
            cerr << "ref count" << static_cast<int>(Py_REFCNT(self.m_weak_parent)) << endl;
            cout << ".def set weak parent" << endl;
        })
//        .def("set_autokill_references", [](CallBackAutoDie& self, py::object parent) {
//            cout << "set autokill refs" << endl;
////            CallBackAutoDie::s_weak2ob[parent.ptr()] = self.m_callback.ptr();
//        })
    ;

    py::class_<CallBackPushEvent>(m,"__CallBackPushEvent", py::dynamic_attr(), "INTERNAL CLASS - DO NOT USE IT")
        .def(py::init<>())
        .def("device", [](CallBackPushEvent& self, Tango::DeviceProxy& dp) {
            self.m_device = dp;
        })
    ;
}
