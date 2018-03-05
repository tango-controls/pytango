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
#include <pyutils.h>

namespace py = pybind11;

namespace PyTango {

class AutoTangoMonitor {
    Tango::AutoTangoMonitor *mon;
    Tango::DeviceImpl *dev;
    Tango::DeviceClass *klass;

public:
    AutoTangoMonitor(Tango::DeviceImpl *dev_arg) :
            mon(), dev(), klass() {
        dev = dev_arg;
    }

    AutoTangoMonitor(Tango::DeviceClass *klass_arg) :
            mon(), dev(), klass() {
        klass = klass_arg;
    }

    void acquire() {
        if (mon != NULL) {
            return;
        }
        if (dev != NULL) {
            AutoPythonAllowThreads no_gil;
            mon = new Tango::AutoTangoMonitor(dev);
        } else if (klass != NULL) {
            AutoPythonAllowThreads no_gil;
            mon = new Tango::AutoTangoMonitor(klass);
        }
    }

    void release() {
        if (mon != NULL) {
            delete mon;
            mon = NULL;
        }
    }

    ~AutoTangoMonitor() {
        release();
    }
};

class AutoTangoAllowThreads {
public:
    AutoTangoAllowThreads(Tango::DeviceImpl *dev) :
            count(0) {
        Tango::Util* util = Tango::Util::instance();
        Tango::SerialModel ser = util->get_serial_model();

        switch (ser) {
        case Tango::BY_DEVICE:
            mon = &(dev->get_dev_monitor());
            break;
        case Tango::BY_CLASS:
            //mon = &(dev->device_class->ext->only_one);
            break;
        case Tango::BY_PROCESS:
            //mon = &(util->ext->only_one);
            break;
        default:
            mon = NULL;
        }
        release();
    }

    void acquire() {
        if (mon == NULL)
            return;

        AutoPythonAllowThreads no_gil;
        for (int i = 0; i < count; ++i) {
            mon->get_monitor();
        }
    }

protected:
    void release() {
        if (mon == NULL)
            return;

        // TODO remove this bullshit
        int cur_thread = omni_thread::self()->id();
        int mon_thread = mon->get_locking_thread_id();
        int lock_count = mon->get_locking_ctr();

        // do something only if the monitor was taken by the current thread
        if ((mon_thread == cur_thread) && lock_count) {
            while (lock_count > 0) {
                mon->rel_monitor();
                lock_count = mon->get_locking_ctr();
                count++;
            }
        } else {
            mon = NULL;
        }
    }

private:
    Tango::TangoMonitor *mon;
    int count;
// TODO remove this bullshit
    omni_thread::ensure_self auto_self;
};

} // end namespace PyTango

void export_auto_tango_monitor(py::module& m) {
    py::class_ < PyTango::AutoTangoMonitor>(m, "AutoTangoMonitor")
        .def(py::init<Tango::DeviceImpl*>())
        .def(py::init<Tango::DeviceClass*>())
        .def("_acquire", &PyTango::AutoTangoMonitor::acquire)
        .def("_release", &PyTango::AutoTangoMonitor::release);

    py::class_ < PyTango::AutoTangoAllowThreads>(m, "AutoTangoAllowThreads")
        .def(py::init<Tango::DeviceImpl*>())
        .def("_acquire", &PyTango::AutoTangoAllowThreads::acquire);
    ;
}
