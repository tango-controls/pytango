#include <boost/python.hpp>
#include <tango.h>

using namespace boost::python;

namespace PySubDevDiag
{
    PyObject *get_sub_devices(Tango::SubDevDiag &self)
    {
        Tango::DevVarStringArray *sub_devs = self.get_sub_devices();
        
        boost::python::list py_sub_devs;
        for(unsigned long i = 0; i < sub_devs->length(); ++i)
        {
            py_sub_devs.append((*sub_devs)[i].in());
        }
        delete sub_devs;
        return boost::python::incref(py_sub_devs.ptr());
    }
}

void export_sub_dev_diag()
{
    class_<Tango::SubDevDiag, boost::noncopyable>
        ("SubDevDiag", no_init)
        .def("set_associated_device", &Tango::SubDevDiag::set_associated_device)
        .def("get_associated_device", &Tango::SubDevDiag::get_associated_device)
        .def("register_sub_device", &Tango::SubDevDiag::register_sub_device)
        .def("remove_sub_devices", (void (Tango::SubDevDiag::*) ())
            &Tango::SubDevDiag::remove_sub_devices)
        .def("remove_sub_devices", (void (Tango::SubDevDiag::*) (std::string))
            &Tango::SubDevDiag::remove_sub_devices)
        .def("get_sub_devices", &PySubDevDiag::get_sub_devices)
        .def("store_sub_devices", &Tango::SubDevDiag::store_sub_devices)
        .def("get_sub_devices_from_cache", &Tango::SubDevDiag::get_sub_devices_from_cache)
    ;
}
