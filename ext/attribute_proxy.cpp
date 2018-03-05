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

namespace py = pybind11;

//extern const char *param_must_be_seq;
//extern const char *unreachable_code;
//extern const char *non_string_seq;
//
//namespace PyAttributeProxy
//{
//    struct PickleSuite : bopy::pickle_suite
//    {
//        static bopy::tuple getinitargs(Tango::AttributeProxy& self)
//        {
//            Tango::DeviceProxy* dev = self.get_device_proxy();
//
//            std::string ret = dev->get_db_host() + ":" + dev->get_db_port() +
//                             "/" + dev->dev_name() + "/" + self.name();
//            return bopy::make_tuple(ret);
//        }
//    };
//}

void export_attribute_proxy(py::module& m)
{
    // The following function declarations are necessary to be able to cast
    // the function parameters from string& to const string&, otherwise python
    // will not recognize the method calls

//    void (Tango::AttributeProxy::*get_property_)(std::string &, Tango::DbData &) =
//        &Tango::AttributeProxy::get_property;
//
//    void (Tango::AttributeProxy::*delete_property_)(std::string &) =
//        &Tango::AttributeProxy::delete_property;

    py::class_<Tango::AttributeProxy>(m, "__AttributeProxy")
        .def(py::init<const Tango::AttributeProxy &>())
        .def("__init__", [](const std::string& name) {
            return std::shared_ptr<Tango::AttributeProxy>(new Tango::AttributeProxy(name.c_str()));
        })
        .def("__init__", [](const Tango::DeviceProxy *dev, const std::string& name) {
            return std::shared_ptr<Tango::AttributeProxy>(new Tango::AttributeProxy(dev, name.c_str()));
        })

        //
        // Pickle
        //
//        .def_pickle(PyAttributeProxy::PickleSuite())
        
        //
        // general methods
        //
        .def("name", &Tango::AttributeProxy::name)
        .def("get_device_proxy", &Tango::AttributeProxy::get_device_proxy,
            py::return_value_policy::reference_internal)
//            py::return_internal_reference<1>())

        //
        // property methods
        //
        .def("_get_property",
            (void (Tango::AttributeProxy::*) (std::string &, Tango::DbData &))
            &Tango::AttributeProxy::get_property)
        .def("_get_property",
            (void (Tango::AttributeProxy::*) (std::vector<std::string>&, Tango::DbData &))
            &Tango::AttributeProxy::get_property)
        .def("_get_property",
            (void (Tango::AttributeProxy::*) (Tango::DbData &))
            &Tango::AttributeProxy::get_property)
        .def("_put_property", &Tango::AttributeProxy::put_property,
            (py::arg("self"), py::arg("propdata")))
        .def("_delete_property", (void (Tango::AttributeProxy::*) (std::string &))
            &Tango::AttributeProxy::delete_property,
            (py::arg("self"), py::arg("propname")))
        .def("_delete_property", (void (Tango::AttributeProxy::*) (std::vector<std::string> &))
            &Tango::AttributeProxy::delete_property,
            (py::arg("self"), py::arg("propnames")))
        .def("_delete_property", (void (Tango::AttributeProxy::*) (Tango::DbData &))
            &Tango::AttributeProxy::delete_property,
            (py::arg("self"), py::arg("propdata")))
    ;
}

