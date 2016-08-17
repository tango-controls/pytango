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
#include "pytgutils.h"

namespace bopy = boost::python;

extern const char *param_must_be_seq;
extern const char *unreachable_code;
extern const char *non_string_seq;

namespace PyAttributeProxy
{
    struct PickleSuite : bopy::pickle_suite
    {
        static bopy::tuple getinitargs(Tango::AttributeProxy& self)
        {
            Tango::DeviceProxy* dev = self.get_device_proxy();
            
            std::string ret = dev->get_db_host() + ":" + dev->get_db_port() + 
                             "/" + dev->dev_name() + "/" + self.name();
            return bopy::make_tuple(ret);
        }
    };

    static boost::shared_ptr<Tango::AttributeProxy> makeAttributeProxy1(const std::string& name)
    {
        return boost::shared_ptr<Tango::AttributeProxy>(new Tango::AttributeProxy(name.c_str()));
    }

    static boost::shared_ptr<Tango::AttributeProxy> makeAttributeProxy2(const Tango::DeviceProxy *dev, const std::string& name)
    {
      return boost::shared_ptr<Tango::AttributeProxy>(new Tango::AttributeProxy(dev, name.c_str()));
    }
}

void export_attribute_proxy()
{
    // The following function declarations are necessary to be able to cast
    // the function parameters from string& to const string&, otherwise python
    // will not recognize the method calls

    void (Tango::AttributeProxy::*get_property_)(std::string &, Tango::DbData &) =
        &Tango::AttributeProxy::get_property;

    void (Tango::AttributeProxy::*delete_property_)(std::string &) =
        &Tango::AttributeProxy::delete_property;

    bopy::class_<Tango::AttributeProxy> AttributeProxy("__AttributeProxy",
        bopy::init<const Tango::AttributeProxy &>())
    ;

    AttributeProxy
        .def("__init__", boost::python::make_constructor(PyAttributeProxy::makeAttributeProxy1))
        .def("__init__", boost::python::make_constructor(PyAttributeProxy::makeAttributeProxy2))

        //
        // Pickle
        //
        .def_pickle(PyAttributeProxy::PickleSuite())
        
        //
        // general methods
        //

        .def("name", &Tango::AttributeProxy::name,
            ( arg_("self") ))

        .def("get_device_proxy", &Tango::AttributeProxy::get_device_proxy,
            ( arg_("self") ),
            bopy::return_internal_reference<1>())

        //
        // property methods
        //
        .def("_get_property",
            (void (Tango::AttributeProxy::*) (const std::string &, Tango::DbData &))
            get_property_,
            ( arg_("self"), arg_("propname"), arg_("propdata") ) )

        .def("_get_property",
            (void (Tango::AttributeProxy::*) (std::vector<std::string>&, Tango::DbData &))
            &Tango::AttributeProxy::get_property,
            ( arg_("self"), arg_("propnames"), arg_("propdata") ) )

        .def("_get_property",
            (void (Tango::AttributeProxy::*) (Tango::DbData &))
            &Tango::AttributeProxy::get_property,
            ( arg_("self"), arg_("propdata") ) )

        .def("_put_property", &Tango::AttributeProxy::put_property,
            ( arg_("self"), arg_("propdata") ) )

        .def("_delete_property", (void (Tango::AttributeProxy::*) (const std::string &))
            delete_property_,
            ( arg_("self"), arg_("propname") ) )

        .def("_delete_property", (void (Tango::AttributeProxy::*) (StdStringVector &))
            &Tango::AttributeProxy::delete_property,
            ( arg_("self"), arg_("propnames") ) )

        .def("_delete_property", (void (Tango::AttributeProxy::*) (Tango::DbData &))
            &Tango::AttributeProxy::delete_property,
            ( arg_("self"), arg_("propdata") ) )
    ;
}

