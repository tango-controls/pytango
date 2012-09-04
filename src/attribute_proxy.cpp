/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
   
   PyTango is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   PyTango is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

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

    bopy::class_<Tango::AttributeProxy> AttributeProxy(
        "__AttributeProxy",
        bopy::init<const char *>())
    ;

    AttributeProxy
        .def(bopy::init<const Tango::DeviceProxy *, const char *>())
        .def(bopy::init<const Tango::AttributeProxy &>())

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

