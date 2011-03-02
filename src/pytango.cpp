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
#include <boost/version.hpp>
#if BOOST_VERSION < 103400
#define DISABLE_BOOST_DOCSTRING_OPTIONS
#endif

#ifndef DISABLE_PYTANGO_NUMPY
#   define PY_ARRAY_UNIQUE_SYMBOL pytango_ARRAY_API
#   include <numpy/arrayobject.h>
#endif

#include <tango/tango.h>
#include <iostream>
#include <string>

using namespace boost::python;

void export_version();
void export_enums();
void export_constants();
void export_base_types();
void export_event_data();
void export_attr_conf_event_data();
void export_data_ready_event_data();
void export_exceptions();
void export_api_util();
void export_connection();
void export_device_proxy();
void export_attribute_proxy();
void export_db();
void export_callback(); /// @todo not sure were to put it...
void export_util();
void export_attr();
void export_attribute();
void export_wattribute();
void export_multi_attribute();
void export_user_default_attr_prop();
void export_sub_dev_diag();
void export_dserver();
void export_device_class();
void export_device_impl();
void export_group();
void export_log4tango();

BOOST_PYTHON_MODULE(_PyTango)
{

#ifndef DISABLE_BOOST_DOCSTRING_OPTIONS
    // Configure generated docstrings
    const bool show_user_defined = false;
    const bool show_py_signatures = false;

    docstring_options doc_opts(show_user_defined,
                               show_py_signatures);
#endif

    // specify that this module is actually a package
    boost::python::object package = boost::python::scope();
    package.attr("__path__") = "PyTango";

    PyEval_InitThreads();

#   ifndef DISABLE_PYTANGO_NUMPY
        import_array();
#   endif

    export_callback(); /// @todo not sure were to put it...

    export_version();
    export_enums();
    export_constants();
    export_base_types();
    export_event_data();
    export_attr_conf_event_data();
    export_data_ready_event_data();
    export_exceptions();
    export_api_util();
    export_connection();
    export_device_proxy();
    export_attribute_proxy();
    export_db();
    export_util();
    export_attr();
    export_attribute();
    export_wattribute();
    export_multi_attribute();
    export_user_default_attr_prop();
    export_sub_dev_diag();
    export_device_class();
    export_device_impl();
    //@warning export_dserver must be made after export_device_impl
    export_dserver();
    export_group();
    export_log4tango();
}
