/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"

#if BOOST_VERSION < 103400
#define DISABLE_BOOST_DOCSTRING_OPTIONS
#endif

#ifndef DISABLE_PYTANGO_NUMPY
#   define PY_ARRAY_UNIQUE_SYMBOL pytango_ARRAY_API
#   include <numpy/arrayobject.h>
#endif

#include <tango.h>

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
void export_pipe();
void export_attr();
void export_attribute();
void export_encoded_attribute();
void export_wattribute();
void export_multi_attribute();
void export_multi_class_attribute();
void export_user_default_attr_prop();
void export_user_default_pipe_prop();
void export_sub_dev_diag();
void export_dserver();
void export_device_class();
void export_device_impl();
void export_group();
void export_log4tango();
void export_auto_tango_monitor();

#ifdef DISABLE_PYTANGO_NUMPY
void init_numpy(void) {}
#elif PY_MAJOR_VERSION >= 3
void* init_numpy(void) { import_array(); return NULL; }
#else
void init_numpy(void) { import_array(); return; }
#endif

BOOST_PYTHON_MODULE(_tango)
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

    init_numpy();

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
    export_pipe();
    export_attr();
    export_attribute();
    export_encoded_attribute();
    export_wattribute();
    export_multi_attribute();
    export_multi_class_attribute();
    export_user_default_attr_prop();
    export_user_default_pipe_prop();
    export_sub_dev_diag();
    export_device_class();
    export_device_impl();
    //@warning export_dserver must be made after export_device_impl
    export_dserver();
    export_group();
    export_log4tango();
    export_auto_tango_monitor();
}
