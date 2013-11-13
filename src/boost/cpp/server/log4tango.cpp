/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include "defs.h"
#include "pytgutils.h"

using namespace boost::python;

extern const char *param_must_be_seq;
extern const char *non_string_seq;

namespace PyLogging
{
    void add_logging_target(object &obj)
    {
        PyObject *obj_ptr = obj.ptr();
        if(PySequence_Check(obj_ptr) == 0)
        {
            raise_(PyExc_TypeError, param_must_be_seq);
        }

        Tango::DevVarStringArray par;
        int len = (int) PySequence_Length(obj_ptr);
        par.length(len);
        for(int i = 0; i < len; ++i)
        {
            PyObject* item_ptr = PySequence_GetItem(obj_ptr, i);
            str item = str(handle<>(item_ptr));
            par[i] = CORBA::string_dup(extract<const char*>(item));
        }
        Tango::Logging::add_logging_target(&par);
    }

    void remove_logging_target(object &obj)
    {
        PyObject *obj_ptr = obj.ptr();
        if(PySequence_Check(obj_ptr) == 0)
        {
            raise_(PyExc_TypeError, param_must_be_seq);
        }

        Tango::DevVarStringArray par;
        int len = (int) PySequence_Length(obj_ptr);
        par.length(len);
        for(int i = 0; i < len; ++i)
        {
            PyObject* item_ptr = PySequence_GetItem(obj_ptr, i);
            str item = str(handle<>(item_ptr));
            par[i] = CORBA::string_dup(extract<const char*>(item));
        }
        Tango::Logging::remove_logging_target(&par);
    }
}

void export_log4tango()
{
    {
        scope level_scope =
            class_<log4tango::Level, boost::noncopyable>("Level", no_init)

            .def("get_name", &log4tango::Level::get_name,
            return_value_policy<copy_const_reference>())
            .def("get_value", &log4tango::Level::get_value)
            .staticmethod("get_name")
            .staticmethod("get_value")
        ;

        enum_<log4tango::Level::LevelLevel>("LevelLevel")
            .value("OFF", log4tango::Level::OFF)
            .value("FATAL", log4tango::Level::FATAL)
            .value("ERROR", log4tango::Level::ERROR)
            .value("WARN", log4tango::Level::WARN)
            .value("INFO", log4tango::Level::INFO)
            .value("DEBUG", log4tango::Level::DEBUG)
        ;
    }

    class_<log4tango::Logger, boost::noncopyable>("Logger",
        init<const std::string &, optional<log4tango::Level::Value> >())

        .def("get_name", &log4tango::Logger::get_name,
            return_value_policy<copy_const_reference>())
        .def("set_level", &log4tango::Logger::set_level)
        .def("get_level", &log4tango::Logger::get_level)
        .def("is_level_enabled", &log4tango::Logger::is_level_enabled)
        .def("__log",
            (void (log4tango::Logger::*)(log4tango::Level::Value, const std::string &))
            &log4tango::Logger::log)
        .def("__log_unconditionally",
            (void (log4tango::Logger::*)(log4tango::Level::Value, const std::string &))
            &log4tango::Logger::log_unconditionally)
        .def("__debug",
            (void (log4tango::Logger::*)(const std::string &))
            &log4tango::Logger::debug)
        .def("__info",
            (void (log4tango::Logger::*)(const std::string &))
            &log4tango::Logger::info)
        .def("__warn",
            (void (log4tango::Logger::*)(const std::string &))
            &log4tango::Logger::warn)
        .def("__error",
            (void (log4tango::Logger::*)(const std::string &))
            &log4tango::Logger::error)
        .def("__fatal",
            (void (log4tango::Logger::*)(const std::string &))
            &log4tango::Logger::fatal)
        .def("is_debug_enabled", &log4tango::Logger::is_debug_enabled)
        .def("is_info_enabled", &log4tango::Logger::is_info_enabled)
        .def("is_warn_enabled", &log4tango::Logger::is_warn_enabled)
        .def("is_error_enabled", &log4tango::Logger::is_error_enabled)
        .def("is_fatal_enabled", &log4tango::Logger::is_fatal_enabled)
    ;

    class_<Tango::Logging, boost::noncopyable>("Logging", no_init)
        .def("get_core_logger", &Tango::Logging::get_core_logger,
            return_value_policy<reference_existing_object>())
        .def("add_logging_target", &PyLogging::add_logging_target)
        .def("remove_logging_target", &PyLogging::remove_logging_target)
        .def("start_logging", &Tango::Logging::start_logging)
        .def("stop_logging", &Tango::Logging::stop_logging)
        .staticmethod("get_core_logger")
        .staticmethod("add_logging_target")
        .staticmethod("remove_logging_target")
        .staticmethod("start_logging")
        .staticmethod("stop_logging")
    ;
}
