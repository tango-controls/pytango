/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include <boost/python.hpp>
#include <tango.h>

#include "defs.h"
#include "pyutils.h"

struct DevEncoded_to_tuple
{
    static inline PyObject* convert(Tango::DevEncoded const& a)
    {
        boost::python::str encoded_format(a.encoded_format);
        bopy::object encoded_data = bopy::object(
            bopy::handle<>(PyBytes_FromStringAndSize(
                (const char*)a.encoded_data.get_buffer(),
                (Py_ssize_t)a.encoded_data.length())));
        boost::python::object result = boost::python::make_tuple(encoded_format, encoded_data);
        return boost::python::incref(result.ptr());
    }

    static const PyTypeObject* get_pytype() { return &PyTuple_Type; }
};

template <typename ContainerType>
struct to_list
{
    static inline PyObject* convert(ContainerType const& a)
    {
        boost::python::list result;
        typedef typename ContainerType::const_iterator const_iter;
        for(const_iter it = a.begin(); it != a.end(); it++)
        {
            result.append(boost::python::object(*it));
        }
        return boost::python::incref(result.ptr());
    }

    static const PyTypeObject* get_pytype() { return &PyList_Type; }
};

template <typename ContainerType>
struct to_tuple
{
    static inline PyObject* convert(ContainerType const& a)
    {
        typedef typename ContainerType::const_iterator const_iter;
        PyObject *t = PyTuple_New(a.size());
        int32_t i = 0;
        for(const_iter it = a.begin(); it != a.end(); ++it, ++i)
        {
            PyTuple_SetItem(t, i, boost::python::incref(it->ptr()));
        }
        return t;
    }

    static const PyTypeObject* get_pytype() { return &PyTuple_Type; }
};

template<typename CorbaContainerType>
struct CORBA_sequence_to_tuple
{
    static PyObject* convert(CorbaContainerType const& a)
    {
        unsigned long size = a.length();
        PyObject *t = PyTuple_New(size);
        for(unsigned long i=0; i < size; ++i)
        {
            boost::python::object x(a[i]);
            PyTuple_SetItem(t, i, boost::python::incref(x.ptr()));
        }
        return t;
    }

    static const PyTypeObject* get_pytype() { return &PyTuple_Type; }
};

template<>
struct CORBA_sequence_to_tuple<Tango::DevVarStringArray>
{
    static PyObject* convert(Tango::DevVarStringArray const& a)
    {
        unsigned long size = a.length();
        PyObject *t = PyTuple_New(size);
        for(unsigned long i=0; i < size; ++i)
        {
            
            boost::python::str x(a[i].in());
            PyTuple_SetItem(t, i, boost::python::incref(x.ptr()));
        }
        return t;
    }

    static const PyTypeObject* get_pytype() { return &PyTuple_Type; }
};

template<>
struct CORBA_sequence_to_tuple<Tango::DevVarLongStringArray>
{
    static PyObject* convert(Tango::DevVarLongStringArray const& a)
    {
        unsigned long lsize = a.lvalue.length();
        unsigned long ssize = a.svalue.length();
        PyObject *lt = PyTuple_New(lsize);
        PyObject *st = PyTuple_New(ssize);

        for(unsigned long i=0; i < lsize; ++i)
        {
            boost::python::object x(a.lvalue[i]);
            PyTuple_SetItem(lt, i, boost::python::incref(x.ptr()));
        }

        for(unsigned long i=0; i < ssize; ++i)
        {
            boost::python::str x(a.svalue[i].in());
            PyTuple_SetItem(st, i, boost::python::incref(x.ptr()));
        }
        PyObject *t = PyTuple_New(2);
        PyTuple_SetItem(t, 0, lt);
        PyTuple_SetItem(t, 1, st);
        return t;
    }

    static const PyTypeObject* get_pytype() { return &PyTuple_Type; }
};

template<>
struct CORBA_sequence_to_tuple<Tango::DevVarDoubleStringArray>
{
    static PyObject* convert(Tango::DevVarDoubleStringArray const& a)
    {
        unsigned long dsize = a.dvalue.length();
        unsigned long ssize = a.svalue.length();
        PyObject *dt = PyTuple_New(dsize);
        PyObject *st = PyTuple_New(ssize);

        for(unsigned long i=0; i < dsize; ++i)
        {
            boost::python::object x(a.dvalue[i]);
            PyTuple_SetItem(dt, i, boost::python::incref(x.ptr()));
        }

        for(unsigned long i=0; i < ssize; ++i)
        {
            boost::python::str x(a.svalue[i].in());
            PyTuple_SetItem(st, i, boost::python::incref(x.ptr()));
        }
        PyObject *t = PyTuple_New(2);
        PyTuple_SetItem(t, 0, dt);
        PyTuple_SetItem(t, 1, st);
        return t;
    }

    static const PyTypeObject* get_pytype() { return &PyTuple_Type; }
};

template<typename CorbaContainerType>
struct CORBA_sequence_to_list
{
    static PyObject* convert(CorbaContainerType const& a)
    {
        unsigned long size = a.length();
        boost::python::list ret;
        for(unsigned long i=0; i < size; ++i)
        {
            ret.append(a[i]);
        }
        return boost::python::incref(ret.ptr());
    }

    static const PyTypeObject* get_pytype() { return &PyList_Type; }
};

template<>
struct CORBA_sequence_to_list<Tango::DevVarStringArray>
{
    static boost::python::list to_list(Tango::DevVarStringArray const& a)
    {
        unsigned long size = a.length();
        boost::python::list ret;
        for(unsigned long i=0; i < size; ++i)
        {
            ret.append(a[i].in());
        }
        return ret;
    }
    
    static PyObject* convert(Tango::DevVarStringArray const& a)
    {
        return boost::python::incref(to_list(a).ptr());
    }

    static const PyTypeObject* get_pytype() { return &PyList_Type; }
};

template<>
struct CORBA_sequence_to_list<Tango::DevVarLongStringArray>
{
    static PyObject* convert(Tango::DevVarLongStringArray const& a)
    {
        unsigned long lsize = a.lvalue.length();
        unsigned long ssize = a.svalue.length();
        
        boost::python::list ret, lt, st;
        for(unsigned long i=0; i < lsize; ++i)
        {
            lt.append(a.lvalue[i]);
        }
        
        for(unsigned long i=0; i < ssize; ++i)
        {
            st.append(a.svalue[i]);
        }
        
        ret.append(lt);
        ret.append(st);
        
        return boost::python::incref(ret.ptr());
    }

    static const PyTypeObject* get_pytype() { return &PyList_Type; }
};

template<>
struct CORBA_sequence_to_list <Tango::DevVarDoubleStringArray>
{
    static PyObject* convert(Tango::DevVarDoubleStringArray const& a)
    {
        unsigned long dsize = a.dvalue.length();
        unsigned long ssize = a.svalue.length();
        
        boost::python::list ret, dt, st;
        for(unsigned long i=0; i < dsize; ++i)
        {
            dt.append(a.dvalue[i]);
        }
        
        for(unsigned long i=0; i < ssize; ++i)
        {
            st.append(a.svalue[i]);
        }
        
        ret.append(dt);
        ret.append(st);
        
        return boost::python::incref(ret.ptr());
    }

    static const PyTypeObject* get_pytype() { return &PyList_Type; }
};

struct CORBA_String_member_to_str
{
    static inline PyObject* convert(CORBA::String_member const& cstr)
    {
        return from_char_to_str(cstr.in());
    }

    //static const PyTypeObject* get_pytype() { return &PyBytes_Type; }
};

struct CORBA_String_member_to_str2
{
    static inline PyObject* convert(_CORBA_String_member const& cstr)
    {
        return from_char_to_str(cstr.in());
    }

    //static const PyTypeObject* get_pytype() { return &PyBytes_Type; }
};

struct CORBA_String_element_to_str
{
    static inline PyObject* convert(_CORBA_String_element const& cstr)
    {
        return from_char_to_str(cstr.in());
    }

    //static const PyTypeObject* get_pytype() { return &PyBytes_Type; }
};

struct String_to_str
{
    static inline PyObject* convert(std::string const& cstr)
    {
        return from_char_to_str(cstr);
    }

    //static const PyTypeObject* get_pytype() { return &PyBytes_Type; }
};

struct char_ptr_to_str
{
    static inline PyObject* convert(const char *cstr)
    {
        return from_char_to_str(cstr);
    }

    //static const PyTypeObject* get_pytype() { return &PyBytes_Type; }
};

boost::python::object to_py(const Tango::AttributeAlarm &);
boost::python::object to_py(const Tango::ChangeEventProp &);
boost::python::object to_py(const Tango::PeriodicEventProp &);
boost::python::object to_py(const Tango::ArchiveEventProp &);
boost::python::object to_py(const Tango::EventProperties &);

template<typename T>
void to_py(Tango::MultiAttrProp<T> &multi_attr_prop, boost::python::object &py_multi_attr_prop)
{
    if(py_multi_attr_prop.ptr() == Py_None)
    {
        PYTANGO_MOD
        py_multi_attr_prop = pytango.attr("MultiAttrProp")();
    }

    py_multi_attr_prop.attr("label") = multi_attr_prop.label;
    py_multi_attr_prop.attr("description") = multi_attr_prop.description;
    py_multi_attr_prop.attr("unit") = multi_attr_prop.unit;
    py_multi_attr_prop.attr("standard_unit") = multi_attr_prop.standard_unit;
    py_multi_attr_prop.attr("display_unit") = multi_attr_prop.display_unit;
    py_multi_attr_prop.attr("format") = multi_attr_prop.format;
    py_multi_attr_prop.attr("min_value") = multi_attr_prop.min_value.get_str();
    py_multi_attr_prop.attr("max_value") = multi_attr_prop.max_value.get_str();
    py_multi_attr_prop.attr("min_alarm") = multi_attr_prop.min_alarm.get_str();
    py_multi_attr_prop.attr("max_alarm") = multi_attr_prop.max_alarm.get_str();
    py_multi_attr_prop.attr("min_warning") = multi_attr_prop.min_warning.get_str();
    py_multi_attr_prop.attr("max_warning") = multi_attr_prop.max_warning.get_str();
    py_multi_attr_prop.attr("delta_t") = multi_attr_prop.delta_t.get_str();
    py_multi_attr_prop.attr("delta_val") = multi_attr_prop.delta_val.get_str();
    py_multi_attr_prop.attr("event_period") = multi_attr_prop.event_period.get_str();
    py_multi_attr_prop.attr("archive_period") = multi_attr_prop.archive_period.get_str();
    py_multi_attr_prop.attr("rel_change") = multi_attr_prop.rel_change.get_str();
    py_multi_attr_prop.attr("abs_change") = multi_attr_prop.abs_change.get_str();
    py_multi_attr_prop.attr("archive_rel_change") = multi_attr_prop.archive_rel_change.get_str();
    py_multi_attr_prop.attr("archive_abs_change") = multi_attr_prop.archive_abs_change.get_str();
}

boost::python::object to_py(const Tango::AttributeConfig &, 
                            boost::python::object py_attr_conf);
boost::python::object to_py(const Tango::AttributeConfig_2 &,
                            boost::python::object py_attr_conf);
boost::python::object to_py(const Tango::AttributeConfig_3 &,
                            boost::python::object py_attr_conf);
boost::python::object to_py(const Tango::AttributeConfig_5 &,
                            boost::python::object py_attr_conf);

boost::python::list to_py(const Tango::AttributeConfigList &);
boost::python::list to_py(const Tango::AttributeConfigList_2 &);
boost::python::list to_py(const Tango::AttributeConfigList_3 &);
boost::python::list to_py(const Tango::AttributeConfigList_5 &);

boost::python::object to_py(const Tango::PipeConfig &,
                            boost::python::object);

boost::python::object to_py(const Tango::PipeConfigList &,
                            boost::python::object);

template<class T>
inline boost::python::object to_py_list(const T *seq)
{
    using namespace boost::python;
    return object(handle<>(CORBA_sequence_to_list<T>::convert(*seq)));
}

template<class T>
inline boost::python::object to_py_tuple(const T *seq)
{
    using namespace boost::python;
    return object(handle<>(CORBA_sequence_to_tuple<T>::convert(*seq)));
}

