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

#pragma once

#include <boost/python.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 103400
#include <boost/python/detail/api_placeholder.hpp>
#endif
#include <tango/tango.h>

#include "defs.h"
#include "tgutils.h"
#include "pyutils.h"
#include "tango_numpy.h"
#include "exception.h"

extern const char *param_must_be_seq;

/**
 * Converter from python sequence of strings to a std::vector<std::string>
 *
 * @param[in] py_value python sequence object or a single string
 * @param[out] result std string vector to be filled
 */
void convert2array(const boost::python::object &py_value, StdStringVector & result);

/**
 * Converter from python sequence of characters to a Tango::DevVarCharArray
 *
 * @param[in] py_value python sequence object or a single string
 * @param[out] result Tango char array to be filled
 */
void convert2array(const boost::python::object &py_value, Tango::DevVarCharArray & result);

/**
 * Converter from python sequence to a Tango CORBA sequence
 *
 * @param[in] py_value python sequence object
 * @param[out] result CORBA sequence to be filled
 */
template<typename TangoElementType>
void convert2array(const boost::python::object &py_value, _CORBA_Sequence<TangoElementType> & result)
{
    size_t size = boost::python::len(py_value);
    result.length(size);
    for (size_t i=0; i < size; ++i) {
        TangoElementType ch = boost::python::extract<TangoElementType>(py_value[i]);
        result[i] = ch;
    }
}


/**
 * Converter from python sequence of strings to a Tango DevVarStringArray
 *
 * @param[in] py_value python sequence object or a single string
 * @param[out] result Tango string array to be filled
 */
void convert2array(const boost::python::object &py_value, Tango::DevVarStringArray & result);

inline void raise_convert2array_DevVarDoubleStringArray()
{
    Tango::Except::throw_exception(
        "PyDs_WrongPythonDataTypeForDoubleStringArray",
        "Converter from python object to DevVarDoubleStringArray needs a python sequence<sequence<double>, sequence<str>>",
        "convert2array()");
}

/**
 * Converter from python sequence<sequence<double>, sequence<str>> to a Tango DevVarDoubleStringArray
 *
 * @param[in] py_value python sequence object
 * @param[out] result Tango array to be filled
 */
void convert2array(const boost::python::object &py_value, Tango::DevVarDoubleStringArray & result);

inline void raise_convert2array_DevVarLongStringArray()
{
    Tango::Except::throw_exception(
        "PyDs_WrongPythonDataTypeForLongStringArray",
        "Converter from python object to DevVarLongStringArray needs a python sequence<sequence<int>, sequence<str>>",
        "convert2array()");
}

/**
 * Converter from python sequence<sequence<int>, sequence<str>> to a Tango DevVarLongStringArray
 *
 * @param[in] py_value python sequence object
 * @param[out] result Tango array to be filled
 */
void convert2array(const boost::python::object &py_value, Tango::DevVarLongStringArray & result);

/**
 * Convert a python sequence into a C++ container
 * The C++ container must have the push_back method
 */
template <typename ContainerType = StdStringVector >
struct from_sequence
{
    static inline void convert(boost::python::object seq, ContainerType& a)
    {
        typedef typename ContainerType::value_type T;
        PyObject *seq_ptr = seq.ptr();
        Py_ssize_t len = PySequence_Length(seq_ptr);
        for(Py_ssize_t i = 0; i < len; ++i)
        {
            PyObject *o_ptr = PySequence_GetItem(seq_ptr, i);
            T s = boost::python::extract<T>(o_ptr);
            a.push_back(s);
            boost::python::decref(o_ptr);
        }
    }

    static inline void convert(boost::python::object seq, Tango::DbData& a)
    {
        PyObject *seq_ptr = seq.ptr();
        Py_ssize_t len = PySequence_Length(seq_ptr);
        for(Py_ssize_t i = 0; i < len; ++i)
        {
            PyObject *o_ptr = PySequence_GetItem(seq_ptr, i);
            a.push_back(Tango::DbDatum(PyString_AsString(o_ptr)));
            boost::python::decref(o_ptr);
        }
    }

    /**
     * Convert a python dictionary to a Tango::DbData. The dictionary keys must
     * be strings representing the DbDatum name. The dictionary value can be
     * be one of the following:
     * - Tango::DbDatum : in this case the key is not used, and the
     *   item inserted in DbData will be a copy of the value
     * - sequence : it is translated into an array of strings and
     *   the DbDatum inserted in DbData will have name as the dict key and value
     *   the sequence of strings
     * - python object : its string representation is used
     *   as a DbDatum to be inserted
     *
     * @param[in] d the python dictionary to be translated
     * @param[out] db_data the array of DbDatum to be filled
     */
    static inline void convert(boost::python::dict d, Tango::DbData& db_data)
    {
        boost::python::object it = d.iteritems();
        int len = boost::python::extract<int>(d.attr("__len__")()) ;
        for(int i = 0 ; i < len; ++i)
        {
            boost::python::tuple pair = (boost::python::tuple)it.attr("next")();
            boost::python::object key = pair[0];
            boost::python::object value = pair[1];
            PyObject *value_ptr = value.ptr();

            boost::python::extract<Tango::DbDatum> ext(value);
            if(ext.check())
            {
                db_data.push_back(ext());
                continue;
            }

            Tango::DbDatum db_datum(PyString_AsString(key.ptr()));
            if((PySequence_Check(value_ptr)) && (!PyString_Check(value_ptr)))
            {
                from_sequence<StdStringVector>::convert(value, db_datum.value_string);
            }
            else
            {
                boost::python::object value_str = value.attr("__str__")();
                db_datum.value_string.push_back(PyString_AsString(value_str.ptr()));
            }
            db_data.push_back(db_datum);
        }
    }
};


extern const char *param_must_be_seq;
/// This class is useful when you need a sequence like C++ type for
/// a function argument, and you have exported this type to python.
/// This will try to convert the parameter directly to the C++ object
/// (valid if the argument passed was an instance of the exported type).
/// If it fails, it will use from_sequence::convert to get a copy
/// of the sequence in the expected format.
/// So for example we can get a function that accepts an object of
/// type StdStringVector, or a list of strings, or a tuple of strings...
template<class SequenceT>
class CSequenceFromPython
{
    SequenceT* m_seq;
    bool m_own;
    public:
    CSequenceFromPython(boost::python::object &py_obj)
    {
        boost::python::extract<SequenceT*> ext(py_obj);
        if (ext.check()) {
            m_seq = ext();
            m_own = false;
        } else {
            if (PySequence_Check(py_obj.ptr()) == 0)
                raise_(PyExc_TypeError, param_must_be_seq);
            if (PyString_Check(py_obj.ptr()) != 0)
                raise_(PyExc_TypeError, param_must_be_seq);

            m_own = true;
            //m_seq = new SequenceT(PySequence_Length(Py_obj.ptr()));
            m_seq = new SequenceT();
            std::auto_ptr<SequenceT> guard(m_seq);
            from_sequence<SequenceT>::convert(py_obj, *m_seq);
            guard.release();
        }
    }
    ~CSequenceFromPython()
    {
        if (m_own)
            delete m_seq;
    }
    SequenceT & operator*()
    {
        return *m_seq;
    }
    const SequenceT & operator*() const
    {
        return *m_seq;
    }
};

void from_py_object(boost::python::object &, Tango::AttributeAlarm &);
void from_py_object(boost::python::object &, Tango::ChangeEventProp &);
void from_py_object(boost::python::object &, Tango::PeriodicEventProp &);
void from_py_object(boost::python::object &, Tango::ArchiveEventProp &);
void from_py_object(boost::python::object &, Tango::EventProperties &);

void from_py_object(boost::python::object &, Tango::AttributeConfig &);
void from_py_object(boost::python::object &, Tango::AttributeConfig_2 &);
void from_py_object(boost::python::object &, Tango::AttributeConfig_3 &);

void from_py_object(boost::python::object &, Tango::AttributeConfigList &);
void from_py_object(boost::python::object &, Tango::AttributeConfigList_2 &);
void from_py_object(boost::python::object &, Tango::AttributeConfigList_3 &);
