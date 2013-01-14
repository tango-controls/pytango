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

#pragma once

#include <boost/python.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION < 103400
#include <boost/python/detail/api_placeholder.hpp>
#endif
#include <tango.h>

#include "defs.h"
#include "tgutils.h"
#include "pyutils.h"
#include "tango_numpy.h"
#include "exception.h"

extern const char *param_must_be_seq;

char* obj_to_new_char(PyObject* obj_ptr);

char* obj_to_new_char(bopy::object obj);

void obj_to_string(PyObject* obj_ptr, std::string& result);

void obj_to_string(bopy::object obj, std::string& result);

/// @bug Not a bug per se, but you should keep in mind: It returns a new
/// string, so if you pass it to Tango with a release flag there will be
/// no problems, but if you have to use it yourself then you must remember
/// to delete[] it!
Tango::DevString PyString_AsCorbaString(PyObject* obj_ptr);

/**
 * Converter from python sequence of strings to a std::vector<std::string>
 *
 * @param[in] py_value python sequence object or a single string
 * @param[out] result std string vector to be filled
 */
void convert2array(const bopy::object &py_value, StdStringVector & result);

/**
 * Converter from python sequence of characters to a Tango::DevVarCharArray
 *
 * @param[in] py_value python sequence object or a single string
 * @param[out] result Tango char array to be filled
 */
void convert2array(const bopy::object &py_value, Tango::DevVarCharArray & result);

/**
 * Converter from python sequence to a Tango CORBA sequence
 *
 * @param[in] py_value python sequence object
 * @param[out] result CORBA sequence to be filled
 */
template<typename TangoElementType>
void convert2array(const bopy::object &py_value, _CORBA_Sequence<TangoElementType> & result)
{
    size_t size = bopy::len(py_value);
    result.length(size);
    for (size_t i=0; i < size; ++i) {
        TangoElementType ch = bopy::extract<TangoElementType>(py_value[i]);
        result[i] = ch;
    }
}


/**
 * Converter from python sequence of strings to a Tango DevVarStringArray
 *
 * @param[in] py_value python sequence object or a single string
 * @param[out] result Tango string array to be filled
 */
void convert2array(const bopy::object &py_value, Tango::DevVarStringArray & result);

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
void convert2array(const bopy::object &py_value, Tango::DevVarDoubleStringArray & result);

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
void convert2array(const bopy::object &py_value, Tango::DevVarLongStringArray & result);

/**
 * Convert a python sequence into a C++ container
 * The C++ container must have the push_back method
 */
template <typename ContainerType = StdStringVector >
struct from_sequence
{
    static inline void convert(bopy::object seq, ContainerType& a)
    {
        typedef typename ContainerType::value_type T;
        PyObject *seq_ptr = seq.ptr();
        Py_ssize_t len = PySequence_Length(seq_ptr);
        for(Py_ssize_t i = 0; i < len; ++i)
        {
            PyObject *o_ptr = PySequence_GetItem(seq_ptr, i);
            T s = bopy::extract<T>(o_ptr);
            a.push_back(s);
            bopy::decref(o_ptr);
        }
    }

    static inline void convert(bopy::object seq, Tango::DbData& a)
    {
        PyObject *seq_ptr = seq.ptr();
        Py_ssize_t len = PySequence_Length(seq_ptr);
        for(Py_ssize_t i = 0; i < len; ++i)
        {
            PyObject *o_ptr = PySequence_GetItem(seq_ptr, i);
            if (PyBytes_Check(o_ptr))
            {
                a.push_back(Tango::DbDatum(PyBytes_AS_STRING(o_ptr)));
            }
            else if(PyUnicode_Check(o_ptr))
            {
                PyObject* o_bytes_ptr = PyUnicode_AsLatin1String(o_ptr);
                a.push_back(Tango::DbDatum(PyBytes_AS_STRING(o_bytes_ptr)));
                Py_DECREF(o_bytes_ptr);
            }
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
    static inline void convert(bopy::dict d, Tango::DbData& db_data)
    {
        bopy::object it = d.iteritems();
        int len = bopy::extract<int>(d.attr("__len__")()) ;
        for(int i = 0 ; i < len; ++i)
        {
            bopy::tuple pair = (bopy::tuple)it.attr("next")();
            bopy::object key = pair[0];
            bopy::object value = pair[1];

            bopy::extract<Tango::DbDatum> ext(value);
            if(ext.check())
            {
                db_data.push_back(ext());
                continue;
            }
            
            char const* key_str = bopy::extract<char const*>(key);
            Tango::DbDatum db_datum(key_str);
            
            bopy::extract<char const*> value_str(value);
            
            if(value_str.check())
            {
                db_datum.value_string.push_back(value_str());
            }
            else
            {
                if(PySequence_Check(value.ptr()))
                {
                    from_sequence<StdStringVector>::convert(value, db_datum.value_string);
                }
                else
                {
                    bopy::object str_value = value.attr("__str__")();
                    bopy::extract<char const*> str_value_str(str_value);
                    db_datum.value_string.push_back(str_value_str());
                }
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
    CSequenceFromPython(bopy::object &py_obj)
    {
        bopy::extract<SequenceT*> ext(py_obj);
        if (ext.check()) {
            m_seq = ext();
            m_own = false;
        } else {
            if (PySequence_Check(py_obj.ptr()) == 0)
                raise_(PyExc_TypeError, param_must_be_seq);
            if (PyUnicode_Check(py_obj.ptr()) != 0)
                raise_(PyExc_TypeError, param_must_be_seq);
            if (PyUnicode_Check(py_obj.ptr()) != 0)
                raise_(PyExc_TypeError, param_must_be_seq);

            m_own = true;
            //m_seq = new SequenceT(PySequence_Length(Py_obj.ptr()));
            m_seq = new SequenceT();
            unique_pointer<SequenceT> guard(m_seq);
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

void from_py_object(bopy::object &, Tango::AttributeAlarm &);
void from_py_object(bopy::object &, Tango::ChangeEventProp &);
void from_py_object(bopy::object &, Tango::PeriodicEventProp &);
void from_py_object(bopy::object &, Tango::ArchiveEventProp &);
void from_py_object(bopy::object &, Tango::EventProperties &);

void from_py_object(bopy::object &, Tango::AttributeConfig &);
void from_py_object(bopy::object &, Tango::AttributeConfig_2 &);
void from_py_object(bopy::object &, Tango::AttributeConfig_3 &);

template<typename T>
void from_py_object(bopy::object &py_obj, Tango::MultiAttrProp<T> &multi_attr_prop)
{
	multi_attr_prop.label = bopy::extract<string>(bopy::str(py_obj.attr("label")));
	multi_attr_prop.description = bopy::extract<string>(bopy::str(py_obj.attr("description")));
	multi_attr_prop.unit = bopy::extract<string>(bopy::str(py_obj.attr("unit")));
	multi_attr_prop.standard_unit = bopy::extract<string>(bopy::str(py_obj.attr("standard_unit")));
	multi_attr_prop.display_unit = bopy::extract<string>(bopy::str(py_obj.attr("display_unit")));
	multi_attr_prop.format = bopy::extract<string>(bopy::str(py_obj.attr("format")));

	bopy::extract<string> min_value(py_obj.attr("min_value"));
	if(min_value.check())
		multi_attr_prop.min_value = min_value();
	else
		multi_attr_prop.min_value = bopy::extract<T>(py_obj.attr("min_value"));

	bopy::extract<string> max_value(py_obj.attr("max_value"));
	if(max_value.check())
		multi_attr_prop.max_value = max_value();
	else
		multi_attr_prop.max_value = bopy::extract<T>(py_obj.attr("max_value"));

	bopy::extract<string> min_alarm(py_obj.attr("min_alarm"));
	if(min_alarm.check())
		multi_attr_prop.min_alarm = min_alarm();
	else
		multi_attr_prop.min_alarm = bopy::extract<T>(py_obj.attr("min_alarm"));

	bopy::extract<string> max_alarm(py_obj.attr("max_alarm"));
	if(max_alarm.check())
		multi_attr_prop.max_alarm = max_alarm();
	else
		multi_attr_prop.max_alarm = bopy::extract<T>(py_obj.attr("max_alarm"));

	bopy::extract<string> min_warning(py_obj.attr("min_warning"));
	if(min_warning.check())
		multi_attr_prop.min_warning = min_warning();
	else
		multi_attr_prop.min_warning = bopy::extract<T>(py_obj.attr("min_warning"));

	bopy::extract<string> max_warning(py_obj.attr("max_warning"));
	if(max_warning.check())
		multi_attr_prop.max_warning = max_warning();
	else
		multi_attr_prop.max_warning = bopy::extract<T>(py_obj.attr("max_warning"));

	bopy::extract<string> delta_t(py_obj.attr("delta_t"));
	if(delta_t.check())
		multi_attr_prop.delta_t = delta_t();
	else
		multi_attr_prop.delta_t = bopy::extract<Tango::DevLong>(py_obj.attr("delta_t")); // Property type is Tango::DevLong!

	bopy::extract<string> delta_val(py_obj.attr("delta_val"));
	if(delta_val.check())
		multi_attr_prop.delta_val = delta_val();
	else
		multi_attr_prop.delta_val = bopy::extract<T>(py_obj.attr("delta_val"));

	bopy::extract<string> event_period(py_obj.attr("event_period"));
	if(event_period.check())
		multi_attr_prop.event_period = event_period();
	else
		multi_attr_prop.event_period = bopy::extract<Tango::DevLong>(py_obj.attr("event_period")); // Property type is Tango::DevLong!

	bopy::extract<string> archive_period(py_obj.attr("archive_period"));
	if(archive_period.check())
		multi_attr_prop.archive_period = archive_period();
	else
		multi_attr_prop.archive_period = bopy::extract<Tango::DevLong>(py_obj.attr("archive_period")); // Property type is Tango::DevLong!

	bopy::extract<string> rel_change(py_obj.attr("rel_change"));
	if(rel_change.check())
		multi_attr_prop.rel_change = rel_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("rel_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.rel_change = change_vec;
		}
		else
			multi_attr_prop.rel_change = bopy::extract<Tango::DevDouble>(py_obj.attr("rel_change")); // Property type is Tango::DevDouble!
	}

	bopy::extract<string> abs_change(py_obj.attr("abs_change"));
	if(abs_change.check())
		multi_attr_prop.abs_change = abs_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("abs_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.abs_change = change_vec;
		}
		else
			multi_attr_prop.abs_change = bopy::extract<Tango::DevDouble>(py_obj.attr("abs_change")); // Property type is Tango::DevDouble!
	}

	bopy::extract<string> archive_rel_change(py_obj.attr("archive_rel_change"));
	if(archive_rel_change.check())
		multi_attr_prop.archive_rel_change = archive_rel_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("archive_rel_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.archive_rel_change = change_vec;
		}
		else
			multi_attr_prop.archive_rel_change = bopy::extract<Tango::DevDouble>(py_obj.attr("archive_rel_change")); // Property type is Tango::DevDouble!
	}

	bopy::extract<string> archive_abs_change(py_obj.attr("archive_abs_change"));
	if(archive_abs_change.check())
		multi_attr_prop.archive_abs_change = archive_abs_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("archive_abs_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.archive_abs_change = change_vec;
		}
		else
			multi_attr_prop.archive_abs_change = bopy::extract<Tango::DevDouble>(py_obj.attr("archive_abs_change")); // Property type is Tango::DevDouble!
	}
}

template<>
inline void from_py_object(bopy::object &py_obj, Tango::MultiAttrProp<Tango::DevEncoded> &multi_attr_prop)
{
	multi_attr_prop.label = bopy::extract<string>(bopy::str(py_obj.attr("label")));
	multi_attr_prop.description = bopy::extract<string>(bopy::str(py_obj.attr("description")));
	multi_attr_prop.unit = bopy::extract<string>(bopy::str(py_obj.attr("unit")));
	multi_attr_prop.standard_unit = bopy::extract<string>(bopy::str(py_obj.attr("standard_unit")));
	multi_attr_prop.display_unit = bopy::extract<string>(bopy::str(py_obj.attr("display_unit")));
	multi_attr_prop.format = bopy::extract<string>(bopy::str(py_obj.attr("format")));

	bopy::extract<string> min_value(py_obj.attr("min_value"));
	if(min_value.check())
		multi_attr_prop.min_value = min_value();
	else
		multi_attr_prop.min_value = bopy::extract<Tango::DevUChar>(py_obj.attr("min_value"));

	bopy::extract<string> max_value(py_obj.attr("max_value"));
	if(max_value.check())
		multi_attr_prop.max_value = max_value();
	else
		multi_attr_prop.max_value = bopy::extract<Tango::DevUChar>(py_obj.attr("max_value"));

	bopy::extract<string> min_alarm(py_obj.attr("min_alarm"));
	if(min_alarm.check())
		multi_attr_prop.min_alarm = min_alarm();
	else
		multi_attr_prop.min_alarm = bopy::extract<Tango::DevUChar>(py_obj.attr("min_alarm"));

	bopy::extract<string> max_alarm(py_obj.attr("max_alarm"));
	if(max_alarm.check())
		multi_attr_prop.max_alarm = max_alarm();
	else
		multi_attr_prop.max_alarm = bopy::extract<Tango::DevUChar>(py_obj.attr("max_alarm"));

	bopy::extract<string> min_warning(py_obj.attr("min_warning"));
	if(min_warning.check())
		multi_attr_prop.min_warning = min_warning();
	else
		multi_attr_prop.min_warning = bopy::extract<Tango::DevUChar>(py_obj.attr("min_warning"));

	bopy::extract<string> max_warning(py_obj.attr("max_warning"));
	if(max_warning.check())
		multi_attr_prop.max_warning = max_warning();
	else
		multi_attr_prop.max_warning = bopy::extract<Tango::DevUChar>(py_obj.attr("max_warning"));

	bopy::extract<string> delta_t(py_obj.attr("delta_t"));
	if(delta_t.check())
		multi_attr_prop.delta_t = delta_t();
	else
		multi_attr_prop.delta_t = bopy::extract<Tango::DevLong>(py_obj.attr("delta_t")); // Property type is Tango::DevLong!

	bopy::extract<string> delta_val(py_obj.attr("delta_val"));
	if(delta_val.check())
		multi_attr_prop.delta_val = delta_val();
	else
		multi_attr_prop.delta_val = bopy::extract<Tango::DevUChar>(py_obj.attr("delta_val"));

	bopy::extract<string> event_period(py_obj.attr("event_period"));
	if(event_period.check())
		multi_attr_prop.event_period = event_period();
	else
		multi_attr_prop.event_period = bopy::extract<Tango::DevLong>(py_obj.attr("event_period")); // Property type is Tango::DevLong!

	bopy::extract<string> archive_period(py_obj.attr("archive_period"));
	if(archive_period.check())
		multi_attr_prop.archive_period = archive_period();
	else
		multi_attr_prop.archive_period = bopy::extract<Tango::DevLong>(py_obj.attr("archive_period")); // Property type is Tango::DevLong!

	bopy::extract<string> rel_change(py_obj.attr("rel_change"));
	if(rel_change.check())
		multi_attr_prop.rel_change = rel_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("rel_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.rel_change = change_vec;
		}
		else
			multi_attr_prop.rel_change = bopy::extract<Tango::DevDouble>(py_obj.attr("rel_change")); // Property type is Tango::DevDouble!
	}

	bopy::extract<string> abs_change(py_obj.attr("abs_change"));
	if(abs_change.check())
		multi_attr_prop.abs_change = abs_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("abs_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.abs_change = change_vec;
		}
		else
			multi_attr_prop.abs_change = bopy::extract<Tango::DevDouble>(py_obj.attr("abs_change")); // Property type is Tango::DevDouble!
	}

	bopy::extract<string> archive_rel_change(py_obj.attr("archive_rel_change"));
	if(archive_rel_change.check())
		multi_attr_prop.archive_rel_change = archive_rel_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("archive_rel_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.archive_rel_change = change_vec;
		}
		else
			multi_attr_prop.archive_rel_change = bopy::extract<Tango::DevDouble>(py_obj.attr("archive_rel_change")); // Property type is Tango::DevDouble!
	}

	bopy::extract<string> archive_abs_change(py_obj.attr("archive_abs_change"));
	if(archive_abs_change.check())
		multi_attr_prop.archive_abs_change = archive_abs_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("archive_abs_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.archive_abs_change = change_vec;
		}
		else
			multi_attr_prop.archive_abs_change = bopy::extract<Tango::DevDouble>(py_obj.attr("archive_abs_change")); // Property type is Tango::DevDouble!
	}
}

template<>
inline void from_py_object(bopy::object &py_obj, Tango::MultiAttrProp<Tango::DevString> &multi_attr_prop)
{
	string empty_str("");

	multi_attr_prop.label = bopy::extract<string>(bopy::str(py_obj.attr("label")));
	multi_attr_prop.description = bopy::extract<string>(bopy::str(py_obj.attr("description")));
	multi_attr_prop.unit = bopy::extract<string>(bopy::str(py_obj.attr("unit")));
	multi_attr_prop.standard_unit = bopy::extract<string>(bopy::str(py_obj.attr("standard_unit")));
	multi_attr_prop.display_unit = bopy::extract<string>(bopy::str(py_obj.attr("display_unit")));
	multi_attr_prop.format = bopy::extract<string>(bopy::str(py_obj.attr("format")));

	bopy::extract<string> min_value(py_obj.attr("min_value"));
	if(min_value.check())
		multi_attr_prop.min_value = min_value();
	else
		multi_attr_prop.min_value = empty_str;

	bopy::extract<string> max_value(py_obj.attr("max_value"));
	if(max_value.check())
		multi_attr_prop.max_value = max_value();
	else
		multi_attr_prop.max_value = empty_str;

	bopy::extract<string> min_alarm(py_obj.attr("min_alarm"));
	if(min_alarm.check())
		multi_attr_prop.min_alarm = min_alarm();
	else
		multi_attr_prop.min_alarm = empty_str;

	bopy::extract<string> max_alarm(py_obj.attr("max_alarm"));
	if(max_alarm.check())
		multi_attr_prop.max_alarm = max_alarm();
	else
		multi_attr_prop.max_alarm = empty_str;

	bopy::extract<string> min_warning(py_obj.attr("min_warning"));
	if(min_warning.check())
		multi_attr_prop.min_warning = min_warning();
	else
		multi_attr_prop.min_warning = empty_str;

	bopy::extract<string> max_warning(py_obj.attr("max_warning"));
	if(max_warning.check())
		multi_attr_prop.max_warning = max_warning();
	else
		multi_attr_prop.max_warning = empty_str;

	bopy::extract<string> delta_t(py_obj.attr("delta_t"));
	if(delta_t.check())
		multi_attr_prop.delta_t = delta_t();
	else
		multi_attr_prop.delta_t = bopy::extract<Tango::DevLong>(py_obj.attr("delta_t")); // Property type is Tango::DevLong!

	bopy::extract<string> delta_val(py_obj.attr("delta_val"));
	if(delta_val.check())
		multi_attr_prop.delta_val = delta_val();
	else
		multi_attr_prop.delta_val = empty_str;

	bopy::extract<string> event_period(py_obj.attr("event_period"));
	if(event_period.check())
		multi_attr_prop.event_period = event_period();
	else
		multi_attr_prop.event_period = bopy::extract<Tango::DevLong>(py_obj.attr("event_period")); // Property type is Tango::DevLong!

	bopy::extract<string> archive_period(py_obj.attr("archive_period"));
	if(archive_period.check())
		multi_attr_prop.archive_period = archive_period();
	else
		multi_attr_prop.archive_period = bopy::extract<Tango::DevLong>(py_obj.attr("archive_period")); // Property type is Tango::DevLong!

	bopy::extract<string> rel_change(py_obj.attr("rel_change"));
	if(rel_change.check())
		multi_attr_prop.rel_change = rel_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("rel_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.rel_change = change_vec;
		}
		else
			multi_attr_prop.rel_change = bopy::extract<Tango::DevDouble>(py_obj.attr("rel_change")); // Property type is Tango::DevDouble!
	}

	bopy::extract<string> abs_change(py_obj.attr("abs_change"));
	if(abs_change.check())
		multi_attr_prop.abs_change = abs_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("abs_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.abs_change = change_vec;
		}
		else
			multi_attr_prop.abs_change = bopy::extract<Tango::DevDouble>(py_obj.attr("abs_change")); // Property type is Tango::DevDouble!
	}

	bopy::extract<string> archive_rel_change(py_obj.attr("archive_rel_change"));
	if(archive_rel_change.check())
		multi_attr_prop.archive_rel_change = archive_rel_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("archive_rel_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.archive_rel_change = change_vec;
		}
		else
			multi_attr_prop.archive_rel_change = bopy::extract<Tango::DevDouble>(py_obj.attr("archive_rel_change")); // Property type is Tango::DevDouble!
	}

	bopy::extract<string> archive_abs_change(py_obj.attr("archive_abs_change"));
	if(archive_abs_change.check())
		multi_attr_prop.archive_abs_change = archive_abs_change();
	else
	{
		bopy::object prop_py_obj = bopy::object(py_obj.attr("archive_abs_change"));
		if(PySequence_Check(prop_py_obj.ptr()))
		{
			vector<Tango::DevDouble> change_vec;
			for(long i = 0; i < bopy::len(prop_py_obj); i++)
				change_vec.push_back(bopy::extract<Tango::DevDouble>(prop_py_obj[i]));
			multi_attr_prop.archive_abs_change = change_vec;
		}
		else
			multi_attr_prop.archive_abs_change = bopy::extract<Tango::DevDouble>(py_obj.attr("archive_abs_change")); // Property type is Tango::DevDouble!
	}
}

void from_py_object(bopy::object &, Tango::AttributeConfigList &);
void from_py_object(bopy::object &, Tango::AttributeConfigList_2 &);
void from_py_object(bopy::object &, Tango::AttributeConfigList_3 &);
