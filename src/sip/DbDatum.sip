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

//
// DbDatum - data object for sending and receiving data from the
//               TANGO database api
//
class DbDatum
{

%TypeHeaderCode
#include <tango.h>
using namespace Tango;
%End

public :
	enum except_flags
	{
		isempty_flag,
		wrongtype_flag,
		numFlags
	};

	std::string name;
	std::vector<std::string> value_string;

//
// constructor methods
//
	DbDatum();
	DbDatum (const char *);
	~DbDatum();
	DbDatum(const DbDatum &);
//	DbDatum &operator=(const DbDatum &);

//	size_t size();
	bool is_empty();

////	void exceptions(bitset<numFlags> fl) { exceptions_flags = fl;}
////	bitset<numFlags> exceptions() {return exceptions_flags;}
	void reset_exceptions(except_flags fl);
	void set_exceptions(except_flags fl);

//
// insert methods
//
/*
	void operator << (bool);
	void operator << (short);
	void operator << (unsigned char);
	void operator << (unsigned short);
	void operator << (DevLong);
	void operator << (DevULong);
	void operator << (DevLong64);
	void operator << (DevULong64);
	void operator << (float);
	void operator << (double);
	void operator << (char *);
//	void operator << (char *&);
	void operator << (const char *);
//	void operator << (const char *&);
	void operator << (std::string&);

	void operator << (vector<std::string>&);
	void operator << (vector<short>&);
	void operator << (vector<unsigned short>&);
	void operator << (vector<DevLong>&);
	void operator << (vector<DevULong>&);
	void operator << (vector<DevLong64>&);
	void operator << (vector<DevULong64>&);
	void operator << (vector<float>&);
	void operator << (vector<double>&);

//
// extract methods
//

	bool operator >> (bool&);
	bool operator >> (short&);
	bool operator >> (unsigned char&);
	bool operator >> (unsigned short&);
	bool operator >> (DevLong&);
	bool operator >> (DevULong&);
	bool operator >> (DevLong64&);
	bool operator >> (DevULong64&);
	bool operator >> (float&);
	bool operator >> (double&);
	bool operator >> (const char*&);
	bool operator >> (std::string&);

	bool operator >> (vector<std::string>&);
	bool operator >> (vector<short>&);
	bool operator >> (vector<unsigned short>&);
	bool operator >> (vector<DevLong>&);
	bool operator >> (vector<DevULong>&);
	bool operator >> (vector<DevLong64>&);
	bool operator >> (vector<DevULong64>&);
	bool operator >> (vector<float>&);
	bool operator >> (vector<double>&);
*/
};

%MappedType DbData
{
%TypeHeaderCode
#include <vector>
#include <tango.h>
using namespace Tango;
%End

%ConvertToTypeCode
    // See if we are just being asked to check the type of the Python
    // object.
    if(sipIsErr == NULL)
    {
        bool aReturnFlag = PyList_Check(sipPy);
        for(int i = 0; aReturnFlag && i < PySequence_Size(sipPy); ++i)
        {
            aReturnFlag = sipCanConvertToType(PySequence_ITEM(sipPy, i),
                                              sipType_DbDatum,
                                              SIP_NOT_NONE);
        }
        return aReturnFlag;
    }

    DbData *dbDatumList = new DbData();
    for(int i = 0; i < PySequence_Size(sipPy); ++i)
    {
        int state;
        DbDatum *aDbDatum = reinterpret_cast<DbDatum*>(
                              sipConvertToType(PySequence_ITEM(sipPy, i),
                              sipType_DbDatum, Py_None,
                              SIP_NOT_NONE,
                              &state, sipIsErr));
        // Deal with any errors.
        if (*sipIsErr)
        {
            sipReleaseType(aDbDatum, sipType_DbDatum, state);

            // Tidy up.
            delete dbDatumList;

            // There is no temporary instance.
            return 0;
        }
        dbDatumList->push_back(*aDbDatum);
        sipReleaseType(aDbDatum, sipType_DbDatum, state);
    }
    *sipCppPtr = dbDatumList;
    return sipGetState(sipTransferObj);
%End

%ConvertFromTypeCode
    PyObject *l;

    // Create the Python list of the correct length.
    if (!(l = PyList_New(sipCpp->size())))
        return NULL;

    // Go through each element in the C++ instance and convert it to a
    // wrapped DbDatum
    int i = 0;
    for(std::vector<DbDatum>::iterator j = sipCpp->begin();
        j != sipCpp->end(); ++j, ++i)
    {
        DbDatum *aDbDatum = new DbDatum(*j);
        PyObject *wobj;

        if (!(wobj = sipConvertFromNewType(aDbDatum, sipType_DbDatum, sipTransferObj)))
        {
            // There was an error so garbage collect the Python list.
            Py_DECREF(l);
            return NULL;
        }
        PyList_SET_ITEM(l, i, wobj);
    }
    return l;
%End
};
