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
	DbDatum (std::string);
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

