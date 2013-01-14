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

%MappedType std::string
{
%TypeHeaderCode
#include <string>
%End

%ConvertToTypeCode
if (sipIsErr == NULL)
    return PyString_Check(sipPy);

    *sipCppPtr = new std::string(PyString_AsString(sipPy), PyString_Size(sipPy));

    return sipGetState(sipTransferObj);
%End

%ConvertFromTypeCode
return PyString_FromStringAndSize(sipCpp->data(),sipCpp->length());
%End

};

%MappedType std::vector<std::string>
{
%TypeHeaderCode
#include <string>
#include <vector>
%End

%ConvertToTypeCode
if (sipIsErr == NULL)
{	
bool aReturnFlag = PyList_Check(sipPy);
for(int i = 0;aReturnFlag && i < PyList_Size(sipPy);++i)
	aReturnFlag = PyString_Check(PyList_GET_ITEM(sipPy,i));
return aReturnFlag;
}
 
*sipCppPtr = new std::vector<std::string>(PyList_Size(sipPy));
for(int i = 0;i < PyList_Size(sipPy);++i)
  (*sipCppPtr)->operator[](i) = PyString_AsString(PyList_GET_ITEM(sipPy,i));

    return sipGetState(sipTransferObj);
%End

%ConvertFromTypeCode
PyObject* aReturnListPt = PyList_New(sipCpp->size());
std::vector<std::string>::iterator vIter = sipCpp->begin();

for(int i = 0;vIter != sipCpp->end();++vIter,++i)
   PyList_SET_ITEM(aReturnListPt,i,PyString_FromStringAndSize(vIter->data(),vIter->length()));

return aReturnListPt;
%End

};

%MappedType std::vector<int>
{
%TypeHeaderCode
#include <vector>
%End

%ConvertToTypeCode
if (sipIsErr == NULL)
{	
bool aReturnFlag = PyList_Check(sipPy);
for(int i = 0;aReturnFlag && i < PyList_Size(sipPy);++i)
	aReturnFlag = PyInt_Check(PyList_GET_ITEM(sipPy,i));
return aReturnFlag;
}
 
*sipCppPtr = new std::vector<int>(PyList_Size(sipPy));
for(int i = 0;i < PyList_Size(sipPy);++i)
  (*sipCppPtr)->operator[](i) = PyInt_AS_LONG(PyList_GET_ITEM(sipPy,i));

    return sipGetState(sipTransferObj);
%End

%ConvertFromTypeCode
PyObject* aReturnListPt = PyList_New(sipCpp->size());
std::vector<int>::iterator vIter = sipCpp->begin();

for(int i = 0;vIter != sipCpp->end();++vIter,++i)
  PyList_SET_ITEM(aReturnListPt,i,PyInt_FromLong(*vIter));

return aReturnListPt;
%End

};

