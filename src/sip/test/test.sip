%Module pytest

%MappedType std::vector<int*>
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
 
*sipCppPtr = new std::vector<int*>(PyList_Size(sipPy));
for(int i = 0;i < PyList_Size(sipPy);++i)
{
  int *v = new int;
  *v = PyInt_AS_LONG(PyList_GET_ITEM(sipPy,i));
  (*sipCppPtr)->operator[](i) = v;
}
    return sipGetState(sipTransferObj);
%End

%ConvertFromTypeCode
PyObject* aReturnListPt = PyList_New(sipCpp->size());
std::vector<int*>::iterator vIter = sipCpp->begin();

for(int i = 0;vIter != sipCpp->end();++vIter,++i)
{
  int *v = *vIter;
  PyList_SET_ITEM(aReturnListPt,i,PyInt_FromLong(*v));
  delete v;
}
//delete sipCpp;
return aReturnListPt;
%End

};

%ModuleHeaderCode
#include "test.h"
%End

std::vector<int *> get_numbers(int n);
