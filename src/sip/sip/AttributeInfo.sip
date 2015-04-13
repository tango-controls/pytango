/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

struct AttributeInfo : DeviceAttributeConfig
{
%TypeHeaderCode
#include <tango.h>
using namespace Tango;
%End

    DispLevel disp_level;

    bool operator==(const AttributeInfo &);
};

%MappedType AttributeInfoList
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
                                              sipType_AttributeInfo,
                                              SIP_NOT_NONE);
        }
        return aReturnFlag;
    }

    AttributeInfoList *attrInfoList = new AttributeInfoList();
    for(int i = 0; i < PySequence_Size(sipPy); ++i)
    {
        int state;
        AttributeInfo *aAttributeInfo = reinterpret_cast<AttributeInfo*>(
                              sipConvertToType(PySequence_ITEM(sipPy, i),
                              sipType_AttributeInfo, Py_None,
                              SIP_NOT_NONE,
                              &state, sipIsErr));
        // Deal with any errors.
        if (*sipIsErr)
        {
            sipReleaseType(aAttributeInfo, sipType_AttributeInfo, state);

            // Tidy up.
            delete attrInfoList;

            // There is no temporary instance.
            return 0;
        }
        attrInfoList->push_back(*aAttributeInfo);
        sipReleaseType(aAttributeInfo, sipType_AttributeInfo, state);
    }
    *sipCppPtr = attrInfoList;
    return sipGetState(sipTransferObj);
%End

%ConvertFromTypeCode
    PyObject *l;

    // Create the Python list of the correct length.
    if (!(l = PyList_New(sipCpp->size())))
        return NULL;

    // Go through each element in the C++ instance and convert it to a
    // wrapped AttributeInfo
    int i = 0;
    for(std::vector<AttributeInfo>::iterator j = sipCpp->begin();
        j != sipCpp->end(); ++j, ++i)
    {
        AttributeInfo *aAttributeInfo = new AttributeInfo(*j);
        PyObject *wobj;

        if (!(wobj = sipConvertFromNewType(aAttributeInfo, 
                                           sipType_AttributeInfo,
                                           sipTransferObj)))
        {
            // There was an error so garbage collect the Python list.
            Py_DECREF(l);
            return NULL;
        }
        PyList_SET_ITEM(l, i, wobj);
    }
    delete sipCpp;
    return l;
%End
};

