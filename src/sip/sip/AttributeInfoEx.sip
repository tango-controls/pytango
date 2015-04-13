/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

struct AttributeAlarmInfo
{
%TypeHeaderCode
#include <string>
#include <vector>
#include <tango.h>
using namespace Tango;
%End
    std::string	min_alarm;
    std::string	max_alarm;
    std::string min_warning;
    std::string	max_warning;
    std::string	delta_t;
    std::string delta_val;
    std::vector<std::string> extensions;
};

struct ChangeEventInfo
{
%TypeHeaderCode
#include <string>
#include <vector>
#include <tango.h>
using namespace Tango;
%End
    std::string rel_change;
    std::string abs_change;
};

struct PeriodicEventInfo
{
%TypeHeaderCode
#include <string>
#include <vector>
#include <tango.h>
using namespace Tango;
%End
    std::string	period;
    std::vector<std::string> extensions;
};

struct ArchiveEventInfo
{
%TypeHeaderCode
#include <string>
#include <vector>
#include <tango.h>
using namespace Tango;
%End
    std::string archive_rel_change;
    std::string archive_abs_change;
    std::string archive_period;
    std::vector<std::string> extensions;
};

struct AttributeEventInfo
{
%TypeHeaderCode
#include <string>
#include <vector>
#include <tango.h>
using namespace Tango;
%End
    ChangeEventInfo ch_event;
    PeriodicEventInfo per_event;
    ArchiveEventInfo arch_event;
};

struct AttributeInfoEx : AttributeInfo
{
%TypeHeaderCode
#include <string>
#include <vector>
#include <tango.h>
using namespace Tango;
%End

    AttributeAlarmInfo alarms;
    AttributeEventInfo events;
    std::vector<std::string> sys_extensions;

    bool operator==(const AttributeInfoEx &);
};


%MappedType AttributeInfoListEx
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
                                              sipType_AttributeInfoEx,
                                              SIP_NOT_NONE);
        }
        return aReturnFlag;
    }

    AttributeInfoListEx *attrInfoListEx = new AttributeInfoListEx();
    for(int i = 0; i < PySequence_Size(sipPy); ++i)
    {
        int state;
        AttributeInfoEx *aAttributeInfoEx = reinterpret_cast<AttributeInfoEx*>(
                              sipConvertToType(PySequence_ITEM(sipPy, i),
                              sipType_AttributeInfoEx, Py_None,
                              SIP_NOT_NONE,
                              &state, sipIsErr));
        // Deal with any errors.
        if (*sipIsErr)
        {
            sipReleaseType(aAttributeInfoEx, sipType_AttributeInfoEx, state);

            // Tidy up.
            delete attrInfoListEx;

            // There is no temporary instance.
            return 0;
        }
        attrInfoListEx->push_back(*aAttributeInfoEx);
        sipReleaseType(aAttributeInfoEx, sipType_AttributeInfoEx, state);
    }
    *sipCppPtr = attrInfoListEx;
    return sipGetState(sipTransferObj);
%End

%ConvertFromTypeCode
    PyObject *l;

    // Create the Python list of the correct length.
    if (!(l = PyList_New(sipCpp->size())))
        return NULL;

    // Go through each element in the C++ instance and convert it to a
    // wrapped AttributeInfoEx
    int i = 0;
    for(std::vector<AttributeInfoEx>::iterator j = sipCpp->begin();
        j != sipCpp->end(); ++j, ++i)
    {
        AttributeInfoEx *aAttributeInfoEx = new AttributeInfoEx(*j);
        PyObject *wobj;

        if (!(wobj = sipConvertFromNewType(aAttributeInfoEx, 
                                           sipType_AttributeInfoEx,
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
