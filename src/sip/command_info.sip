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

struct DevCommandInfo
{
%TypeHeaderCode
#include <tango.h>
using namespace Tango;
%End

    std::string cmd_name;
    long        cmd_tag;
    long        in_type;
    long        out_type;
    std::string in_type_desc;
    std::string out_type_desc;

    bool operator==(const DevCommandInfo &);
};

struct CommandInfo : DevCommandInfo
{
%TypeHeaderCode
#include <tango.h>
using namespace Tango;
%End

    DispLevel disp_level;

    bool operator==(const CommandInfo &);
};

/*
%MappedType CommandInfoList
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
                                              sipType_CommandInfo,
                                              SIP_NOT_NONE);
        }
        return aReturnFlag;
    }
    
    CommandInfoList *cmdInfoList = new CommandInfoList();
    for(int i = 0; i < PySequence_Size(sipPy); ++i)
    {
        int state;
        CommandInfo *aCommandInfo = reinterpret_cast<CommandInfo*>(
                              sipConvertToType(PySequence_ITEM(sipPy, i),
                              sipType_CommandInfo, Py_None,
                              SIP_NOT_NONE,
                              &state, sipIsErr));
        // Deal with any errors.
        if (*sipIsErr)
        {
            sipReleaseType(aCommandInfo, sipType_CommandInfo, state);
    
            // Tidy up.
            delete cmdInfoList;
    
            // There is no temporary instance.
            return 0;
        }
        cmdInfoList->push_back(*aCommandInfo);
        sipReleaseType(aCommandInfo, sipType_CommandInfo, state);
    }
    *sipCppPtr = cmdInfoList;
    return sipGetState(sipTransferObj);
%End

%ConvertFromTypeCode
    PyObject *l;

    // Create the Python list of the correct length.
    if (!(l = PyList_New(sipCpp->size())))
        return NULL;

    // Go through each element in the C++ instance and convert it to a
    // wrapped CommandInfo
    int i = 0;
    for(std::vector<CommandInfo>::iterator j = sipCpp->begin();
        j != sipCpp->end(); ++j, ++i)
    {
        CommandInfo &aCommandInfo = *j;
        PyObject *wobj;

        if (!(wobj = sipConvertFromNewType(&aCommandInfo, sipType_CommandInfo, sipTransferObj)))
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
*/
