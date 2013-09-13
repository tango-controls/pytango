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

%Module Tango 008001000

%Timeline { TANGO_7_2 TANGO_8_0 TANGO_8_1 }

%Platforms {WIN32_PLATFORM POSIX_PLATFORM MACOS_PLATFORM}

%Include std_utils.sip
%Include constants.sip
%Include CommandInfo.sip
%Include DeviceInfo.sip
%Include DeviceAttributeConfig.sip
%Include AttributeInfo.sip
%Include DbDatum.sip
%Include DbDevInfo.sip
%Include DbDevImportInfo.sip
%Include connection.sip
%Include DeviceProxy.sip
%Include Database.sip

%PostInitialisationCode

    PyObject* init_tango = PyImport_ImportModule("__init_tango");
    if (init_tango == NULL)
    {
        if (!PyErr_Occurred())
        {

        }
        return;
    }
    PyObject* init_name = PyString_FromString("init");
    PyObject* ret = PyObject_CallMethodObjArgs(init_tango, init_name, sipModule, sipModuleDict, NULL);
    Py_DECREF(init_name);
    Py_DECREF(ret);
    Py_DECREF(init_tango);
%End
