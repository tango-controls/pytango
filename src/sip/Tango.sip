/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

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
