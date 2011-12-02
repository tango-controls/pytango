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

#include "precompiled_header.hpp"
#include <tango.h>

using namespace boost::python;

long TANGO_VERSION_HEX;

void export_constants()
{
    object consts_module(handle<>(borrowed(PyImport_AddModule("PyTango.constants"))));
    scope().attr("constants") = consts_module;
    scope consts_scope = consts_module;
    
    consts_scope.attr("__doc__") = "module containing several Tango constants.\n"
        "\nNew in PyTango 7.0.0";

#ifdef DISABLE_PYTANGO_NUMPY
    consts_scope.attr("NUMPY_SUPPORT") = false;
#else
    consts_scope.attr("NUMPY_SUPPORT") = true;
#endif

    str py_TgLibVers = TgLibVers;
    boost::python::list pylist_TgLibVers = py_TgLibVers.split(".");
    long_ major = long_(pylist_TgLibVers[0]);
    long_ minor = long_(pylist_TgLibVers[1]);
    long_ patch = long_(pylist_TgLibVers[2]);
    object h = "0x%02d%02d%02d00" % boost::python::make_tuple(major, minor, patch);
    PyObject *ptr = PyInt_FromString(PyString_AsString(h.ptr()), NULL, 0);
    TANGO_VERSION_HEX = PyInt_AsLong(ptr);
    Py_DECREF(ptr);
    consts_scope.attr("TANGO_VERSION_HEX") = TANGO_VERSION_HEX;

    //
    // From tango_const.h
    //
    
    //
    // Some general interest define
    //
    
    consts_scope.attr("TgLibVers") = TgLibVers;
    consts_scope.attr("DevVersion") = DevVersion;
    consts_scope.attr("DefaultMaxSeq") = DefaultMaxSeq;
    consts_scope.attr("DefaultBlackBoxDepth") = DefaultBlackBoxDepth;
    consts_scope.attr("DefaultPollRingDepth") = DefaultPollRingDepth;
    
    consts_scope.attr("InitialOutput") = InitialOutput;
    consts_scope.attr("DSDeviceDomain") = DSDeviceDomain;
    consts_scope.attr("DefaultDocUrl") = DefaultDocUrl;
    consts_scope.attr("EnvVariable") = EnvVariable;
    consts_scope.attr("DbObjName") = DbObjName;
    consts_scope.attr("DescNotSet") = DescNotSet;
    consts_scope.attr("ResNotDefined") = ResNotDefined;
    consts_scope.attr("MessBoxTitle") = MessBoxTitle;
    consts_scope.attr("StatusNotSet") = StatusNotSet;
    
    consts_scope.attr("DefaultWritAttrProp") = DefaultWritAttrProp;
    consts_scope.attr("AllAttr") = AllAttr;
    consts_scope.attr("AllAttr_3") = AllAttr_3;
    
    consts_scope.attr("PollCommand") = PollCommand;
    consts_scope.attr("PollAttribute") = PollAttribute;
    
    consts_scope.attr("MIN_POLL_PERIOD") = MIN_POLL_PERIOD;
    consts_scope.attr("DELTA_T") = DELTA_T;
    consts_scope.attr("MIN_DELTA_WORK") = MIN_DELTA_WORK;
    consts_scope.attr("TIME_HEARTBEAT") = TIME_HEARTBEAT;
    consts_scope.attr("POLL_LOOP_NB") = POLL_LOOP_NB;
    consts_scope.attr("ONE_SECOND") = ONE_SECOND;
    consts_scope.attr("DISCARD_THRESHOLD") = DISCARD_THRESHOLD;
    
    consts_scope.attr("DEFAULT_TIMEOUT") = DEFAULT_TIMEOUT;
    consts_scope.attr("DEFAULT_POLL_OLD_FACTOR") = DEFAULT_POLL_OLD_FACTOR;
    
    consts_scope.attr("TG_IMP_MINOR_TO") = TG_IMP_MINOR_TO;
    consts_scope.attr("TG_IMP_MINOR_DEVFAILED") = TG_IMP_MINOR_DEVFAILED;
    consts_scope.attr("TG_IMP_MINOR_NON_DEVFAILED") = TG_IMP_MINOR_NON_DEVFAILED;
    
    consts_scope.attr("TANGO_PY_MOD_NAME") = TANGO_PY_MOD_NAME;
    consts_scope.attr("DATABASE_CLASS") = DATABASE_CLASS;
    
    //
    // Event related define
    //
    
    consts_scope.attr("EVENT_HEARTBEAT_PERIOD") = EVENT_HEARTBEAT_PERIOD;
    consts_scope.attr("EVENT_RESUBSCRIBE_PERIOD") = EVENT_RESUBSCRIBE_PERIOD;
    consts_scope.attr("DEFAULT_EVENT_PERIOD") = DEFAULT_EVENT_PERIOD;
    consts_scope.attr("DELTA_PERIODIC") = DELTA_PERIODIC;
    consts_scope.attr("DELTA_PERIODIC_LONG") = DELTA_PERIODIC_LONG;
    consts_scope.attr("HEARTBEAT") = HEARTBEAT;
    
    //
    // Locking feature related defines
    //
    
    consts_scope.attr("DEFAULT_LOCK_VALIDITY") = DEFAULT_LOCK_VALIDITY;
    consts_scope.attr("DEVICE_UNLOCKED_REASON") = DEVICE_UNLOCKED_REASON;
    consts_scope.attr("MIN_LOCK_VALIDITY") = MIN_LOCK_VALIDITY;
    
    //
    // Client timeout as defined by omniORB4.0.0
    //
    
    consts_scope.attr("CLNT_TIMEOUT_STR") = CLNT_TIMEOUT_STR;
    consts_scope.attr("CLNT_TIMEOUT") = CLNT_TIMEOUT;
    
    //
    // Connection and call timeout for database device
    //
    
    consts_scope.attr("DB_CONNECT_TIMEOUT") = DB_CONNECT_TIMEOUT;
    consts_scope.attr("DB_RECONNECT_TIMEOUT") = DB_RECONNECT_TIMEOUT;
    consts_scope.attr("DB_TIMEOUT") = DB_TIMEOUT;
    consts_scope.attr("DB_START_PHASE_RETRIES") = DB_START_PHASE_RETRIES;
    
    //
    // Time to wait before trying to reconnect after
    // a connevtion failure
    //
    consts_scope.attr("RECONNECTION_DELAY") = RECONNECTION_DELAY;
    
    //
    // Access Control related defines
    // WARNING: these string are also used within the Db stored procedure
    // introduced in Tango V6.1. If you chang eit here, don't forget to
    // also update the stored procedure
    //
    
    consts_scope.attr("CONTROL_SYSTEM") = CONTROL_SYSTEM;
    consts_scope.attr("SERVICE_PROP_NAME") = SERVICE_PROP_NAME;
    consts_scope.attr("ACCESS_SERVICE") = ACCESS_SERVICE;
    
    //
    // Polling threads pool related defines
    //
    
    consts_scope.attr("DEFAULT_POLLING_THREADS_POOL_SIZE") = DEFAULT_POLLING_THREADS_POOL_SIZE;
    
    //
    // Max transfer size 256 MBytes (in byte). Needed by omniORB
    //
    
    consts_scope.attr("MAX_TRANSFER_SIZE") = MAX_TRANSFER_SIZE;
    
    //
    // Tango name length
    //
    
    consts_scope.attr("MaxServerNameLength") = MaxServerNameLength;
    
    //
    // Files used to retrieve env. variables
    //
    
    consts_scope.attr("USER_ENV_VAR_FILE") = USER_ENV_VAR_FILE;
    
    consts_scope.attr("kLogTargetConsole") = kLogTargetConsole;
    consts_scope.attr("kLogTargetFile") = kLogTargetFile;
    consts_scope.attr("kLogTargetDevice") = kLogTargetDevice;
    consts_scope.attr("kLogTargetSep") = kLogTargetSep;
    
    consts_scope.attr("AlrmValueNotSpec") = AlrmValueNotSpec;
    consts_scope.attr("AssocWritNotSpec") = AssocWritNotSpec;
    consts_scope.attr("LabelNotSpec") = LabelNotSpec;
    consts_scope.attr("DescNotSpec") = DescNotSpec;
    consts_scope.attr("UnitNotSpec") = UnitNotSpec;
    consts_scope.attr("StdUnitNotSpec") = StdUnitNotSpec;
    consts_scope.attr("DispUnitNotSpec") = DispUnitNotSpec;
    consts_scope.attr("FormatNotSpec") = FormatNotSpec;
    consts_scope.attr("NotANumber") = NotANumber;
    consts_scope.attr("MemNotUsed") = MemNotUsed;
    consts_scope.attr("MemAttrPropName") = MemAttrPropName;
}
