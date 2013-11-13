/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include "precompiled_header.hpp"
#include <tango.h>

#define _TOSTRING(s) #s
#define TOSTRING(s) _TOSTRING(s)

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

#ifdef PYTANGO_NUMPY_VERSION
    consts_scope.attr("NUMPY_VERSION") = TOSTRING(PYTANGO_NUMPY_VERSION);
#else
    consts_scope.attr("NUMPY_VERSION") = "0.0.0";
#endif


    consts_scope.attr("PY_MAJOR_VERSION") = PY_MAJOR_VERSION;
    consts_scope.attr("PY_MINOR_VERSION") = PY_MINOR_VERSION;
    consts_scope.attr("PY_MICRO_VERSION") = PY_MICRO_VERSION;
    consts_scope.attr("PY_VERSION") = PY_VERSION;
    consts_scope.attr("PY_VERSION_HEX") = PY_VERSION_HEX;

    consts_scope.attr("BOOST_MAJOR_VERSION") = BOOST_VERSION / 100000;
    consts_scope.attr("BOOST_MINOR_VERSION") = BOOST_VERSION / 100 % 1000;
    consts_scope.attr("BOOST_PATCH_VERSION") = BOOST_VERSION % 100;
    // missing BOOST_VERSION => do it in python


    //
    // From tango_const.h
    //
    
    //
    // Some general interest define
    //
    consts_scope.attr("TANGO_VERSION_MAJOR") = TANGO_VERSION_MAJOR;
    consts_scope.attr("TANGO_VERSION_MINOR") = TANGO_VERSION_MINOR;
    consts_scope.attr("TANGO_VERSION_PATCH") = TANGO_VERSION_PATCH;
    consts_scope.attr("TANGO_VERSION") = TgLibVers;

    consts_scope.attr("TgLibVers") = TgLibVers;
    consts_scope.attr("TgLibVersNb") = TgLibVersNb;
    consts_scope.attr("DevVersion") = DevVersion;
    consts_scope.attr("DefaultMaxSeq") = DefaultMaxSeq;
    consts_scope.attr("DefaultBlackBoxDepth") = DefaultBlackBoxDepth;
    consts_scope.attr("DefaultPollRingDepth") = DefaultPollRingDepth;
    
    consts_scope.attr("InitialOutput") = InitialOutput;
    consts_scope.attr("DSDeviceDomain") = DSDeviceDomain;
    consts_scope.attr("DefaultDocUrl") = DefaultDocUrl;
    consts_scope.attr("EnvVariable") = EnvVariable;
    consts_scope.attr("DbObjName") = DbObjName;
    
    // Changed in tango 8 from DescNotSet to NotSet. We keep the old constant
    // to try to maintain backward compatibility
    consts_scope.attr("DescNotSet") = NotSet;
    consts_scope.attr("NotSet") = NotSet;
    
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
#ifdef FormatNotSpec
    consts_scope.attr("FormatNotSpec") = FormatNotSpec;
#else
    consts_scope.attr("FormatNotSpec") = FormatNotSpec_FL;
#endif
    consts_scope.attr("FormatNotSpec_FL") = FormatNotSpec_FL;
    consts_scope.attr("FormatNotSpec_INT") = FormatNotSpec_INT;
    consts_scope.attr("FormatNotSpec_STR") = FormatNotSpec_STR;
    
    consts_scope.attr("NotANumber") = NotANumber;
    consts_scope.attr("MemNotUsed") = MemNotUsed;
    consts_scope.attr("MemAttrPropName") = MemAttrPropName;
 
#ifdef TANGO_LONG64
    consts_scope.attr("TANGO_LONG32") = false;
    consts_scope.attr("TANGO_LONG64") = true;
#else
    consts_scope.attr("TANGO_LONG32") = true;
    consts_scope.attr("TANGO_LONG64") = false;
#endif    

    consts_scope.attr("API_AttrConfig") = API_AttrConfig;
    consts_scope.attr("API_AttrEventProp") = API_AttrEventProp;
    consts_scope.attr("API_AttrIncorrectDataNumber") = API_AttrIncorrectDataNumber;
    consts_scope.attr("API_AttrNoAlarm") = API_AttrNoAlarm;
    consts_scope.attr("API_AttrNotAllowed") = API_AttrNotAllowed;
    consts_scope.attr("API_AttrNotFound") = API_AttrNotFound;
    consts_scope.attr("API_AttrNotWritable") = API_AttrNotWritable;
    consts_scope.attr("API_AttrOptProp") = API_AttrOptProp;
    consts_scope.attr("API_AttrPropValueNotSet") = API_AttrPropValueNotSet;
    consts_scope.attr("API_AttrValueNotSet") = API_AttrValueNotSet;
    consts_scope.attr("API_AttrWrongDefined") = API_AttrWrongDefined;
    consts_scope.attr("API_AttrWrongMemValue") = API_AttrWrongMemValue;
    consts_scope.attr("API_BadConfigurationProperty") = API_BadConfigurationProperty;
    consts_scope.attr("API_BlackBoxArgument") = API_BlackBoxArgument;
    consts_scope.attr("API_BlackBoxEmpty") = API_BlackBoxEmpty;
    consts_scope.attr("API_CannotCheckAccessControl") = API_CannotCheckAccessControl;
    consts_scope.attr("API_CannotOpenFile") = API_CannotOpenFile;
    consts_scope.attr("API_CantActivatePOAManager") = API_CantActivatePOAManager;
    consts_scope.attr("API_CantCreateClassPoa") = API_CantCreateClassPoa;
    consts_scope.attr("API_CantCreateLockingThread") = API_CantCreateLockingThread;
    consts_scope.attr("API_CantFindLockingThread") = API_CantFindLockingThread;
    consts_scope.attr("API_CantGetClientIdent") = API_CantGetClientIdent;
    consts_scope.attr("API_CantGetDevObjectId") = API_CantGetDevObjectId;
    consts_scope.attr("API_CantInstallSignal") = API_CantInstallSignal;
    consts_scope.attr("API_CantRetrieveClass") = API_CantRetrieveClass;
    consts_scope.attr("API_CantRetrieveClassList") = API_CantRetrieveClassList;
    consts_scope.attr("API_CantStoreDeviceClass") = API_CantStoreDeviceClass;
    consts_scope.attr("API_ClassNotFound") = API_ClassNotFound;
    consts_scope.attr("API_CmdArgumentTypeNotSupported") = API_CmdArgumentTypeNotSupported;
    consts_scope.attr("API_CommandNotAllowed") = API_CommandNotAllowed;
    consts_scope.attr("API_CommandNotFound") = API_CommandNotFound;
    consts_scope.attr("API_CorbaSysException") = API_CorbaSysException;
    consts_scope.attr("API_CorruptedDatabase") = API_CorruptedDatabase;
    consts_scope.attr("API_DatabaseAccess") = API_DatabaseAccess;
    consts_scope.attr("API_DeviceLocked") = API_DeviceLocked;
    consts_scope.attr("API_DeviceNotFound") = API_DeviceNotFound;
    consts_scope.attr("API_DeviceNotLocked") = API_DeviceNotLocked;
    consts_scope.attr("API_DeviceUnlockable") = API_DeviceUnlockable;
    consts_scope.attr("API_DeviceUnlocked") = API_DeviceUnlocked;
    consts_scope.attr("API_EventSupplierNotConstructed") = API_EventSupplierNotConstructed;
    consts_scope.attr("API_IncoherentDbData") = API_IncoherentDbData;
    consts_scope.attr("API_IncoherentDevData") = API_IncoherentDevData;
    consts_scope.attr("API_IncoherentValues") = API_IncoherentValues;
    consts_scope.attr("API_IncompatibleAttrDataType") = API_IncompatibleAttrDataType;
    consts_scope.attr("API_IncompatibleCmdArgumentType") = API_IncompatibleCmdArgumentType;
    consts_scope.attr("API_InitMethodNotFound") = API_InitMethodNotFound;
    consts_scope.attr("API_InitNotPublic") = API_InitNotPublic;
    consts_scope.attr("API_InitThrowsException") = API_InitThrowsException;
    consts_scope.attr("API_JavaRuntimeSecurityException") = API_JavaRuntimeSecurityException;
    consts_scope.attr("API_MemoryAllocation") = API_MemoryAllocation;
    consts_scope.attr("API_MethodArgument") = API_MethodArgument;
    consts_scope.attr("API_MethodNotFound") = API_MethodNotFound;
    consts_scope.attr("API_MissedEvents") = API_MissedEvents;
    consts_scope.attr("API_NotSupportedFeature") = API_NotSupportedFeature;
    consts_scope.attr("API_NtDebugWindowError") = API_NtDebugWindowError;
    consts_scope.attr("API_OverloadingNotSupported") = API_OverloadingNotSupported;
    consts_scope.attr("API_PolledDeviceNotInPoolConf") = API_PolledDeviceNotInPoolConf;
    consts_scope.attr("API_PolledDeviceNotInPoolMap") = API_PolledDeviceNotInPoolMap;
    consts_scope.attr("API_PollingThreadNotFound") = API_PollingThreadNotFound;
    consts_scope.attr("API_ReadOnlyMode") = API_ReadOnlyMode;
    consts_scope.attr("API_SignalOutOfRange") = API_SignalOutOfRange;
    consts_scope.attr("API_SystemCallFailed") = API_SystemCallFailed;
    consts_scope.attr("API_WAttrOutsideLimit") = API_WAttrOutsideLimit;
    consts_scope.attr("API_WizardConfError") = API_WizardConfError;
    consts_scope.attr("API_WrongEventData") = API_WrongEventData;
    consts_scope.attr("API_WrongHistoryDataBuffer") = API_WrongHistoryDataBuffer;
    consts_scope.attr("API_WrongLockingStatus") = API_WrongLockingStatus;
    consts_scope.attr("API_ZmqInitFailed") = API_ZmqInitFailed;

}
