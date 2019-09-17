/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <tango.h>
#include <pybind11/pybind11.h>
#include "defs.h"

namespace py = pybind11;

long TANGO_VERSION_HEX;

void export_constants(py::module &m) {
    py::module m2 = m.def_submodule("constants", "module containing several Tango constants.");

    m2.attr("__doc__") = "module containing several Tango constants.\n"
                         "\nNew in PyTango 7.0.0";

#ifdef DISABLE_PYTANGO_NUMPY
    m2.attr("NUMPY_SUPPORT") = false;
    m2.attr("NUMPY_VERSION") = "0.0.0";
#else
    m2.attr("NUMPY_SUPPORT") = true;
#ifdef PYTANGO_NUMPY_VERSION
    m2.attr("NUMPY_VERSION") = PYTANGO_NUMPY_VERSION;
#else
    m2.attr("NUMPY_VERSION") = "0.0.0";
#endif
#endif

    m2.attr("PY_MAJOR_VERSION") = PY_MAJOR_VERSION;
    m2.attr("PY_MINOR_VERSION") = PY_MINOR_VERSION;
    m2.attr("PY_MICRO_VERSION") = PY_MICRO_VERSION;
    m2.attr("PY_VERSION") = PY_VERSION;
    m2.attr("PY_VERSION_HEX") = PY_VERSION_HEX;

//    m2.attr("PYBIND11_MAJOR_VERSION") = PYBIND11_VERSION / 100000;
//    m2.attr("PYBIND11_MINOR_VERSION") = PYBIND11_VERSION / 100 % 1000;
//    m2.attr("PYBIND11_PATCH_VERSION") = PYBIND11_VERSION % 100;

    //
    // From tango_const.h
    //

    //
    // Some general interest define
    //
    m2.attr("TANGO_VERSION_MAJOR") = TANGO_VERSION_MAJOR;
    m2.attr("TANGO_VERSION_MINOR") = TANGO_VERSION_MINOR;
    m2.attr("TANGO_VERSION_PATCH") = TANGO_VERSION_PATCH;
//    m2.attr("TANGO_VERSION_NB") = TANGO_VERSION_NB;
    m2.attr("TANGO_VERSION") = Tango::TgLibVers;

    m2.attr("TgLibVers") = Tango::TgLibVers;
    m2.attr("TgLibMajorVers") = Tango::TgLibMajorVers;
    m2.attr("TgLibVersNb") = Tango::TgLibVersNb;
    m2.attr("DevVersion") = Tango::DevVersion;
    m2.attr("DefaultMaxSeq") = Tango::DefaultMaxSeq;
    m2.attr("DefaultBlackBoxDepth") = Tango::DefaultBlackBoxDepth;
    m2.attr("DefaultPollRingDepth") = Tango::DefaultPollRingDepth;

    m2.attr("InitialOutput") =Tango:: InitialOutput;
    m2.attr("DSDeviceDomain") = Tango::DSDeviceDomain;
    m2.attr("DefaultDocUrl") = Tango::DefaultDocUrl;
    m2.attr("EnvVariable") = Tango::EnvVariable;
    m2.attr("WindowsEnvVariable") = Tango::WindowsEnvVariable;
    m2.attr("DbObjName") = Tango::DbObjName;

    // Changed in tango 8 from DescNotSet to NotSet. We keep the old constant
    // to try to maintain backward compatibility
    m2.attr("DescNotSet") = Tango::NotSet;
    m2.attr("NotSet") = Tango::NotSet;

    m2.attr("ResNotDefined") = Tango::ResNotDefined;
    m2.attr("MessBoxTitle") = Tango::MessBoxTitle;
    m2.attr("StatusNotSet") = Tango::StatusNotSet;
    m2.attr("TangoHostNotSet") = Tango::TangoHostNotSet;
    m2.attr("RootAttNotDef") = Tango::RootAttNotDef;

    m2.attr("DefaultWritAttrProp") = Tango::DefaultWritAttrProp;
    m2.attr("AllAttr") = Tango::AllAttr;
    m2.attr("AllAttr_3") = Tango::AllAttr_3;
    m2.attr("AllPipe") = Tango::AllPipe;
    m2.attr("AllCmd") = Tango::AllCmd;

    m2.attr("PollCommand") = Tango::PollCommand;
    m2.attr("PollAttribute") = Tango::PollAttribute;

    m2.attr("LOCAL_POLL_REQUEST") = Tango::LOCAL_POLL_REQUEST;
    m2.attr("LOCAL_REQUEST_STR_SIZE") = Tango::LOCAL_REQUEST_STR_SIZE;

    m2.attr("MIN_POLL_PERIOD") = Tango::MIN_POLL_PERIOD;
    m2.attr("DELTA_T") = Tango::DELTA_T;
    m2.attr("MIN_DELTA_WORK") = Tango::MIN_DELTA_WORK;
    m2.attr("TIME_HEARTBEAT") = Tango::TIME_HEARTBEAT;
    m2.attr("POLL_LOOP_NB") = Tango::POLL_LOOP_NB;
    m2.attr("ONE_SECOND") = Tango::ONE_SECOND;
    m2.attr("DISCARD_THRESHOLD") = Tango::DISCARD_THRESHOLD;

    m2.attr("DEFAULT_TIMEOUT") = Tango::DEFAULT_TIMEOUT;
    m2.attr("DEFAULT_POLL_OLD_FACTOR") = Tango::DEFAULT_POLL_OLD_FACTOR;

    m2.attr("TG_IMP_MINOR_TO") = Tango::TG_IMP_MINOR_TO;
    m2.attr("TG_IMP_MINOR_DEVFAILED") = Tango::TG_IMP_MINOR_DEVFAILED;
    m2.attr("TG_IMP_MINOR_NON_DEVFAILED") = Tango::TG_IMP_MINOR_NON_DEVFAILED;

    m2.attr("TANGO_PY_MOD_NAME") = Tango::TANGO_PY_MOD_NAME;
    m2.attr("DATABASE_CLASS") = Tango::DATABASE_CLASS;

    m2.attr("TANGO_FLOAT_PRECISION") = Tango::TANGO_FLOAT_PRECISION;
    m2.attr("NoClass") = Tango::NoClass;

    //
    // Event related define
    //

    m2.attr("EVENT_HEARTBEAT_PERIOD") = Tango::EVENT_HEARTBEAT_PERIOD;
    m2.attr("EVENT_RESUBSCRIBE_PERIOD") = Tango::EVENT_RESUBSCRIBE_PERIOD;
    m2.attr("DEFAULT_EVENT_PERIOD") = Tango::DEFAULT_EVENT_PERIOD;
    m2.attr("DELTA_PERIODIC") = Tango::DELTA_PERIODIC;
    m2.attr("DELTA_PERIODIC_LONG") = Tango::DELTA_PERIODIC_LONG;
    m2.attr("HEARTBEAT") = Tango::HEARTBEAT;

    //
    // ZMQ event system related define
    //
    m2.attr("ZMQ_EVENT_PROT_VERSION") = Tango::ZMQ_EVENT_PROT_VERSION;
    m2.attr("HEARTBEAT_METHOD_NAME") = Tango::HEARTBEAT_METHOD_NAME;
    m2.attr("EVENT_METHOD_NAME") = Tango::EVENT_METHOD_NAME;
    m2.attr("HEARTBEAT_EVENT_NAME") = Tango::HEARTBEAT_EVENT_NAME;
    m2.attr("CTRL_SOCK_ENDPOINT") = Tango::CTRL_SOCK_ENDPOINT;
    m2.attr("MCAST_PROT") = Tango::MCAST_PROT;
    m2.attr("MCAST_HOPS") = Tango::MCAST_HOPS;
    m2.attr("PGM_RATE") = Tango::PGM_RATE;
    m2.attr("PGM_IVL") = Tango::PGM_IVL;
    m2.attr("MAX_SOCKET_SUB") = Tango::MAX_SOCKET_SUB;
    m2.attr("PUB_HWM") = Tango::PUB_HWM;
    m2.attr("SUB_HWM") = Tango::SUB_HWM;
    m2.attr("SUB_SEND_HWM") = Tango::SUB_SEND_HWM;

    m2.attr("NOTIFD_CHANNEL") = Tango::NOTIFD_CHANNEL;

    //
    // Locking feature related defines
    //

    m2.attr("DEFAULT_LOCK_VALIDITY") = Tango::DEFAULT_LOCK_VALIDITY;
    m2.attr("DEVICE_UNLOCKED_REASON") = Tango::DEVICE_UNLOCKED_REASON;
    m2.attr("MIN_LOCK_VALIDITY") = Tango::MIN_LOCK_VALIDITY;
    m2.attr("TG_LOCAL_HOST") = Tango::TG_LOCAL_HOST;

    //
    // Client timeout as defined by omniORB4.0.0
    //

    m2.attr("CLNT_TIMEOUT_STR") = Tango::CLNT_TIMEOUT_STR;
    m2.attr("CLNT_TIMEOUT") = Tango::CLNT_TIMEOUT;
    m2.attr("NARROW_CLNT_TIMEOUT") = Tango::NARROW_CLNT_TIMEOUT;

    //
    // Connection and call timeout for database device
    //

    m2.attr("DB_CONNECT_TIMEOUT") = Tango::DB_CONNECT_TIMEOUT;
    m2.attr("DB_RECONNECT_TIMEOUT") = Tango::DB_RECONNECT_TIMEOUT;
    m2.attr("DB_TIMEOUT") = Tango::DB_TIMEOUT;
    m2.attr("DB_START_PHASE_RETRIES") = Tango::DB_START_PHASE_RETRIES;

    //
    // Time to wait before trying to reconnect after
    // a connevtion failure
    //
    m2.attr("RECONNECTION_DELAY") = Tango::RECONNECTION_DELAY;

    //
    // Access Control related defines
    // WARNING: these string are also used within the Db stored procedure
    // introduced in Tango V6.1. If you chang eit here, don't forget to
    // also update the stored procedure
    //

    m2.attr("CONTROL_SYSTEM") = Tango::CONTROL_SYSTEM;
    m2.attr("SERVICE_PROP_NAME") = Tango::SERVICE_PROP_NAME;
    m2.attr("ACCESS_SERVICE") = Tango::ACCESS_SERVICE;

    //
    // Polling threads pool related defines
    //

    m2.attr("DEFAULT_POLLING_THREADS_POOL_SIZE") = Tango::DEFAULT_POLLING_THREADS_POOL_SIZE;

    //
    // Max transfer size 256 MBytes (in byte). Needed by omniORB
    //

    m2.attr("MAX_TRANSFER_SIZE") = Tango::MAX_TRANSFER_SIZE;

    //
    // Max GIOP connection per server . Needed by omniORB
    //

    m2.attr("MAX_GIOP_PER_SERVER") = Tango::MAX_GIOP_PER_SERVER;

    //
    // Tango name length
    //

    m2.attr("MaxServerNameLength") = Tango::MaxServerNameLength;
    m2.attr("MaxDevPropLength") = Tango::MaxDevPropLength;

    //
    // For forwarded attribute implementation
    //
    m2.attr("MIN_IDL_CONF5") = Tango::MIN_IDL_CONF5;
    m2.attr("MIN_IDL_DEV_INTR") = Tango::MIN_IDL_DEV_INTR;
    m2.attr("ALL_EVENTS") = Tango::ALL_EVENTS;

    // --------------------------------------------------------

    //
    // Files used to retrieve env. variables
    //

    m2.attr("USER_ENV_VAR_FILE") = Tango::USER_ENV_VAR_FILE;

    m2.attr("kLogTargetConsole") = Tango::kLogTargetConsole;
    m2.attr("kLogTargetFile") = Tango::kLogTargetFile;
    m2.attr("kLogTargetDevice") = Tango::kLogTargetDevice;
    m2.attr("kLogTargetSep") = Tango::kLogTargetSep;

    m2.attr("AlrmValueNotSpec") = Tango::AlrmValueNotSpec;
    m2.attr("AssocWritNotSpec") = Tango::AssocWritNotSpec;
    m2.attr("LabelNotSpec") = Tango::LabelNotSpec;
    m2.attr("DescNotSpec") = Tango::DescNotSpec;
    m2.attr("UnitNotSpec") = Tango::UnitNotSpec;
    m2.attr("StdUnitNotSpec") = Tango::StdUnitNotSpec;
    m2.attr("DispUnitNotSpec") = Tango::DispUnitNotSpec;
#ifdef FormatNotSpec
    m2.attr("FormatNotSpec") = Tango::FormatNotSpec;
#else
    m2.attr("FormatNotSpec") = Tango::FormatNotSpec_FL;
#endif
    m2.attr("FormatNotSpec_FL") = Tango::FormatNotSpec_FL;
    m2.attr("FormatNotSpec_INT") = Tango::FormatNotSpec_INT;
    m2.attr("FormatNotSpec_STR") = Tango::FormatNotSpec_STR;

    m2.attr("NotANumber") = Tango::NotANumber;
    m2.attr("MemNotUsed") = Tango::MemNotUsed;
    m2.attr("MemAttrPropName") = Tango::MemAttrPropName;

#ifdef TANGO_LONG64
    m2.attr("TANGO_LONG32") = false;
    m2.attr("TANGO_LONG64") = true;
#else
    m2.attr("TANGO_LONG32") = true;
    m2.attr("TANGO_LONG64") = false;
#endif

    m2.attr("API_AttrConfig") = Tango::API_AttrConfig;
    m2.attr("API_AttrEventProp") = Tango::API_AttrEventProp;
    m2.attr("API_AttrIncorrectDataNumber") = Tango::API_AttrIncorrectDataNumber;
    m2.attr("API_AttrNoAlarm") = Tango::API_AttrNoAlarm;
    m2.attr("API_AttrNotAllowed") = Tango::API_AttrNotAllowed;
    m2.attr("API_AttrNotFound") = Tango::API_AttrNotFound;
    m2.attr("API_AttrNotWritable") = Tango::API_AttrNotWritable;
    m2.attr("API_AttrOptProp") = Tango::API_AttrOptProp;
    m2.attr("API_AttrPropValueNotSet") = Tango::API_AttrPropValueNotSet;
    m2.attr("API_AttrValueNotSet") = Tango::API_AttrValueNotSet;
    m2.attr("API_AttrWrongDefined") = Tango::API_AttrWrongDefined;
    m2.attr("API_AttrWrongMemValue") = Tango::API_AttrWrongMemValue;
    m2.attr("API_BadConfigurationProperty") = Tango::API_BadConfigurationProperty;
    m2.attr("API_BlackBoxArgument") = Tango::API_BlackBoxArgument;
    m2.attr("API_BlackBoxEmpty") = Tango::API_BlackBoxEmpty;
    m2.attr("API_CannotCheckAccessControl") = Tango::API_CannotCheckAccessControl;
    m2.attr("API_CannotOpenFile") = Tango::API_CannotOpenFile;
    m2.attr("API_CantActivatePOAManager") = Tango::API_CantActivatePOAManager;
    m2.attr("API_CantCreateClassPoa") = Tango::API_CantCreateClassPoa;
    m2.attr("API_CantCreateLockingThread") = Tango::API_CantCreateLockingThread;
    m2.attr("API_CantFindLockingThread") = Tango::API_CantFindLockingThread;
    m2.attr("API_CantGetClientIdent") = Tango::API_CantGetClientIdent;
    m2.attr("API_CantGetDevObjectId") = Tango::API_CantGetDevObjectId;
    m2.attr("API_CantInstallSignal") = Tango::API_CantInstallSignal;
    m2.attr("API_CantRetrieveClass") = Tango::API_CantRetrieveClass;
    m2.attr("API_CantRetrieveClassList") = Tango::API_CantRetrieveClassList;
    m2.attr("API_CantStoreDeviceClass") = Tango::API_CantStoreDeviceClass;
    m2.attr("API_ClassNotFound") = Tango::API_ClassNotFound;
    m2.attr("API_CmdArgumentTypeNotSupported") = Tango::API_CmdArgumentTypeNotSupported;
    m2.attr("API_CommandNotAllowed") = Tango::API_CommandNotAllowed;
    m2.attr("API_CommandNotFound") = Tango::API_CommandNotFound;
    m2.attr("API_CorbaSysException") = Tango::API_CorbaSysException;
    m2.attr("API_CorruptedDatabase") = Tango::API_CorruptedDatabase;
    m2.attr("API_DatabaseAccess") = Tango::API_DatabaseAccess;
    m2.attr("API_DeviceLocked") = Tango::API_DeviceLocked;
    m2.attr("API_DeviceNotFound") = Tango::API_DeviceNotFound;
    m2.attr("API_DeviceNotLocked") = Tango::API_DeviceNotLocked;
    m2.attr("API_DeviceUnlockable") = Tango::API_DeviceUnlockable;
    m2.attr("API_DeviceUnlocked") = Tango::API_DeviceUnlocked;
    m2.attr("API_EventSupplierNotConstructed") = Tango::API_EventSupplierNotConstructed;
    m2.attr("API_IncoherentDbData") = Tango::API_IncoherentDbData;
    m2.attr("API_IncoherentDevData") = Tango::API_IncoherentDevData;
    m2.attr("API_IncoherentValues") = Tango::API_IncoherentValues;
    m2.attr("API_IncompatibleAttrDataType") = Tango::API_IncompatibleAttrDataType;
    m2.attr("API_IncompatibleCmdArgumentType") = Tango::API_IncompatibleCmdArgumentType;
    m2.attr("API_InitMethodNotFound") = Tango::API_InitMethodNotFound;
    m2.attr("API_InitNotPublic") = Tango::API_InitNotPublic;
    m2.attr("API_InitThrowsException") = Tango::API_InitThrowsException;
    m2.attr("API_JavaRuntimeSecurityException") = Tango::API_JavaRuntimeSecurityException;
    m2.attr("API_MemoryAllocation") = Tango::API_MemoryAllocation;
    m2.attr("API_MethodArgument") = Tango::API_MethodArgument;
    m2.attr("API_MethodNotFound") = Tango::API_MethodNotFound;
    m2.attr("API_MissedEvents") = Tango::API_MissedEvents;
    m2.attr("API_NotSupportedFeature") = Tango::API_NotSupportedFeature;
    m2.attr("API_NtDebugWindowError") = Tango::API_NtDebugWindowError;
    m2.attr("API_OverloadingNotSupported") = Tango::API_OverloadingNotSupported;
    m2.attr("API_PolledDeviceNotInPoolConf") = Tango::API_PolledDeviceNotInPoolConf;
    m2.attr("API_PolledDeviceNotInPoolMap") = Tango::API_PolledDeviceNotInPoolMap;
    m2.attr("API_PollingThreadNotFound") = Tango::API_PollingThreadNotFound;
    m2.attr("API_ReadOnlyMode") = Tango::API_ReadOnlyMode;
    m2.attr("API_SignalOutOfRange") = Tango::API_SignalOutOfRange;
    m2.attr("API_SystemCallFailed") = Tango::API_SystemCallFailed;
    m2.attr("API_WAttrOutsideLimit") = Tango::API_WAttrOutsideLimit;
    m2.attr("API_WizardConfError") = Tango::API_WizardConfError;
    m2.attr("API_WrongEventData") = Tango::API_WrongEventData;
    m2.attr("API_WrongHistoryDataBuffer") = Tango::API_WrongHistoryDataBuffer;
    m2.attr("API_WrongLockingStatus") = Tango::API_WrongLockingStatus;
    m2.attr("API_ZmqInitFailed") = Tango::API_ZmqInitFailed
    ;
}
