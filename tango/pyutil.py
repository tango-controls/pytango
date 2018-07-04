# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

"""
This is an internal PyTango module.
"""

__all__ = ("Util", "pyutil_init")

__docformat__ = "restructuredtext"

import os
import copy

from ._tango import Util, Except, DevFailed, DbDevInfo
from .utils import document_method as __document_method
# from utils import document_static_method as __document_static_method
from .globals import class_list, cpp_class_list, get_constructed_classes
import collections


def __simplify_device_name(dev_name):
    if dev_name.startswith("tango://"):
        dev_name = dev_name[8:]
    if dev_name.count("/") > 2:
        dev_name = dev_name[dev_name.index("/") + 1:]
    return dev_name.lower()


#
# Methods on Util
#

def __Util__get_class_list(self):
    """
        get_class_list(self) -> seq<DeviceClass>

                Returns a list of objects of inheriting from DeviceClass

            Parameters : None

            Return     : (seq<DeviceClass>) a list of objects of inheriting from DeviceClass"""
    return get_constructed_classes()


def __Util__create_device(self, klass_name, device_name, alias=None, cb=None):
    """
        create_device(self, klass_name, device_name, alias=None, cb=None) -> None

            Creates a new device of the given class in the database, creates a new
            DeviceImpl for it and calls init_device (just like it is done for
            existing devices when the DS starts up)

            An optional parameter callback is called AFTER the device is
            registered in the database and BEFORE the init_device for the
            newly created device is called

            Throws tango.DevFailed:
                - the device name exists already or
                - the given class is not registered for this DS.
                - the cb is not a callable

        New in PyTango 7.1.2

        Parameters :
            - klass_name : (str) the device class name
            - device_name : (str) the device name
            - alias : (str) optional alias. Default value is None meaning do not create device alias
            - cb : (callable) a callback that is called AFTER the device is registered
                   in the database and BEFORE the init_device for the newly created
                   device is called. Typically you may want to put device and/or attribute
                   properties in the database here. The callback must receive a parameter:
                   device name (str). Default value is None meaning no callback

        Return     : None"""
    if cb is not None and not isinstance(cb, collections.Callable):
        Except.throw_exception("PyAPI_InvalidParameter",
                               "The optional cb parameter must be a python callable",
                               "Util.create_device")

    db = self.get_database()

    device_name = __simplify_device_name(device_name)

    device_exists = True
    try:
        db.import_device(device_name)
    except DevFailed as df:
        device_exists = not df[0].reason == "DB_DeviceNotDefined"

    # 1 - Make sure device name doesn't exist already in the database
    if device_exists:
        Except.throw_exception("PyAPI_DeviceAlreadyDefined",
                               "The device %s is already defined in the database" % device_name,
                               "Util.create_device")

    # 2 - Make sure the device class is known
    klass_list = self.get_class_list()
    klass = None
    for k in klass_list:
        name = k.get_name()
        if name == klass_name:
            klass = k
            break
    if klass is None:
        Except.throw_exception("PyAPI_UnknownDeviceClass",
                               "The device class %s could not be found" % klass_name,
                               "Util.create_device")

    # 3 - Create entry in the database (with alias if necessary)
    dev_info = DbDevInfo()
    dev_info.name = device_name
    dev_info._class = klass_name
    dev_info.server = self.get_ds_name()

    db.add_device(dev_info)

    if alias is not None:
        db.put_device_alias(device_name, alias)

    # from this point on, if anything wrong happens we need to clean the database
    try:
        # 4 - run the callback which tipically is used to initialize
        #     device and/or attribute properties in the database
        if cb is not None:
            cb(device_name)

        # 5 - Initialize device object on this server
        k.device_factory([device_name])
    except:
        try:
            if alias is not None:
                db.delete_device_alias(alias)
        except:
            pass
        db.delete_device(device_name)


def __Util__delete_device(self, klass_name, device_name):
    """
        delete_device(self, klass_name, device_name) -> None

            Deletes an existing device from the database and from this running
            server

            Throws tango.DevFailed:
                - the device name doesn't exist in the database
                - the device name doesn't exist in this DS.

        New in PyTango 7.1.2

        Parameters :
            - klass_name : (str) the device class name
            - device_name : (str) the device name

        Return     : None"""

    db = self.get_database()
    device_name = __simplify_device_name(device_name)
    device_exists = True
    try:
        db.import_device(device_name)
    except DevFailed as df:
        device_exists = not df[0].reason == "DB_DeviceNotDefined"

    # 1 - Make sure device name exists in the database
    if not device_exists:
        Except.throw_exception("PyAPI_DeviceNotDefined",
                               "The device %s is not defined in the database" % device_name,
                               "Util.delete_device")

    # 2 - Make sure device name is defined in this server
    class_device_name = "%s::%s" % (klass_name, device_name)
    ds = self.get_dserver_device()
    dev_names = ds.query_device()
    device_exists = False
    for dev_name in dev_names:
        p = dev_name.index("::")
        dev_name = dev_name[:p] + dev_name[p:].lower()
        if dev_name == class_device_name:
            device_exists = True
            break
    if not device_exists:
        Except.throw_exception("PyAPI_DeviceNotDefinedInServer",
                               "The device %s is not defined in this server" % class_device_name,
                               "Util.delete_device")

    db.delete_device(device_name)

    dimpl = self.get_device_by_name(device_name)

    dc = dimpl.get_device_class()
    dc.device_destroyer(device_name)


def __Util__init__(self, args):
    args = copy.copy(args)
    args[0] = os.path.splitext(args[0])[0]
    Util.__init_orig__(self, args)


def __Util__add_TgClass(self, klass_device_class, klass_device,
                        device_class_name=None):
    """Register a new python tango class. Example::

           util.add_TgClass(MotorClass, Motor)
           util.add_TgClass(MotorClass, Motor, 'Motor') # equivalent to previous line

       .. deprecated:: 7.1.2
           Use :meth:`tango.Util.add_class` instead."""
    if device_class_name is None:
        device_class_name = klass_device.__name__
    class_list.append((klass_device_class, klass_device, device_class_name))


def __Util__add_Cpp_TgClass(self, device_class_name, tango_device_class_name):
    """Register a new C++ tango class.

       If there is a shared library file called MotorClass.so which
       contains a MotorClass class and a _create_MotorClass_class method. Example::

           util.add_Cpp_TgClass('MotorClass', 'Motor')

       .. note:: the parameter 'device_class_name' must match the shared
                 library name.

       .. deprecated:: 7.1.2
           Use :meth:`tango.Util.add_class` instead."""
    cpp_class_list.append((device_class_name, tango_device_class_name))


def __Util__add_class(self, *args, **kwargs):
    """
        add_class(self, class<DeviceClass>, class<DeviceImpl>, language="python") -> None

            Register a new tango class ('python' or 'c++').

            If language is 'python' then args must be the same as
            :meth:`tango.Util.add_TgClass`. Otherwise, args should be the ones
            in :meth:`tango.Util.add_Cpp_TgClass`. Example::

                util.add_class(MotorClass, Motor)
                util.add_class('CounterClass', 'Counter', language='c++')

        New in PyTango 7.1.2"""
    language = kwargs.get("language", "python")
    f = self.add_TgClass
    if language != "python":
        f = f = self.add_Cpp_TgClass
    return f(*args)


def __init_Util():
    Util.__init_orig__ = Util.__init__
    Util.__init__ = __Util__init__
    Util.add_TgClass = __Util__add_TgClass
    Util.add_Cpp_TgClass = __Util__add_Cpp_TgClass
    Util.add_class = __Util__add_class
    Util.get_class_list = __Util__get_class_list
    Util.create_device = __Util__create_device
    Util.delete_device = __Util__delete_device


def __doc_Util():
    Util.__doc__ = """\
    This class is a used to store TANGO device server process data and to
    provide the user with a set of utilities method.

    This class is implemented using the singleton design pattern.
    Therefore a device server process can have only one instance of this
    class and its constructor is not public. Example::

        util = tango.Util.instance()
            print(util.get_host_name())
    """

    def document_method(method_name, desc, append=True):
        return __document_method(Util, method_name, desc, append)

    #    def document_static_method(method_name, desc, append=True):
    #        return __document_static_method(_Util, method_name, desc, append)

    #    document_static_method("instance", """
    #    instance(exit = True) -> Util
    #
    #            Static method that gets the singleton object reference.
    #            If the class has not been initialised with it's init method,
    #            this method prints a message and aborts the device server process
    #
    #        Parameters :
    #            - exit : (bool)
    #
    #        Return     : (Util) the tango Util object
    #    """ )

    document_method("set_trace_level", """
    set_trace_level(self, level) -> None

            Set the process trace level.

        Parameters :
            - level : (int) the new process level
        Return     : None
    """)

    document_method("get_trace_level", """
    get_trace_level(self) -> int

            Get the process trace level.

        Parameters : None
        Return     : (int) the process trace level.
    """)

    document_method("get_ds_inst_name", """
    get_ds_inst_name(self) -> str

            Get a COPY of the device server instance name.

        Parameters : None
        Return     : (str) a COPY of the device server instance name.

        New in PyTango 3.0.4
    """)

    document_method("get_ds_exec_name", """
    get_ds_exec_name(self) -> str

            Get a COPY of the device server executable name.

        Parameters : None
        Return     : (str) a COPY of the device server executable name.

        New in PyTango 3.0.4
    """)

    document_method("get_ds_name", """
    get_ds_name(self) -> str

            Get the device server name.
            The device server name is the <device server executable name>/<the device server instance name>

        Parameters : None
        Return     : (str) device server name

        New in PyTango 3.0.4
    """)

    document_method("get_host_name", """
    get_host_name(self) -> str

            Get the host name where the device server process is running.

        Parameters : None
        Return     : (str) the host name where the device server process is running

        New in PyTango 3.0.4
    """)

    document_method("get_pid_str", """
    get_pid_str(self) -> str

            Get the device server process identifier as a string.

        Parameters : None
        Return     : (str) the device server process identifier as a string

        New in PyTango 3.0.4
    """)

    document_method("get_pid", """
    get_pid(self) -> TangoSys_Pid

            Get the device server process identifier.

        Parameters : None
        Return     : (int) the device server process identifier
    """)

    document_method("get_tango_lib_release", """
    get_tango_lib_release(self) -> int

            Get the TANGO library version number.

        Parameters : None
        Return     : (int) The Tango library release number coded in
                     3 digits (for instance 550,551,552,600,....)
    """)

    document_method("get_version_str", """
    get_version_str(self) -> str

            Get the IDL TANGO version.

        Parameters : None
        Return     : (str) the IDL TANGO version.

        New in PyTango 3.0.4
    """)

    document_method("get_server_version", """
    get_server_version(self) -> str

            Get the device server version.

        Parameters : None
        Return     : (str) the device server version.
    """)

    document_method("set_server_version", """
    set_server_version(self, vers) -> None

            Set the device server version.

        Parameters :
            - vers : (str) the device server version
        Return     : None
    """)

    document_method("set_serial_model", """
    set_serial_model(self, ser) -> None

            Set the serialization model.

        Parameters :
            - ser : (SerialModel) the new serialization model. The serialization model must
                    be one of BY_DEVICE, BY_CLASS, BY_PROCESS or NO_SYNC
        Return     : None
    """)

    document_method("get_serial_model", """
    get_serial_model(self) ->SerialModel

            Get the serialization model.

        Parameters : None
        Return     : (SerialModel) the serialization model
    """)

    document_method("connect_db", """
    connect_db(self) -> None

            Connect the process to the TANGO database.
            If the connection to the database failed, a message is
            displayed on the screen and the process is aborted

        Parameters : None
        Return     : None
    """)

    document_method("reset_filedatabase", """
    reset_filedatabase(self) -> None

            Reread the file database.

        Parameters : None
        Return     : None
    """)

    document_method("unregister_server", """
    unregister_server(self) -> None

            Unregister a device server process from the TANGO database.

        Parameters : None
        Return     : None
    """)

    document_method("get_dserver_device", """
    get_dserver_device(self) -> DServer

            Get a reference to the dserver device attached to the device server process.

        Parameters : None
        Return     : (DServer) the dserver device attached to the device server process

        New in PyTango 7.0.0
    """)

    document_method("server_init", """
    server_init(self, with_window = False) -> None

            Initialize all the device server pattern(s) embedded in a device server process.

        Parameters :
            - with_window : (bool) default value is False
        Return     : None

        Throws     : DevFailed If the device pattern initialistaion failed
    """)

    document_method("server_run", """
    server_run(self) -> None

            Run the CORBA event loop.
            This method runs the CORBA event loop. For UNIX or Linux operating system,
            this method does not return. For Windows in a non-console mode, this method
            start a thread which enter the CORBA event loop.

        Parameters : None
        Return     : None
    """)

    document_method("trigger_cmd_polling", """
    trigger_cmd_polling(self, dev, name) -> None

            Trigger polling for polled command.
            This method send the order to the polling thread to poll one object registered
            with an update period defined as "externally triggerred"

        Parameters :
            - dev : (DeviceImpl) the TANGO device
            - name : (str) the command name which must be polled
        Return     : None

        Throws     : DevFailed If the call failed
    """)

    document_method("trigger_attr_polling", """
    trigger_attr_polling(self, dev, name) -> None

            Trigger polling for polled attribute.
            This method send the order to the polling thread to poll one object registered
            with an update period defined as "externally triggerred"

        Parameters :
            - dev : (DeviceImpl) the TANGO device
            - name : (str) the attribute name which must be polled
        Return     : None
    """)

    document_method("set_polling_threads_pool_size", """
    set_polling_threads_pool_size(self, thread_nb) -> None

            Set the polling threads pool size.

        Parameters :
            - thread_nb : (int) the maximun number of threads in the polling threads pool
        Return     : None

        New in PyTango 7.0.0
    """)

    document_method("get_polling_threads_pool_size", """
    get_polling_threads_pool_size(self) -> int

            Get the polling threads pool size.

        Parameters : None
        Return     : (int) the maximun number of threads in the polling threads pool
    """)

    document_method("is_svr_starting", """
    is_svr_starting(self) -> bool

            Check if the device server process is in its starting phase

        Parameters : None
        Return     : (bool) True if the server is in its starting phase

        New in PyTango 8.0.0
    """)

    document_method("is_svr_shutting_down", """
    is_svr_shutting_down(self) -> bool

            Check if the device server process is in its shutting down sequence

        Parameters : None
        Return     : (bool) True if the server is in its shutting down phase.

        New in PyTango 8.0.0
    """)

    document_method("is_device_restarting", """
    is_device_restarting(self, (str)dev_name) -> bool

            Check if the device is actually restarted by the device server
            process admin device with its DevRestart command

        Parameters :
            dev_name : (str) device name
        Return     : (bool) True if the device is restarting.

        New in PyTango 8.0.0
    """)

    document_method("get_sub_dev_diag", """
    get_sub_dev_diag(self) -> SubDevDiag

            Get the internal sub device manager

        Parameters : None
        Return     : (SubDevDiag) the sub device manager

        New in PyTango 7.0.0
    """)

    document_method("reset_filedatabase", """
    reset_filedatabase(self) -> None

            Reread the file database

        Parameters : None
        Return     : None

        New in PyTango 7.0.0
    """)

    document_method("get_database", """
    get_database(self) -> Database

            Get a reference to the TANGO database object

        Parameters : None
        Return     : (Database) the database

        New in PyTango 7.0.0
    """)

    document_method("unregister_server", """
    unregister_server(self) -> None

            Unregister a device server process from the TANGO database.
            If the database call fails, a message is displayed on the screen
            and the process is aborted

        Parameters : None
        Return     : None

        New in PyTango 7.0.0
    """)

    document_method("get_device_list_by_class", """
    get_device_list_by_class(self, class_name) -> sequence<DeviceImpl>

            Get the list of device references for a given TANGO class.
            Return the list of references for all devices served by one implementation
            of the TANGO device pattern implemented in the process.

        Parameters :
            - class_name : (str) The TANGO device class name

        Return     : (sequence<DeviceImpl>) The device reference list

        New in PyTango 7.0.0
    """)

    document_method("get_device_by_name", """
    get_device_by_name(self, dev_name) -> DeviceImpl

            Get a device reference from its name

        Parameters :
            - dev_name : (str) The TANGO device name
        Return     : (DeviceImpl) The device reference

        New in PyTango 7.0.0
    """)

    document_method("get_dserver_device", """
    get_dserver_device(self) -> DServer

            Get a reference to the dserver device attached to the device server process

        Parameters : None
        Return     : (DServer) A reference to the dserver device

        New in PyTango 7.0.0
    """)

    document_method("get_device_list", """
    get_device_list(self) -> sequence<DeviceImpl>

            Get device list from name.
            It is possible to use a wild card ('*') in the name parameter
            (e.g. "*", "/tango/tangotest/n*", ...)

        Parameters : None
        Return     : (sequence<DeviceImpl>) the list of device objects

        New in PyTango 7.0.0
    """)

    document_method("server_set_event_loop", """
    server_set_event_loop(self, event_loop) -> None

        This method registers an event loop function in a Tango server.
        This function will be called by the process main thread in an infinite loop
        The process will not use the classical ORB blocking event loop.
        It is the user responsability to code this function in a way that it implements
        some kind of blocking in order not to load the computer CPU. The following
        piece of code is an example of how you can use this feature::

            _LOOP_NB = 1
            def looping():
                global _LOOP_NB
                print "looping", _LOOP_NB
                time.sleep(0.1)
                _LOOP_NB += 1
                return _LOOP_NB > 100

            def main():
                py = tango.Util(sys.argv)

                # ...

                U = tango.Util.instance()
                U.server_set_event_loop(looping)
                U.server_init()
                U.server_run()

        Parameters : None
        Return     : None

        New in PyTango 8.1.0
    """)


#    document_static_method("init_python", """
#    init_python() -> None
#
#            Static method
#            For internal usage.
#
#        Parameters : None
#        Return     : None
#    """ )

def pyutil_init(doc=True):
    __init_Util()
    if doc:
        __doc_Util()
