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

from __future__ import print_function

import copy

from ._tango import (
    DeviceImpl, Device_3Impl, Device_4Impl, Device_5Impl,
    DevFailed, Attribute, WAttribute,
    MultiAttribute, MultiClassAttribute,
    Attr, Logger, AttrWriteType, AttrDataFormat,
    DispLevel, UserDefaultAttrProp, StdStringVector)

from .utils import document_method as __document_method
from .utils import copy_doc, get_latest_device_class
from .attr_data import AttrData

from .log4tango import TangoStream

__docformat__ = "restructuredtext"

__all__ = ("ChangeEventProp", "PeriodicEventProp",
           "ArchiveEventProp", "AttributeAlarm", "EventProperties",
           "AttributeConfig", "AttributeConfig_2",
           "AttributeConfig_3", "AttributeConfig_5",
           "MultiAttrProp", "device_server_init")


class LatestDeviceImpl(get_latest_device_class()):
    __doc__ = """\
    Latest implementation of the TANGO device base class (alias for {0}).

    It inherits from CORBA classes where all the network layer is implemented.
    """.format(get_latest_device_class().__name__)


class AttributeAlarm(object):
    """This class represents the python interface for the Tango IDL object
    AttributeAlarm."""

    def __init__(self):
        self.min_alarm = ''
        self.max_alarm = ''
        self.min_warning = ''
        self.max_warning = ''
        self.delta_t = ''
        self.delta_val = ''
        self.extensions = []


class ChangeEventProp(object):
    """This class represents the python interface for the Tango IDL object
    ChangeEventProp."""

    def __init__(self):
        self.rel_change = ''
        self.abs_change = ''
        self.extensions = []


class PeriodicEventProp(object):
    """This class represents the python interface for the Tango IDL object
    PeriodicEventProp."""

    def __init__(self):
        self.period = ''
        self.extensions = []


class ArchiveEventProp(object):
    """This class represents the python interface for the Tango IDL object
    ArchiveEventProp."""

    def __init__(self):
        self.rel_change = ''
        self.abs_change = ''
        self.period = ''
        self.extensions = []


class EventProperties(object):
    """This class represents the python interface for the Tango IDL object
    EventProperties."""

    def __init__(self):
        self.ch_event = ChangeEventProp()
        self.per_event = PeriodicEventProp()
        self.arch_event = ArchiveEventProp()


class MultiAttrProp(object):
    """This class represents the python interface for the Tango IDL object
    MultiAttrProp."""

    def __init__(self):
        self.label = ''
        self.description = ''
        self.unit = ''
        self.standard_unit = ''
        self.display_unit = ''
        self.format = ''
        self.min_value = ''
        self.max_value = ''
        self.min_alarm = ''
        self.max_alarm = ''
        self.min_warning = ''
        self.max_warning = ''
        self.delta_t = ''
        self.delta_val = ''
        self.event_period = ''
        self.archive_period = ''
        self.rel_change = ''
        self.abs_change = ''
        self.archive_rel_change = ''
        self.archive_abs_change = ''


def _init_attr_config(attr_cfg):
    """Helper function to initialize attribute config objects"""
    attr_cfg.name = ''
    attr_cfg.writable = AttrWriteType.READ
    attr_cfg.data_format = AttrDataFormat.SCALAR
    attr_cfg.data_type = 0
    attr_cfg.max_dim_x = 0
    attr_cfg.max_dim_y = 0
    attr_cfg.description = ''
    attr_cfg.label = ''
    attr_cfg.unit = ''
    attr_cfg.standard_unit = ''
    attr_cfg.display_unit = ''
    attr_cfg.format = ''
    attr_cfg.min_value = ''
    attr_cfg.max_value = ''
    attr_cfg.writable_attr_name = ''
    attr_cfg.extensions = []


class AttributeConfig(object):
    """This class represents the python interface for the Tango IDL object
    AttributeConfig."""

    def __init__(self):
        _init_attr_config(self)
        self.min_alarm = ''
        self.max_alarm = ''


class AttributeConfig_2(object):
    """This class represents the python interface for the Tango IDL object
    AttributeConfig_2."""

    def __init__(self):
        _init_attr_config(self)
        self.level = DispLevel.OPERATOR
        self.min_alarm = ''
        self.max_alarm = ''


class AttributeConfig_3(object):
    """This class represents the python interface for the Tango IDL object
    AttributeConfig_3."""

    def __init__(self):
        _init_attr_config(self)
        self.level = -1
        self.att_alarm = AttributeAlarm()
        self.event_prop = EventProperties()
        self.sys_extensions = []


class AttributeConfig_5(object):
    """This class represents the python interface for the Tango IDL object
    AttributeConfig_5."""

    def __init__(self):
        _init_attr_config(self)
        self.memorized = False
        self.mem_init = False
        self.level = -1
        self.root_attr_name = ''
        self.enum_labels = []
        self.att_alarm = AttributeAlarm()
        self.event_prop = EventProperties()
        self.sys_extensions = []


def __Attribute__get_properties(self, attr_cfg=None):
    """get_properties(self, attr_cfg = None) -> AttributeConfig

                Get attribute properties.

            Parameters :
                - conf : the config object to be filled with
                         the attribute configuration. Default is None meaning the
                         method will create internally a new AttributeConfig_5
                         and return it.
                         Can be AttributeConfig, AttributeConfig_2,
                         AttributeConfig_3, AttributeConfig_5 or
                         MultiAttrProp

            Return     : (AttributeConfig) the config object filled with
                         attribute configuration information

            New in PyTango 7.1.4
    """

    if attr_cfg is None:
        attr_cfg = MultiAttrProp()
    if not isinstance(attr_cfg, MultiAttrProp):
        raise TypeError("attr_cfg must be an instance of MultiAttrProp")
    return self._get_properties_multi_attr_prop(attr_cfg)


def __Attribute__set_properties(self, attr_cfg, dev=None):
    """set_properties(self, attr_cfg, dev) -> None

                Set attribute properties.

                This method sets the attribute properties value with the content
                of the fileds in the AttributeConfig/ AttributeConfig_3 object

            Parameters :
                - conf : (AttributeConfig or AttributeConfig_3) the config
                         object.
                - dev : (DeviceImpl) the device (not used, maintained
                        for backward compatibility)

            New in PyTango 7.1.4
    """

    if not isinstance(attr_cfg, MultiAttrProp):
        raise TypeError("attr_cfg must be an instance of MultiAttrProp")
    return self._set_properties_multi_attr_prop(attr_cfg)


def __Attribute__str(self):
    return '%s(%s)' % (self.__class__.__name__, self.get_name())


def __init_Attribute():
    Attribute.__str__ = __Attribute__str
    Attribute.__repr__ = __Attribute__str
    Attribute.get_properties = __Attribute__get_properties
    Attribute.set_properties = __Attribute__set_properties


def __DeviceImpl__get_device_class(self):
    try:
        return self._device_class_instance
    except AttributeError:
        return None


def __DeviceImpl__get_device_properties(self, ds_class=None):
    """get_device_properties(self, ds_class = None) -> None

                Utility method that fetches all the device properties from the database
                and converts them into members of this DeviceImpl.

            Parameters :
                - ds_class : (DeviceClass) the DeviceClass object. Optional. Default value is
                             None meaning that the corresponding DeviceClass object for this
                             DeviceImpl will be used

            Return     : None

            Throws     : DevFailed
    """
    if ds_class is None:
        try:
            # Call this method in a try/except in case this is called during the DS shutdown sequence
            ds_class = self.get_device_class()
        except:
            return
    try:
        pu = self.prop_util = ds_class.prop_util
        self.device_property_list = copy.deepcopy(ds_class.device_property_list)
        class_prop = ds_class.class_property_list
        pu.get_device_properties(self, class_prop, self.device_property_list)
        for prop_name in class_prop:
            setattr(self, prop_name, pu.get_property_values(prop_name, class_prop))
        for prop_name in self.device_property_list:
            setattr(self, prop_name, self.prop_util.get_property_values(prop_name, self.device_property_list))
    except DevFailed as df:
        print(80 * "-")
        print(df)
        raise df


def __DeviceImpl__add_attribute(self, attr, r_meth=None, w_meth=None, is_allo_meth=None):
    """add_attribute(self, attr, r_meth=None, w_meth=None, is_allo_meth=None) -> Attr

            Add a new attribute to the device attribute list. Please, note that if you add
            an attribute to a device at device creation time, this attribute will be added
            to the device class attribute list. Therefore, all devices belonging to the
            same class created after this attribute addition will also have this attribute.

        Parameters :
            - attr : (Attr or AttrData) the new attribute to be added to the list.
            - r_meth : (callable) the read method to be called on a read request
            - w_meth : (callable) the write method to be called on a write request
                       (if attr is writable)
            - is_allo_meth: (callable) the method that is called to check if it
                            is possible to access the attribute or not

        Return     : (Attr) the newly created attribute.

        Throws     : DevFailed
    """

    attr_data = None
    if isinstance(attr, AttrData):
        attr_data = attr
        attr = attr.to_attr()

    att_name = attr.get_name()

    add_name_in_list = False

    r_name = 'read_%s' % att_name
    if r_meth is None:
        if attr_data is not None:
            r_name = attr_data.read_method_name
    else:
        r_name = r_meth.__name__

    w_name = 'write_%s' % att_name
    if w_meth is None:
        if attr_data is not None:
            w_name = attr_data.write_method_name
    else:
        w_name = w_meth.__name__

    ia_name = 'is_%s_allowed' % att_name
    if is_allo_meth is None:
        if attr_data is not None:
            ia_name = attr_data.is_allowed_name
    else:
        ia_name = is_allo_meth.__name__

    try:
        self._add_attribute(attr, r_name, w_name, ia_name)
        if add_name_in_list:
            cl = self.get_device_class()
            cl.dyn_att_added_methods.append(att_name)
    except:
        if add_name_in_list:
            self._remove_attr_meth(att_name)
        raise
    return attr


def __DeviceImpl__remove_attribute(self, attr_name):
    """remove_attribute(self, attr_name) -> None

            Remove one attribute from the device attribute list.

        Parameters :
            - attr_name : (str) attribute name

        Return     : None

        Throws     : DevFailed
    """
    try:
        # Call this method in a try/except in case remove_attribute
        # is called during the DS shutdown sequence
        cl = self.get_device_class()
    except:
        return

    dev_list = cl.get_device_list()
    nb_dev = len(dev_list)
    if nb_dev == 1:
        self._remove_attr_meth(attr_name)
    else:
        nb_except = 0
        for dev in dev_list:
            try:
                dev.get_device_attr().get_attr_by_name(attr_name)
            except:
                nb_except += 1
        if nb_except == nb_dev - 1:
            self._remove_attr_meth(attr_name)
    self._remove_attribute(attr_name)


def __DeviceImpl___remove_attr_meth(self, attr_name):
    """for internal usage only"""
    cl = self.get_device_class()
    if cl.dyn_att_added_methods.count(attr_name) != 0:
        r_meth_name = 'read_%s' % attr_name
        if hasattr(self.__class__, r_meth_name):
            delattr(self.__class__, r_meth_name)

        w_meth_name = 'write_%s' % attr_name
        if hasattr(self.__class__, w_meth_name):
            delattr(self.__class__, w_meth_name)

        allo_meth_name = 'is_%s_allowed' % attr_name
        if hasattr(self.__class__, allo_meth_name):
            delattr(self.__class__, allo_meth_name)
        cl.dyn_att_added_methods.remove(attr_name)


def __DeviceImpl__add_command(self, cmd, device_level=True):
    """add_command(self, cmd, level=TANGO::OPERATOR) -> cmd

        Add a new command to the device command list.

        Parameters :
            - cmd          : the new command to be added to the list
            - device_level : Set this flag to true if the command must be added
                             for only this device

        Return     : Command

        Throws     : DevFailed
    """
    add_name_in_list = False      # This flag is always False, what use is it?
    try:
        config = dict(cmd.__tango_command__[1][2])
        if config and ("Display level" in config):
            disp_level = config["Display level"]
        else:
            disp_level = DispLevel.OPERATOR
        self._add_command(cmd.__name__, cmd.__tango_command__[1], disp_level,
                          device_level)
        if add_name_in_list:
            cl = self.get_device_class()
            cl.dyn_cmd_added_methods.append(cmd.__name__)
    except:
        if add_name_in_list:
            self._remove_cmd(cmd.__name__)
        raise
    return cmd


def __DeviceImpl__remove_command(self, cmd_name, free_it=False, clean_db=True):
    """
    remove_command(self, attr_name) -> None

            Remove one command from the device command list.

        Parameters :
            - cmd_name : (str) command name to be removed from the list
            - free_it  : Boolean set to true if the command object must be freed.
            - clean_db : Clean command related information (included polling info
                         if the command is polled) from database.
        Return     : None

        Throws     : DevFailed
    """
    try:
        # Call this method in a try/except in case remove
        # is called during the DS shutdown sequence
        cl = self.get_device_class()
    except:
        return

    if cl.dyn_cmd_added_methods.count(cmd_name) != 0:
        cl.dyn_cmd_added_methods.remove(cmd_name)

    self._remove_command(cmd_name, free_it, clean_db)


def __DeviceImpl__debug_stream(self, msg, *args):
    """
    debug_stream(self, msg, *args) -> None

            Sends the given message to the tango debug stream.

            Since PyTango 7.1.3, the same can be achieved with::

                print(msg, file=self.log_debug)

        Parameters :
            - msg : (str) the message to be sent to the debug stream
        Return     : None
    """
    self.__debug_stream(msg % args)


def __DeviceImpl__info_stream(self, msg, *args):
    """
    info_stream(self, msg, *args) -> None

            Sends the given message to the tango info stream.

            Since PyTango 7.1.3, the same can be achieved with::

                print(msg, file=self.log_info)

        Parameters :
            - msg : (str) the message to be sent to the info stream
        Return     : None
    """
    self.__info_stream(msg % args)


def __DeviceImpl__warn_stream(self, msg, *args):
    """
    warn_stream(self, msg, *args) -> None

            Sends the given message to the tango warn stream.

            Since PyTango 7.1.3, the same can be achieved with::

                print(msg, file=self.log_warn)

        Parameters :
            - msg : (str) the message to be sent to the warn stream
        Return     : None
    """
    self.__warn_stream(msg % args)


def __DeviceImpl__error_stream(self, msg, *args):
    """
    error_stream(self, msg, *args) -> None

            Sends the given message to the tango error stream.

            Since PyTango 7.1.3, the same can be achieved with::

                print(msg, file=self.log_error)

        Parameters :
            - msg : (str) the message to be sent to the error stream
        Return     : None
    """
    self.__error_stream(msg % args)


def __DeviceImpl__fatal_stream(self, msg, *args):
    """
    fatal_stream(self, msg, *args) -> None

            Sends the given message to the tango fatal stream.

            Since PyTango 7.1.3, the same can be achieved with::

                print(msg, file=self.log_fatal)

        Parameters :
            - msg : (str) the message to be sent to the fatal stream
        Return     : None
    """
    self.__fatal_stream(msg % args)


@property
def __DeviceImpl__debug(self):
    if not hasattr(self, "_debug_s"):
        self._debug_s = TangoStream(self.debug_stream)
    return self._debug_s


@property
def __DeviceImpl__info(self):
    if not hasattr(self, "_info_s"):
        self._info_s = TangoStream(self.info_stream)
    return self._info_s


@property
def __DeviceImpl__warn(self):
    if not hasattr(self, "_warn_s"):
        self._warn_s = TangoStream(self.warn_stream)
    return self._warn_s


@property
def __DeviceImpl__error(self):
    if not hasattr(self, "_error_s"):
        self._error_s = TangoStream(self.error_stream)
    return self._error_s


@property
def __DeviceImpl__fatal(self):
    if not hasattr(self, "_fatal_s"):
        self._fatal_s = TangoStream(self.fatal_stream)
    return self._fatal_s


def __DeviceImpl__str(self):
    return '%s(%s)' % (self.__class__.__name__, self.get_name())


def __init_DeviceImpl():
    DeviceImpl._device_class_instance = None
    DeviceImpl.get_device_class = __DeviceImpl__get_device_class
    DeviceImpl.get_device_properties = __DeviceImpl__get_device_properties
    DeviceImpl.add_attribute = __DeviceImpl__add_attribute
    DeviceImpl.remove_attribute = __DeviceImpl__remove_attribute
    DeviceImpl._remove_attr_meth = __DeviceImpl___remove_attr_meth
    DeviceImpl.add_command = __DeviceImpl__add_command
    DeviceImpl.remove_command = __DeviceImpl__remove_command
    DeviceImpl.__str__ = __DeviceImpl__str
    DeviceImpl.__repr__ = __DeviceImpl__str
    DeviceImpl.debug_stream = __DeviceImpl__debug_stream
    DeviceImpl.info_stream = __DeviceImpl__info_stream
    DeviceImpl.warn_stream = __DeviceImpl__warn_stream
    DeviceImpl.error_stream = __DeviceImpl__error_stream
    DeviceImpl.fatal_stream = __DeviceImpl__fatal_stream
    DeviceImpl.log_debug = __DeviceImpl__debug
    DeviceImpl.log_info = __DeviceImpl__info
    DeviceImpl.log_warn = __DeviceImpl__warn
    DeviceImpl.log_error = __DeviceImpl__error
    DeviceImpl.log_fatal = __DeviceImpl__fatal


def __Logger__log(self, level, msg, *args):
    """
    log(self, level, msg, *args) -> None

            Sends the given message to the tango the selected stream.

        Parameters :
            - level: (Level.LevelLevel) Log level
            - msg : (str) the message to be sent to the stream
            - args: (seq<str>) list of optional message arguments
        Return     : None

    .. versionchanged:
    """
    self.__log(level, msg % args)


def __Logger__log_unconditionally(self, level, msg, *args):
    """
    log_unconditionally(self, level, msg, *args) -> None

            Sends the given message to the tango the selected stream,
            without checking the level.

        Parameters :
            - level: (Level.LevelLevel) Log level
            - msg : (str) the message to be sent to the stream
            - args: (seq<str>) list of optional message arguments
        Return     : None
    """
    self.__log_unconditionally(level, msg % args)


def __Logger__debug(self, msg, *args):
    """
    debug(self, msg, *args) -> None

            Sends the given message to the tango debug stream.

        Parameters :
            - msg : (str) the message to be sent to the debug stream
            - args: (seq<str>) list of optional message arguments
        Return     : None
    """
    self.__debug(msg % args)


def __Logger__info(self, msg, *args):
    """
    info(self, msg, *args) -> None

            Sends the given message to the tango info stream.

        Parameters :
            - msg : (str) the message to be sent to the info stream
            - args: (seq<str>) list of optional message arguments
        Return     : None
    """
    self.__info(msg % args)


def __Logger__warn(self, msg, *args):
    """
    warn(self, msg, *args) -> None

            Sends the given message to the tango warn stream.

        Parameters :
            - msg : (str) the message to be sent to the warn stream
            - args: (seq<str>) list of optional message arguments
        Return     : None
    """
    self.__warn(msg % args)


def __Logger__error(self, msg, *args):
    """
    error(self, msg, *args) -> None

            Sends the given message to the tango error stream.

        Parameters :
            - msg : (str) the message to be sent to the error stream
            - args: (seq<str>) list of optional message arguments
        Return     : None
    """
    self.__error(msg % args)


def __Logger__fatal(self, msg, *args):
    """
    fatal(self, msg, *args) -> None

            Sends the given message to the tango fatal stream.

        Parameters :
            - msg : (str) the message to be sent to the fatal stream
            - args: (seq<str>) list of optional message arguments
        Return     : None
    """
    self.__fatal(msg % args)


def __UserDefaultAttrProp_set_enum_labels(self, enum_labels):
    """
    set_enum_labels(self, enum_labels) -> None

            Set default enumeration labels.

        Parameters :
            - enum_labels : (seq<str>) list of enumeration labels

        New in PyTango 9.2.0
    """
    elbls = StdStringVector()
    for enu in enum_labels:
        elbls.append(enu)
    return self._set_enum_labels(elbls)


def __Attr__str(self):
    return '%s(%s)' % (self.__class__.__name__, self.get_name())


def __init_Attr():
    Attr.__str__ = __Attr__str
    Attr.__repr__ = __Attr__str


def __init_UserDefaultAttrProp():
    UserDefaultAttrProp.set_enum_labels = __UserDefaultAttrProp_set_enum_labels


def __init_Logger():
    Logger.log = __Logger__log
    Logger.log_unconditionally = __Logger__log_unconditionally
    Logger.debug = __Logger__debug
    Logger.info = __Logger__info
    Logger.warn = __Logger__warn
    Logger.error = __Logger__error
    Logger.fatal = __Logger__fatal


def __doc_DeviceImpl():
    def document_method(method_name, desc, append=True):
        return __document_method(DeviceImpl, method_name, desc, append)

    DeviceImpl.__doc__ = """
    Base class for all TANGO device.
    This class inherits from CORBA classes where all the network layer is implemented.
    """

    document_method("init_device", """
    init_device(self) -> None

            Intialize the device.

        Parameters : None
        Return     : None

    """)

    document_method("set_state", """
    set_state(self, new_state) -> None

            Set device state.

        Parameters :
            - new_state : (DevState) the new device state
        Return     : None

    """)

    document_method("get_state", """
    get_state(self) -> DevState

            Get a COPY of the device state.

        Parameters : None
        Return     : (DevState) Current device state

    """)

    document_method("get_prev_state", """
    get_prev_state(self) -> DevState

            Get a COPY of the device's previous state.

        Parameters : None
        Return     : (DevState) the device's previous state

    """)

    document_method("get_name", """
    get_name(self) -> (str)

            Get a COPY of the device name.

        Parameters : None
        Return     : (str) the device name

    """)

    document_method("get_device_attr", """
    get_device_attr(self) -> MultiAttribute

            Get device multi attribute object.

        Parameters : None
        Return     : (MultiAttribute) the device's MultiAttribute object

    """)

    document_method("register_signal", """
    register_signal(self, signo) -> None

            Register a signal.
            Register this device as device to be informed when signal signo
            is sent to to the device server process

        Parameters :
            - signo : (int) signal identifier
        Return     : None

    """)

    document_method("unregister_signal", """
    unregister_signal(self, signo) -> None

            Unregister a signal.
            Unregister this device as device to be informed when signal signo
            is sent to to the device server process

        Parameters :
            - signo : (int) signal identifier
        Return     : None

    """)

    document_method("get_status", """
    get_status(self, ) -> str

            Get a COPY of the device status.

        Parameters : None
        Return     : (str) the device status

    """)

    document_method("set_status", """
    set_status(self, new_status) -> None

            Set device status.

        Parameters :
            - new_status : (str) the new device status
        Return     : None

    """)

    document_method("append_status", """
    append_status(self, status, new_line=False) -> None

            Appends a string to the device status.

        Parameters :
            status : (str) the string to be appened to the device status
            new_line : (bool) If true, appends a new line character before the string. Default is False
        Return     : None

    """)

    document_method("dev_state", """
    dev_state(self) -> DevState

            Get device state.
            Default method to get device state. The behaviour of this method depends
            on the device state. If the device state is ON or ALARM, it reads the
            attribute(s) with an alarm level defined, check if the read value is
            above/below the alarm and eventually change the state to ALARM, return
            the device state. For all th other device state, this method simply
            returns the state This method can be redefined in sub-classes in case
            of the default behaviour does not fullfill the needs.

        Parameters : None
        Return     : (DevState) the device state

        Throws     : DevFailed - If it is necessary to read attribute(s) and a problem occurs during the reading
    """)

    document_method("dev_status", """
    dev_status(self) -> str

            Get device status.
            Default method to get device status. It returns the contents of the device
            dev_status field. If the device state is ALARM, alarm messages are added
            to the device status. This method can be redefined in sub-classes in case
            of the default behaviour does not fullfill the needs.

        Parameters : None
        Return     : (str) the device status

        Throws     : DevFailed - If it is necessary to read attribute(s) and a problem occurs during the reading
    """)

    document_method("set_change_event", """
    set_change_event(self, attr_name, implemented, detect=True) -> None

            Set an implemented flag for the attribute to indicate that the server fires
            change events manually, without the polling to be started.
            If the detect parameter is set to true, the criteria specified for the
            change event are verified and the event is only pushed if they are fullfilled.
            If detect is set to false the event is fired without any value checking!

        Parameters :
            - attr_name : (str) attribute name
            - implemented : (bool) True when the server fires change events manually.
            - detect : (bool) Triggers the verification of the change event properties
                       when set to true. Default value is true.
        Return     : None
    """)

    document_method("set_archive_event", """
    set_archive_event(self, attr_name, implemented, detect=True) -> None

            Set an implemented flag for the attribute to indicate that the server fires
            archive events manually, without the polling to be started.
            If the detect parameter is set to true, the criteria specified for the
            archive event are verified and the event is only pushed if they are fullfilled.
            If detect is set to false the event is fired without any value checking!

        Parameters :
            - attr_name : (str) attribute name
            - implemented : (bool) True when the server fires change events manually.
            - detect : (bool) Triggers the verification of the change event properties
                       when set to true. Default value is true.
        Return     : None

    """)

    document_method("push_change_event", """
    push_change_event(self, attr_name) -> None
    push_change_event(self, attr_name, except) -> None
    push_change_event(self, attr_name, data, dim_x = 1, dim_y = 0) -> None
    push_change_event(self, attr_name, str_data, data) -> None
    push_change_event(self, attr_name, data, time_stamp, quality, dim_x = 1, dim_y = 0) -> None
    push_change_event(self, attr_name, str_data, data, time_stamp, quality) -> None

        Push a change event for the given attribute name.
        The event is pushed to the notification daemon.

        Parameters:
            - attr_name : (str) attribute name
            - data : the data to be sent as attribute event data. Data must be compatible with the
                     attribute type and format.
                     for SPECTRUM and IMAGE attributes, data can be any type of sequence of elements
                     compatible with the attribute type
            - str_data : (str) special variation for DevEncoded data type. In this case 'data' must
                         be a str or an object with the buffer interface.
            - except: (DevFailed) Instead of data, you may want to send an exception.
            - dim_x : (int) the attribute x length. Default value is 1
            - dim_y : (int) the attribute y length. Default value is 0
            - time_stamp : (double) the time stamp
            - quality : (AttrQuality) the attribute quality factor

        Throws     : DevFailed If the attribute data type is not coherent.
    """)

    document_method("push_archive_event", """
    push_archive_event(self, attr_name) -> None
    push_archive_event(self, attr_name, except) -> None
    push_archive_event(self, attr_name, data, dim_x = 1, dim_y = 0) -> None
    push_archive_event(self, attr_name, str_data, data) -> None
    push_archive_event(self, attr_name, data, time_stamp, quality, dim_x = 1, dim_y = 0) -> None
    push_archive_event(self, attr_name, str_data, data, time_stamp, quality) -> None

            Push an archive event for the given attribute name.
            The event is pushed to the notification daemon.

        Parameters:
            - attr_name : (str) attribute name
            - data : the data to be sent as attribute event data. Data must be compatible with the
                     attribute type and format.
                     for SPECTRUM and IMAGE attributes, data can be any type of sequence of elements
                     compatible with the attribute type
            - str_data : (str) special variation for DevEncoded data type. In this case 'data' must
                         be a str or an object with the buffer interface.
            - except: (DevFailed) Instead of data, you may want to send an exception.
            - dim_x : (int) the attribute x length. Default value is 1
            - dim_y : (int) the attribute y length. Default value is 0
            - time_stamp : (double) the time stamp
            - quality : (AttrQuality) the attribute quality factor

        Throws     : DevFailed If the attribute data type is not coherent.
    """)

    document_method("push_event", """
    push_event(self, attr_name, filt_names, filt_vals) -> None
    push_event(self, attr_name, filt_names, filt_vals, data, dim_x = 1, dim_y = 0) -> None
    push_event(self, attr_name, filt_names, filt_vals, str_data, data) -> None
    push_event(self, attr_name, filt_names, filt_vals, data, time_stamp, quality, dim_x = 1, dim_y = 0) -> None
    push_event(self, attr_name, filt_names, filt_vals, str_data, data, time_stamp, quality) -> None

            Push a user event for the given attribute name.
            The event is pushed to the notification daemon.

        Parameters:
            - attr_name : (str) attribute name
            - filt_names : (sequence<str>) the filterable fields name
            - filt_vals : (sequence<double>) the filterable fields value
            - data : the data to be sent as attribute event data. Data must be compatible with the
                     attribute type and format.
                     for SPECTRUM and IMAGE attributes, data can be any type of sequence of elements
                     compatible with the attribute type
            - str_data : (str) special variation for DevEncoded data type. In this case 'data' must
                         be a str or an object with the buffer interface.
            - dim_x : (int) the attribute x length. Default value is 1
            - dim_y : (int) the attribute y length. Default value is 0
            - time_stamp : (double) the time stamp
            - quality : (AttrQuality) the attribute quality factor

        Throws     : DevFailed If the attribute data type is not coherent.
    """)

    document_method("push_data_ready_event", """
    push_data_ready_event(self, attr_name, counter = 0) -> None

            Push a data ready event for the given attribute name.
            The event is pushed to the notification daemon.

            The method needs only the attribue name and an optional
            "counter" which will be passed unchanged within the event

        Parameters :
            - attr_name : (str) attribute name
            - counter : (int) the user counter
        Return     : None

        Throws     : DevFailed If the attribute name is unknown.
    """)

    document_method("push_pipe_event", """
    push_pipe_event(self, pipe_name, except) -> None
    push_pipe_event(self, pipe_name, blob, reuse_it) -> None
    push_pipe_event(self, pipe_name, blob, timeval, reuse_it) -> None

            Push a pipe event for the given blob.

        Parameters :
            - pipe_name : (str) pipe name
            - blob  : (DevicePipeBlob) the blob data
        Return     : None

        Throws     : DevFailed If the pipe name is unknown.

        New in PyTango 9.2.2
    """)

    document_method("get_logger", """
    get_logger(self) -> Logger

            Returns the Logger object for this device

        Parameters : None
        Return     : (Logger) the Logger object for this device
    """)

    document_method("get_exported_flag", """
    get_exported_flag(self) -> bool

            Returns the state of the exported flag

        Parameters : None
        Return     : (bool) the state of the exported flag

        New in PyTango 7.1.2
    """)

    document_method("get_poll_ring_depth", """
    get_poll_ring_depth(self) -> int

            Returns the poll ring depth

        Parameters : None
        Return     : (int) the poll ring depth

        New in PyTango 7.1.2
    """)

    document_method("get_poll_old_factor", """
    get_poll_old_factor(self) -> int

            Returns the poll old factor

        Parameters : None
        Return     : (int) the poll old factor

        New in PyTango 7.1.2
    """)

    document_method("is_polled", """
    is_polled(self) -> bool

            Returns if it is polled

        Parameters : None
        Return     : (bool) True if it is polled or False otherwise

        New in PyTango 7.1.2
    """)

    document_method("get_polled_cmd", """
    get_polled_cmd(self) -> sequence<str>

            Returns a COPY of the list of polled commands

        Parameters : None
        Return     : (sequence<str>) a COPY of the list of polled commands

        New in PyTango 7.1.2
    """)

    document_method("get_polled_attr", """
    get_polled_attr(self) -> sequence<str>

            Returns a COPY of the list of polled attributes

        Parameters : None
        Return     : (sequence<str>) a COPY of the list of polled attributes

        New in PyTango 7.1.2
    """)

    document_method("get_non_auto_polled_cmd", """
    get_non_auto_polled_cmd(self) -> sequence<str>

            Returns a COPY of the list of non automatic polled commands

        Parameters : None
        Return     : (sequence<str>) a COPY of the list of non automatic polled commands

        New in PyTango 7.1.2
    """)

    document_method("get_non_auto_polled_attr", """
    get_non_auto_polled_attr(self) -> sequence<str>

            Returns a COPY of the list of non automatic polled attributes

        Parameters : None
        Return     : (sequence<str>) a COPY of the list of non automatic polled attributes

        New in PyTango 7.1.2
    """)

    document_method("stop_polling", """
    stop_polling(self) -> None
    stop_polling(self, with_db_upd) -> None

            Stop all polling for a device. if the device is polled, call this
            method before deleting it.

        Parameters :
            - with_db_upd : (bool)  Is it necessary to update db ?
        Return     : None

        New in PyTango 7.1.2
    """)

    document_method("get_attribute_poll_period", """
    get_attribute_poll_period(self, attr_name) -> int

            Returns the attribute polling period (ms) or 0 if the attribute
            is not polled.

        Parameters :
            - attr_name : (str) attribute name
        Return     : (int) attribute polling period (ms) or 0 if it is not polled

        New in PyTango 8.0.0
    """)

    document_method("get_command_poll_period", """
    get_command_poll_period(self, cmd_name) -> int

            Returns the command polling period (ms) or 0 if the command
            is not polled.

        Parameters :
            - cmd_name : (str) command name
        Return     : (int) command polling period (ms) or 0 if it is not polled

        New in PyTango 8.0.0
    """)

    document_method("check_command_exists", """
    check_command_exists(self) -> None

            This method check that a command is supported by the device and
            does not need input value. The method throws an exception if the
            command is not defined or needs an input value

        Parameters :
            - cmd_name: (str) the command name
        Return     : None

        Throws     : DevFailed API_IncompatibleCmdArgumentType, API_CommandNotFound

        New in PyTango 7.1.2
    """)

    document_method("get_dev_idl_version", """
    get_dev_idl_version(self) -> int

            Returns the IDL version

        Parameters : None
        Return     : (int) the IDL version

        New in PyTango 7.1.2
    """)

    document_method("get_cmd_poll_ring_depth", """
    get_cmd_poll_ring_depth(self, cmd_name) -> int

            Returns the command poll ring depth

        Parameters :
            - cmd_name: (str) the command name
        Return     : (int) the command poll ring depth

        New in PyTango 7.1.2
    """)

    document_method("get_attr_poll_ring_depth", """
    get_attr_poll_ring_depth(self, attr_name) -> int

            Returns the attribute poll ring depth

        Parameters :
            - attr_name: (str) the attribute name
        Return     : (int) the attribute poll ring depth

        New in PyTango 7.1.2
    """)

    document_method("is_device_locked", """
    is_device_locked(self) -> bool

            Returns if this device is locked by a client

        Parameters : None
        Return     : (bool) True if it is locked or False otherwise

        New in PyTango 7.1.2
    """)

    document_method("get_min_poll_period", """
    get_min_poll_period(self) -> int

            Returns the min poll period

        Parameters : None
        Return     : (int) the min poll period

        New in PyTango 7.2.0
    """)

    document_method("get_cmd_min_poll_period", """
    get_cmd_min_poll_period(self) -> seq<str>

            Returns the min command poll period

        Parameters : None
        Return     : (seq<str>) the min command poll period

        New in PyTango 7.2.0
    """)

    document_method("get_attr_min_poll_period", """
    get_attr_min_poll_period(self) -> seq<str>

            Returns the min attribute poll period

        Parameters : None
        Return     : (seq<str>) the min attribute poll period

        New in PyTango 7.2.0
    """)

    document_method("push_att_conf_event", """
    push_att_conf_event(self, attr) -> None

            Push an attribute configuration event.

        Parameters : (Attribute) the attribute for which the configuration event
                     will be sent.
        Return     : None

        New in PyTango 7.2.1
    """)

    document_method("push_pipe_event", """
    push_pipe_event(self, blob) -> None

            Push an pipe event.

        Parameters :  the blob which pipe event will be send.
        Return     : None

        New in PyTango 9.2.2
    """)

    document_method("is_there_subscriber", """
    is_there_subscriber(self, att_name, event_type) -> bool

            Check if there is subscriber(s) listening for the event.

            This method returns a boolean set to true if there are some
            subscriber(s) listening on the event specified by the two method
            arguments. Be aware that there is some delay (up to 600 sec)
            between this method returning false and the last subscriber
            unsubscription or crash...

            The device interface change event is not supported by this method.

        Parameters :
            - att_name: (str) the attribute name
            - event_type (EventType): the event type
        Return     : True if there is at least one listener or False otherwise
    """)


def __doc_extra_DeviceImpl(cls):
    def document_method(method_name, desc, append=True):
        return __document_method(cls, method_name, desc, append)

    document_method("delete_device", """
    delete_device(self) -> None

            Delete the device.

        Parameters : None
        Return     : None

    """)

    document_method("always_executed_hook", """
    always_executed_hook(self) -> None

            Hook method.
            Default method to implement an action necessary on a device before
            any command is executed. This method can be redefined in sub-classes
            in case of the default behaviour does not fullfill the needs

        Parameters : None
        Return     : None

        Throws     : DevFailed This method does not throw exception but a redefined method can.
    """)

    document_method("read_attr_hardware", """
    read_attr_hardware(self, attr_list) -> None

            Read the hardware to return attribute value(s).
            Default method to implement an action necessary on a device to read
            the hardware involved in a a read attribute CORBA call. This method
            must be redefined in sub-classes in order to support attribute reading

        Parameters :
            attr_list : (sequence<int>) list of indices in the device object attribute vector
                        of an attribute to be read.
        Return     : None

        Throws     : DevFailed This method does not throw exception but a redefined method can.
    """)

    document_method("write_attr_hardware", """
    write_attr_hardware(self) -> None

            Write the hardware for attributes.
            Default method to implement an action necessary on a device to write
            the hardware involved in a a write attribute. This method must be
            redefined in sub-classes in order to support writable attribute

        Parameters :
            attr_list : (sequence<int>) list of indices in the device object attribute vector
                        of an attribute to be written.
        Return     : None

        Throws     : DevFailed This method does not throw exception but a redefined method can.
    """)

    document_method("signal_handler", """
    signal_handler(self, signo) -> None

            Signal handler.
            The method executed when the signal arrived in the device server process.
            This method is defined as virtual and then, can be redefined following
            device needs.

        Parameters :
            - signo : (int) the signal number
        Return     : None

        Throws     : DevFailed This method does not throw exception but a redefined method can.
    """)

    copy_doc(cls, "dev_state")
    copy_doc(cls, "dev_status")


def __doc_Attribute():
    def document_method(method_name, desc, append=True):
        return __document_method(Attribute, method_name, desc, append)

    Attribute.__doc__ = """
    This class represents a Tango attribute.
    """

    document_method("is_write_associated", """
    is_write_associated(self) -> bool

            Check if the attribute has an associated writable attribute.

        Parameters : None
        Return     : (bool) True if there is an associated writable attribute
    """)

    document_method("is_min_alarm", """
    is_min_alarm(self) -> bool

            Check if the attribute is in minimum alarm condition.

        Parameters : None
        Return     : (bool) true if the attribute is in alarm condition (read value below the min. alarm).
    """)

    document_method("is_max_alarm", """
    is_max_alarm(self) -> bool

            Check if the attribute is in maximum alarm condition.

        Parameters : None
        Return     : (bool) true if the attribute is in alarm condition (read value above the max. alarm).
    """)

    document_method("is_min_warning", """
    is_min_warning(self) -> bool

            Check if the attribute is in minimum warning condition.

        Parameters : None
        Return     : (bool) true if the attribute is in warning condition (read value below the min. warning).
    """)

    document_method("is_max_warning", """
    is_max_warning(self) -> bool

            Check if the attribute is in maximum warning condition.

        Parameters : None
        Return     : (bool) true if the attribute is in warning condition (read value above the max. warning).
    """)

    document_method("is_rds_alarm", """
    is_rds_alarm(self) -> bool

            Check if the attribute is in RDS alarm condition.

        Parameters : None
        Return     : (bool) true if the attribute is in RDS condition (Read Different than Set).
    """)

    document_method("is_polled", """
    is_polled(self) -> bool

            Check if the attribute is polled.

        Parameters : None
        Return     : (bool) true if the attribute is polled.
    """)

    document_method("check_alarm", """
    check_alarm(self) -> bool

            Check if the attribute read value is below/above the alarm level.

        Parameters : None
        Return     : (bool) true if the attribute is in alarm condition.

        Throws     : DevFailed If no alarm level is defined.
    """)

    document_method("get_writable", """
    get_writable(self) -> AttrWriteType

            Get the attribute writable type (RO/WO/RW).

        Parameters : None
        Return     : (AttrWriteType) The attribute write type.
    """)

    document_method("get_name", """
    get_name(self) -> str

            Get attribute name.

        Parameters : None
        Return     : (str) The attribute name
    """)

    document_method("get_data_type", """
    get_data_type(self) -> int

            Get attribute data type.

        Parameters : None
        Return     : (int) the attribute data type
    """)

    document_method("get_data_format", """
    get_data_format(self) -> AttrDataFormat

            Get attribute data format.

        Parameters : None
        Return     : (AttrDataFormat) the attribute data format
    """)

    document_method("get_assoc_name", """
    get_assoc_name(self) -> str

            Get name of the associated writable attribute.

        Parameters : None
        Return     : (str) the associated writable attribute name
    """)

    document_method("get_assoc_ind", """
    get_assoc_ind(self) -> int

            Get index of the associated writable attribute.

        Parameters : None
        Return     : (int) the index in the main attribute vector of the associated writable attribute
    """)

    document_method("set_assoc_ind", """
    set_assoc_ind(self, index) -> None

            Set index of the associated writable attribute.

        Parameters :
            - index : (int) The new index in the main attribute vector of the associated writable attribute
        Return     : None
    """)

    document_method("get_date", """
    get_date(self) -> TimeVal

            Get a COPY of the attribute date.

        Parameters : None
        Return     : (TimeVal) the attribute date
    """)

    document_method("set_date", """
    set_date(self, new_date) -> None

            Set attribute date.

        Parameters :
            - new_date : (TimeVal) the attribute date
        Return     : None
    """)

    document_method("get_label", """
    get_label(self, ) -> str

            Get attribute label property.

        Parameters : None
        Return     : (str) he attribute label
    """)

    document_method("get_quality", """
    get_quality(self) -> AttrQuality

            Get a COPY of the attribute data quality.

        Parameters : None
        Return     : (AttrQuality) the attribute data quality
    """)

    document_method("set_quality", """
    set_quality(self, quality, send_event=False) -> None

            Set attribute data quality.

        Parameters :
            - quality : (AttrQuality) the new attribute data quality
            - send_event : (bool) true if a change event should be sent. Default is false.
        Return     : None
    """)

    document_method("get_data_size", """
    get_data_size(self) -> None

            Get attribute data size.

        Parameters : None
        Return     : (int) the attribute data size
    """)

    document_method("get_x", """
    get_x(self) -> int

            Get attribute data size in x dimension.

        Parameters : None
        Return     : (int) the attribute data size in x dimension. Set to 1 for scalar attribute
    """)

    document_method("get_max_dim_x", """
    get_max_dim_x(self) -> int

            Get attribute maximum data size in x dimension.

        Parameters : None
        Return     : (int) the attribute maximum data size in x dimension. Set to 1 for scalar attribute
    """)

    document_method("get_y", """
    get_y(self) -> int

            Get attribute data size in y dimension.

        Parameters : None
        Return     : (int) the attribute data size in y dimension. Set to 1 for scalar attribute
    """)

    document_method("get_max_dim_y", """
    get_max_dim_y(self) -> int

            Get attribute maximum data size in y dimension.

        Parameters : None
        Return     : (int) the attribute maximum data size in y dimension. Set to 0 for scalar attribute
    """)

    document_method("get_polling_period", """
    get_polling_period(self) -> int

            Get attribute polling period.

        Parameters : None
        Return     : (int) The attribute polling period in mS. Set to 0 when the attribute is not polled
    """)

    document_method("set_attr_serial_model", """
    set_attr_serial_model(self, ser_model) -> void

            Set attribute serialization model.
            This method allows the user to choose the attribute serialization model.

        Parameters :
            - ser_model : (AttrSerialModel) The new serialisation model. The
                          serialization model must be one of ATTR_BY_KERNEL,
                          ATTR_BY_USER or ATTR_NO_SYNC
        Return     : None

        New in PyTango 7.1.0
    """)

    document_method("get_attr_serial_model", """
    get_attr_serial_model(self) -> AttrSerialModel

            Get attribute serialization model.

        Parameters : None
        Return     : (AttrSerialModel) The attribute serialization model

        New in PyTango 7.1.0
    """)

    document_method("set_value", """
    set_value(self, data, dim_x = 1, dim_y = 0) -> None <= DEPRECATED
    set_value(self, data) -> None
    set_value(self, str_data, data) -> None

            Set internal attribute value.
            This method stores the attribute read value inside the object.
            This method also stores the date when it is called and initializes the attribute quality factor.

        Parameters :
            - data : the data to be set. Data must be compatible with the attribute type and format.
                     In the DEPRECATED form for SPECTRUM and IMAGE attributes, data
                     can be any type of FLAT sequence of elements compatible with the
                     attribute type.
                     In the new form (without dim_x or dim_y) data should be any
                     sequence for SPECTRUM and a SEQUENCE of equal-length SEQUENCES
                     for IMAGE attributes.
                     The recommended sequence is a C continuous and aligned numpy
                     array, as it can be optimized.
            - str_data : (str) special variation for DevEncoded data type. In this case 'data' must
                         be a str or an object with the buffer interface.
            - dim_x : (int) [DEPRECATED] the attribute x length. Default value is 1
            - dim_y : (int) [DEPRECATED] the attribute y length. Default value is 0
        Return     : None
    """)

    document_method("set_value_date_quality", """
    set_value_date_quality(self, data, time_stamp, quality, dim_x = 1, dim_y = 0) -> None <= DEPRECATED
    set_value_date_quality(self, data, time_stamp, quality) -> None
    set_value_date_quality(self, str_data, data, time_stamp, quality) -> None

            Set internal attribute value, date and quality factor.
            This method stores the attribute read value, the date and the attribute quality
            factor inside the object.

        Parameters :
            - data : the data to be set. Data must be compatible with the attribute type and format.
                     In the DEPRECATED form for SPECTRUM and IMAGE attributes, data
                     can be any type of FLAT sequence of elements compatible with the
                     attribute type.
                     In the new form (without dim_x or dim_y) data should be any
                     sequence for SPECTRUM and a SEQUENCE of equal-length SEQUENCES
                     for IMAGE attributes.
                     The recommended sequence is a C continuous and aligned numpy
                     array, as it can be optimized.
            - str_data : (str) special variation for DevEncoded data type. In this case 'data' must
                         be a str or an object with the buffer interface.
            - dim_x : (int) [DEPRECATED] the attribute x length. Default value is 1
            - dim_y : (int) [DEPRECATED] the attribute y length. Default value is 0
            - time_stamp : (double) the time stamp
            - quality : (AttrQuality) the attribute quality factor
        Return     : None
    """)

    document_method("set_change_event", """
    set_change_event(self, implemented, detect = True) -> None

            Set a flag to indicate that the server fires change events manually,
            without the polling to be started for the attribute.
            If the detect parameter is set to true, the criteria specified for
            the change event are verified and the event is only pushed if they
            are fullfilled. If detect is set to false the event is fired without
            any value checking!

        Parameters :
            - implemented : (bool) True when the server fires change events manually.
            - detect : (bool) (optional, default is True) Triggers the verification of
                       the change event properties when set to true.
        Return     : None

        New in PyTango 7.1.0
    """)

    document_method("set_archive_event", """
    set_archive_event(self, implemented, detect = True) -> None

            Set a flag to indicate that the server fires archive events manually,
            without the polling to be started for the attribute If the detect parameter
            is set to true, the criteria specified for the archive event are verified
            and the event is only pushed if they are fullfilled.

        Parameters :
            - implemented : (bool) True when the server fires archive events manually.
            - detect : (bool) (optional, default is True) Triggers the verification of
                       the archive event properties when set to true.
        Return     : None

        New in PyTango 7.1.0
    """)

    document_method("is_change_event", """
    is_change_event(self) -> bool

            Check if the change event is fired manually (without polling) for this attribute.

        Parameters : None
        Return     : (bool) True if a manual fire change event is implemented.

        New in PyTango 7.1.0
    """)

    document_method("is_check_change_criteria", """
    is_check_change_criteria(self) -> bool

            Check if the change event criteria should be checked when firing the
            event manually.

        Parameters : None
        Return     : (bool) True if a change event criteria will be checked.

        New in PyTango 7.1.0
    """)

    document_method("is_archive_event", """
    is_archive_event(self) -> bool

            Check if the archive event is fired manually (without polling) for this attribute.

        Parameters : None
        Return     : (bool) True if a manual fire archive event is implemented.

        New in PyTango 7.1.0
    """)

    document_method("is_check_archive_criteria", """
    is_check_archive_criteria(self) -> bool

            Check if the archive event criteria should be checked when firing the
            event manually.

        Parameters : None
        Return     : (bool) True if a archive event criteria will be checked.

        New in PyTango 7.1.0
    """)

    document_method("set_data_ready_event", """
    set_data_ready_event(self, implemented) -> None

            Set a flag to indicate that the server fires data ready events.

        Parameters :
            - implemented : (bool) True when the server fires data ready events manually.
        Return     : None

        New in PyTango 7.2.0
    """)

    document_method("is_data_ready_event", """
    is_data_ready_event(self) -> bool

            Check if the data ready event is fired manually (without polling)
            for this attribute.

        Parameters : None
        Return     : (bool) True if a manual fire data ready event is implemented.

        New in PyTango 7.2.0
    """)

    document_method("remove_configuration", """
    remove_configuration(self) -> None

            Remove the attribute configuration from the database.
            This method can be used to clean-up all the configuration of an
            attribute to come back to its default values or the remove all
            configuration of a dynamic attribute before deleting it.

            The method removes all configured attribute properties and removes
            the attribute from the list of polled attributes.

        Parameters : None
        Return     : None

        New in PyTango 7.1.0
    """)


def __doc_WAttribute():
    def document_method(method_name, desc, append=True):
        return __document_method(WAttribute, method_name, desc, append)

    WAttribute.__doc__ = """
    This class represents a Tango writable attribute.
    """

    document_method("get_min_value", """
    get_min_value(self) -> obj

            Get attribute minimum value or throws an exception if the
            attribute does not have a minimum value.

        Parameters : None
        Return     : (obj) an object with the python minimum value
    """)

    document_method("get_max_value", """
    get_max_value(self) -> obj

            Get attribute maximum value or throws an exception if the
            attribute does not have a maximum value.

        Parameters : None
        Return     : (obj) an object with the python maximum value
    """)

    document_method("set_min_value", """
    set_min_value(self, data) -> None

            Set attribute minimum value.

        Parameters :
            - data : the attribute minimum value. python data type must be compatible
                     with the attribute data format and type.
        Return     : None
    """)

    document_method("set_max_value", """
    set_max_value(self, data) -> None

            Set attribute maximum value.

        Parameters :
            - data : the attribute maximum value. python data type must be compatible
                     with the attribute data format and type.
        Return     : None
    """)

    document_method("is_min_value", """
    is_min_value(self) -> bool

            Check if the attribute has a minimum value.

        Parameters : None
        Return     : (bool) true if the attribute has a minimum value defined
    """)

    document_method("is_max_value", """
    is_max_value(self, ) -> bool

            Check if the attribute has a maximum value.


        Parameters : None
        Return     : (bool) true if the attribute has a maximum value defined
    """)

    document_method("get_write_value_length", """
    get_write_value_length(self) -> int

            Retrieve the new value length (data number) for writable attribute.

        Parameters : None
        Return     : (int) the new value data length
    """)

    #    document_method("set_write_value", """
    #    set_write_value(self, data, dim_x = 1, dim_y = 0) -> None
    #
    #            Set the writable attribute value.
    #
    #        Parameters :
    #            - data : the data to be set. Data must be compatible with the attribute type and format.
    #                     for SPECTRUM and IMAGE attributes, data can be any type of sequence of elements
    #                     compatible with the attribute type
    #            - dim_x : (int) the attribute set value x length. Default value is 1
    #            - dim_y : (int) the attribute set value y length. Default value is 0
    #        Return     : None
    #    """)

    document_method("get_write_value", """
    get_write_value(self, lst) -> None  <= DEPRECATED
    get_write_value(self, extract_as=ExtractAs.Numpy) -> obj

            Retrieve the new value for writable attribute.

        Parameters :
            - extract_as: (ExtractAs)
            - lst : [out] (list) a list object that will be filled with the attribute write value (DEPRECATED)
        Return     : (obj) the attribute write value.
    """)


def __doc_MultiClassAttribute():
    def document_method(method_name, desc, append=True):
        return __document_method(MultiClassAttribute, method_name, desc, append)

    MultiClassAttribute.__doc__ = """
    There is one instance of this class for each device class.
    This class is mainly an aggregate of :class:`~tango.Attr` objects.
    It eases management of multiple attributes

    New in PyTango 7.2.1"""

    document_method("get_attr", """
    get_attr(self, attr_name) -> Attr

            Get the :class:`~tango.Attr` object for the attribute with
            name passed as parameter

        Parameters :
            - attr_name : (str) attribute name
        Return     : (Attr) the attribute object

        Throws     : DevFailed If the attribute is not defined.

        New in PyTango 7.2.1
    """)

    document_method("remove_attr", """
    remove_attr(self, attr_name, cl_name) -> None

            Remove the :class:`~tango.Attr` object for the attribute with
            name passed as parameter. Does nothing if the attribute does not
            exist.

        Parameters :
            - attr_name : (str) attribute name
            - cl_name : (str) the attribute class name

        New in PyTango 7.2.1
    """)

    document_method("get_attr_list", """
    get_attr_list(self) -> seq<Attr>

            Get the list of :class:`~tango.Attr` for this device class.

        Return     : (seq<Attr>) the list of attribute objects

        New in PyTango 7.2.1
    """)


def __doc_MultiAttribute():
    def document_method(method_name, desc, append=True):
        return __document_method(MultiAttribute, method_name, desc, append)

    MultiAttribute.__doc__ = """
    There is one instance of this class for each device.
    This class is mainly an aggregate of :class:`~tango.Attribute` or
    :class:`~tango.WAttribute` objects. It eases management of multiple
    attributes"""

    document_method("get_attr_by_name", """
    get_attr_by_name(self, attr_name) -> Attribute

            Get :class:`~tango.Attribute` object from its name.
            This method returns an :class:`~tango.Attribute` object with a
            name passed as parameter. The equality on attribute name is case
            independant.

        Parameters :
            - attr_name : (str) attribute name
        Return     : (Attribute) the attribute object

        Throws     : DevFailed If the attribute is not defined.
    """)

    document_method("get_attr_by_ind", """
    get_attr_by_ind(self, ind) -> Attribute

            Get :class:`~tango.Attribute` object from its index.
            This method returns an :class:`~tango.Attribute` object from the
            index in the main attribute vector.

        Parameters :
            - ind : (int) the attribute index
        Return     : (Attribute) the attribute object
    """)

    document_method("get_w_attr_by_name", """
    get_w_attr_by_name(self, attr_name) -> WAttribute

            Get a writable attribute object from its name.
            This method returns an :class:`~tango.WAttribute` object with a
            name passed as parameter. The equality on attribute name is case
            independant.

        Parameters :
            - attr_name : (str) attribute name
        Return     : (WAttribute) the attribute object

        Throws     : DevFailed If the attribute is not defined.
    """)

    document_method("get_w_attr_by_ind", """
    get_w_attr_by_ind(self, ind) -> WAttribute

            Get a writable attribute object from its index.
            This method returns an :class:`~tango.WAttribute` object from the
            index in the main attribute vector.

        Parameters :
            - ind : (int) the attribute index
        Return     : (WAttribute) the attribute object
    """)

    document_method("get_attr_ind_by_name", """
    get_attr_ind_by_name(self, attr_name) -> int

            Get Attribute index into the main attribute vector from its name.
            This method returns the index in the Attribute vector (stored in the
            :class:`~tango.MultiAttribute` object) of an attribute with a
            given name. The name equality is case independant.

        Parameters :
            - attr_name : (str) attribute name
        Return     : (int) the attribute index

        Throws     : DevFailed If the attribute is not found in the vector.

        New in PyTango 7.0.0
    """)

    document_method("get_attr_nb", """
    get_attr_nb(self) -> int

            Get attribute number.

        Parameters : None
        Return     : (int) the number of attributes

        New in PyTango 7.0.0
    """)

    document_method("check_alarm", """
    check_alarm(self) -> bool
    check_alarm(self, attr_name) -> bool
    check_alarm(self, ind) -> bool

            - The 1st version of the method checks alarm on all attribute(s) with an alarm defined.
            - The 2nd version of the method checks alarm for one attribute with a given name.
            - The 3rd version of the method checks alarm for one attribute from its index in the main attributes vector.

        Parameters :
            - attr_name : (str) attribute name
            - ind : (int) the attribute index
        Return     : (bool) True if at least one attribute is in alarm condition

        Throws     : DevFailed If at least one attribute does not have any alarm level defined

        New in PyTango 7.0.0
    """)

    document_method("read_alarm", """
    read_alarm(self, status) -> None

            Add alarm message to device status.
            This method add alarm mesage to the string passed as parameter.
            A message is added for each attribute which is in alarm condition

        Parameters :
            - status : (str) a string (should be the device status)
        Return     : None

        New in PyTango 7.0.0
    """)

    document_method("get_attribute_list", """
    get_attribute_list(self) -> seq<Attribute>

            Get the list of attribute objects.

        Return     : (seq<Attribute>) list of attribute objects

        New in PyTango 7.2.1
    """)


def __doc_Attr():
    def document_method(method_name, desc, append=True):
        return __document_method(Attr, method_name, desc, append)

    Attr.__doc__ = """
    This class represents a Tango writable attribute.
    """

    document_method("set_default_properties", """
    set_default_properties(self) -> None

            Set default attribute properties.

        Parameters :
            - attr_prop : (UserDefaultAttrProp) the user default property class
        Return     : None
    """)

    document_method("set_disp_level", """
    set_disp_level(self, disp_lelel) -> None

            Set the attribute display level.

        Parameters :
            - disp_level : (DispLevel) the new display level
        Return     : None
    """)

    document_method("set_polling_period", """
    set_polling_period(self, period) -> None

            Set the attribute polling update period.

        Parameters :
            - period : (int) the attribute polling period (in mS)
        Return     : None
    """)

    document_method("set_memorized", """
    set_memorized(self) -> None

            Set the attribute as memorized in database (only for scalar
            and writable attribute) With no argument the setpoint will be
            written to the attribute during initialisation!

        Parameters : None
        Return     : None
    """)

    document_method("set_memorized_init", """
    set_memorized_init(self, write_on_init) -> None

            Set the initialisation flag for memorized attributes
            true = the setpoint value will be written to the attribute on initialisation
            false = only the attribute setpoint is initialised.
            No action is taken on the attribute

        Parameters :
            - write_on_init : (bool) if true the setpoint value will be written
                              to the attribute on initialisation
        Return     : None
    """)

    document_method("set_change_event", """
    set_change_event(self, implemented, detect) -> None

            Set a flag to indicate that the server fires change events manually
            without the polling to be started for the attribute.
            If the detect parameter is set to true, the criteria specified for
            the change event are verified and the event is only pushed if they
            are fullfilled.

            If detect is set to false the event is fired without checking!

        Parameters :
            - implemented : (bool) True when the server fires change events manually.
            - detect : (bool) Triggers the verification of the change event properties
                       when set to true.
        Return     : None
    """)

    document_method("is_change_event", """
    is_change_event(self) -> bool

            Check if the change event is fired manually for this attribute.

        Parameters : None
        Return     : (bool) true if a manual fire change event is implemented.
    """)

    document_method("is_check_change_criteria", """
    is_check_change_criteria(self) -> bool

            Check if the change event criteria should be checked when firing the event manually.

        Parameters : None
        Return     : (bool) true if a change event criteria will be checked.
    """)

    document_method("set_archive_event", """
    set_archive_event(self) -> None

            Set a flag to indicate that the server fires archive events manually
            without the polling to be started for the attribute If the detect
            parameter is set to true, the criteria specified for the archive
            event are verified and the event is only pushed if they are fullfilled.

            If detect is set to false the event is fired without checking!

        Parameters :
            - implemented : (bool) True when the server fires change events manually.
            - detect : (bool) Triggers the verification of the archive event properties
                       when set to true.
        Return     : None
    """)

    document_method("is_archive_event", """
    is_archive_event(self) -> bool

            Check if the archive event is fired manually for this attribute.

        Parameters : None
        Return     : (bool) true if a manual fire archive event is implemented.
    """)

    document_method("is_check_archive_criteria", """
    is_check_archive_criteria(self) -> bool

            Check if the archive event criteria should be checked when firing the event manually.

        Parameters : None
        Return     : (bool) true if a archive event criteria will be checked.
    """)

    document_method("set_data_ready_event", """
    set_data_ready_event(self, implemented) -> None

            Set a flag to indicate that the server fires data ready events.

        Parameters :
            - implemented : (bool) True when the server fires data ready events
        Return     : None

        New in PyTango 7.2.0
    """)

    document_method("is_data_ready_event", """
    is_data_ready_event(self) -> bool

            Check if the data ready event is fired for this attribute.

        Parameters : None
        Return     : (bool) true if firing data ready event is implemented.

        New in PyTango 7.2.0
    """)

    document_method("get_name", """
    get_name(self) -> str

            Get the attribute name.

        Parameters : None
        Return     : (str) the attribute name
    """)

    document_method("get_format", """
    get_format(self) -> AttrDataFormat

            Get the attribute format

        Parameters : None
        Return     : (AttrDataFormat) the attribute format
    """)

    document_method("get_writable", """
    get_writable(self) -> AttrWriteType

            Get the attribute write type

        Parameters : None
        Return     : (AttrWriteType) the attribute write type
    """)

    document_method("get_type", """
    get_type(self) -> int

            Get the attribute data type

        Parameters : None
        Return     : (int) the attribute data type
    """)

    document_method("get_disp_level", """
    get_disp_level(self) -> DispLevel

            Get the attribute display level

        Parameters : None
        Return     : (DispLevel) the attribute display level
    """)

    document_method("get_polling_period", """
    get_polling_period(self) -> int

            Get the polling period (mS)

        Parameters : None
        Return     : (int) the polling period (mS)
    """)

    document_method("get_memorized", """
    get_memorized(self) -> bool

            Determine if the attribute is memorized or not.

        Parameters : None
        Return     : (bool) True if the attribute is memorized
    """)

    document_method("get_memorized_init", """
    get_memorized_init(self) -> bool

            Determine if the attribute is written at startup from the memorized value if
            it is memorized

        Parameters : None
        Return     : (bool) True if initialized with memorized value or not
    """)

    document_method("get_assoc", """
    get_assoc(self) -> str

            Get the associated name.

        Parameters : None
        Return     : (bool) the associated name
    """)

    document_method("is_assoc", """
    is_assoc(self) -> bool

            Determine if it is assoc.

        Parameters : None
        Return     : (bool) if it is assoc
    """)

    document_method("get_cl_name", """
    get_cl_name(self) -> str

            Returns the class name

        Parameters : None
        Return     : (str) the class name

        New in PyTango 7.2.0
    """)

    document_method("set_cl_name", """
    set_cl_name(self, cl) -> None

            Sets the class name

        Parameters :
            - cl : (str) new class name
        Return     : None

        New in PyTango 7.2.0
    """)

    document_method("get_class_properties", """
    get_class_properties(self) -> sequence<AttrProperty>

            Get the class level attribute properties

        Parameters : None
        Return     : (sequence<AttrProperty>) the class attribute properties
    """)

    document_method("get_user_default_properties", """
    get_user_default_properties(self) -> sequence<AttrProperty>

            Get the user default attribute properties

        Parameters : None
        Return     : (sequence<AttrProperty>) the user default attribute properties
    """)

    document_method("set_class_properties", """
    set_class_properties(self, props) -> None

            Set the class level attribute properties

        Parameters :
            - props : (StdAttrPropertyVector) new class level attribute properties
        Return     : None
    """)


def __doc_UserDefaultAttrProp():
    def document_method(method_name, desc, append=True):
        return __document_method(UserDefaultAttrProp, method_name, desc, append)

    UserDefaultAttrProp.__doc__ = """
    User class to set attribute default properties.
    This class is used to set attribute default properties.
    Three levels of attributes properties setting are implemented within Tango.
    The highest property setting level is the database.
    Then the user default (set using this UserDefaultAttrProp class) and finally
    a Tango library default value
    """

    document_method("set_label", """
    set_label(self, def_label) -> None

            Set default label property.

        Parameters :
            - def_label : (str) the user default label property
        Return     : None
    """)

    document_method("set_description", """
    set_description(self, def_description) -> None

            Set default description property.

        Parameters :
            - def_description : (str) the user default description property
        Return     : None
    """)

    document_method("set_format", """
    set_format(self, def_format) -> None

            Set default format property.

        Parameters :
            - def_format : (str) the user default format property
        Return     : None
    """)

    document_method("set_unit", """
    set_unit(self, def_unit) -> None

            Set default unit property.

        Parameters :
            - def_unit : (str) te user default unit property
        Return     : None
    """)

    document_method("set_standard_unit", """
    set_standard_unit(self, def_standard_unit) -> None

            Set default standard unit property.

        Parameters :
            - def_standard_unit : (str) the user default standard unit property
        Return     : None
    """)

    document_method("set_display_unit", """
    set_display_unit(self, def_display_unit) -> None

            Set default display unit property.

        Parameters :
            - def_display_unit : (str) the user default display unit property
        Return     : None
    """)

    document_method("set_min_value", """
    set_min_value(self, def_min_value) -> None

            Set default min_value property.

        Parameters :
            - def_min_value : (str) the user default min_value property
        Return     : None
    """)

    document_method("set_max_value", """
    set_max_value(self, def_max_value) -> None

            Set default max_value property.

        Parameters :
            - def_max_value : (str) the user default max_value property
        Return     : None
    """)

    document_method("set_min_alarm", """
    set_min_alarm(self, def_min_alarm) -> None

            Set default min_alarm property.

        Parameters :
            - def_min_alarm : (str) the user default min_alarm property
        Return     : None
    """)

    document_method("set_max_alarm", """
    set_max_alarm(self, def_max_alarm) -> None

            Set default max_alarm property.

        Parameters :
            - def_max_alarm : (str) the user default max_alarm property
        Return     : None
    """)

    document_method("set_min_warning", """
    set_min_warning(self, def_min_warning) -> None

            Set default min_warning property.

        Parameters :
            - def_min_warning : (str) the user default min_warning property
        Return     : None
    """)

    document_method("set_max_warning", """
    set_max_warning(self, def_max_warning) -> None

            Set default max_warning property.

        Parameters :
            - def_max_warning : (str) the user default max_warning property
        Return     : None
    """)

    document_method("set_delta_t", """
    set_delta_t(self, def_delta_t) -> None

            Set default RDS alarm delta_t property.

        Parameters :
            - def_delta_t : (str) the user default RDS alarm delta_t property
        Return     : None
    """)

    document_method("set_delta_val", """
    set_delta_val(self, def_delta_val) -> None

            Set default RDS alarm delta_val property.

        Parameters :
            - def_delta_val : (str) the user default RDS alarm delta_val property
        Return     : None
    """)

    document_method("set_abs_change", """
    set_abs_change(self, def_abs_change) -> None <= DEPRECATED

            Set default change event abs_change property.

        Parameters :
            - def_abs_change : (str) the user default change event abs_change property
        Return     : None

        Deprecated since PyTango 8.0. Please use set_event_abs_change instead.
    """)

    document_method("set_event_abs_change", """
    set_event_abs_change(self, def_abs_change) -> None

            Set default change event abs_change property.

        Parameters :
            - def_abs_change : (str) the user default change event abs_change property
        Return     : None

        New in PyTango 8.0
    """)

    document_method("set_rel_change", """
    set_rel_change(self, def_rel_change) -> None <= DEPRECATED

            Set default change event rel_change property.

        Parameters :
            - def_rel_change : (str) the user default change event rel_change property
        Return     : None

        Deprecated since PyTango 8.0. Please use set_event_rel_change instead.
    """)

    document_method("set_event_rel_change", """
    set_event_rel_change(self, def_rel_change) -> None

            Set default change event rel_change property.

        Parameters :
            - def_rel_change : (str) the user default change event rel_change property
        Return     : None

        New in PyTango 8.0
    """)

    document_method("set_period", """
    set_period(self, def_period) -> None <= DEPRECATED

            Set default periodic event period property.

        Parameters :
            - def_period : (str) the user default periodic event period property
        Return     : None

        Deprecated since PyTango 8.0. Please use set_event_period instead.
    """)

    document_method("set_event_period", """
    set_event_period(self, def_period) -> None

            Set default periodic event period property.

        Parameters :
            - def_period : (str) the user default periodic event period property
        Return     : None

        New in PyTango 8.0
    """)

    document_method("set_archive_abs_change", """
    set_archive_abs_change(self, def_archive_abs_change) -> None <= DEPRECATED

            Set default archive event abs_change property.

        Parameters :
            - def_archive_abs_change : (str) the user default archive event abs_change property
        Return     : None

        Deprecated since PyTango 8.0. Please use set_archive_event_abs_change instead.
    """)

    document_method("set_archive_event_abs_change", """
    set_archive_event_abs_change(self, def_archive_abs_change) -> None

            Set default archive event abs_change property.

        Parameters :
            - def_archive_abs_change : (str) the user default archive event abs_change property
        Return     : None

        New in PyTango 8.0
    """)

    document_method("set_archive_rel_change", """
    set_archive_rel_change(self, def_archive_rel_change) -> None <= DEPRECATED

            Set default archive event rel_change property.

        Parameters :
            - def_archive_rel_change : (str) the user default archive event rel_change property
        Return     : None

        Deprecated since PyTango 8.0. Please use set_archive_event_rel_change instead.
    """)

    document_method("set_archive_event_rel_change", """
    set_archive_event_rel_change(self, def_archive_rel_change) -> None

            Set default archive event rel_change property.

        Parameters :
            - def_archive_rel_change : (str) the user default archive event rel_change property
        Return     : None

        New in PyTango 8.0
    """)

    document_method("set_archive_period", """
    set_archive_period(self, def_archive_period) -> None <= DEPRECATED

            Set default archive event period property.

        Parameters :
            - def_archive_period : (str) t
        Return     : None

        Deprecated since PyTango 8.0. Please use set_archive_event_period instead.
    """)

    document_method("set_archive_event_period", """
    set_archive_event_period(self, def_archive_period) -> None

            Set default archive event period property.

        Parameters :
            - def_archive_period : (str) t
        Return     : None

        New in PyTango 8.0
    """)


def device_server_init(doc=True):
    __init_DeviceImpl()
    __init_Attribute()
    __init_Attr()
    __init_UserDefaultAttrProp()
    __init_Logger()
    if doc:
        __doc_DeviceImpl()
        __doc_extra_DeviceImpl(Device_3Impl)
        __doc_extra_DeviceImpl(Device_4Impl)
        __doc_extra_DeviceImpl(Device_5Impl)
        __doc_Attribute()
        __doc_WAttribute()
        __doc_MultiAttribute()
        __doc_MultiClassAttribute()
        __doc_UserDefaultAttrProp()
        __doc_Attr()
