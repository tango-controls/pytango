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

__all__ = ("DeviceClass", "device_class_init")

__docformat__ = "restructuredtext"

import collections

from ._tango import Except, DevFailed, DeviceClass, CmdArgType, \
    DispLevel, UserDefaultAttrProp
from .pyutil import Util

from .utils import is_pure_str, is_non_str_seq, seqStr_2_obj, obj_2_str, \
    is_array
from .utils import document_method as __document_method

from .globals import get_class, get_class_by_class, \
    get_constructed_class_by_class
from .attr_data import AttrData
from .pipe_data import PipeData


class PropUtil:
    """An internal Property util class"""

    scalar_int_types = (CmdArgType.DevShort, CmdArgType.DevUShort,
                        CmdArgType.DevInt, CmdArgType.DevLong, CmdArgType.DevULong,)

    scalar_long_types = (CmdArgType.DevLong64, CmdArgType.DevULong64)

    scalar_float_types = (CmdArgType.DevFloat, CmdArgType.DevDouble,)

    scalar_numerical_types = scalar_int_types + scalar_long_types + scalar_float_types

    scalar_str_types = (CmdArgType.DevString, CmdArgType.ConstDevString,)

    scalar_types = scalar_numerical_types + scalar_str_types + \
        (CmdArgType.DevBoolean, CmdArgType.DevEncoded,
         CmdArgType.DevUChar, CmdArgType.DevVoid)

    def __init__(self):
        self.db = None
        if Util._UseDb:
            self.db = Util.instance().get_database()

    def set_default_property_values(self, dev_class, class_prop, dev_prop):
        """
            set_default_property_values(self, dev_class, class_prop, dev_prop) -> None

                Sets the default property values

            Parameters :
                - dev_class : (DeviceClass) device class object
                - class_prop : (dict<str,>) class properties
                - dev_prop : (dict<str,>) device properties

            Return     : None
        """
        for name in class_prop:
            type = self.get_property_type(name, class_prop)
            val = self.get_property_values(name, class_prop)
            val = self.values2string(val, type)
            desc = self.get_property_description(name, class_prop)
            dev_class.add_wiz_class_prop(name, desc, val)

        for name in dev_prop:
            type = self.get_property_type(name, dev_prop)
            val = self.get_property_values(name, dev_prop)
            val = self.values2string(val, type)
            desc = self.get_property_description(name, dev_prop)
            dev_class.add_wiz_dev_prop(name, desc, val)

    def get_class_properties(self, dev_class, class_prop):
        """
            get_class_properties(self, dev_class, class_prop) -> None

                    Returns the class properties

                Parameters :
                    - dev_class : (DeviceClass) the DeviceClass object
                    - class_prop : [in, out] (dict<str, None>) the property names. Will be filled
                                   with property values

                Return     : None"""
        # initialize default values
        if class_prop == {} or not Util._UseDb:
            return

        # call database to get properties
        props = self.db.get_class_property(dev_class.get_name(), list(class_prop.keys()))

        # if value defined in database, store it
        for name in class_prop:
            if props[name]:
                type = self.get_property_type(name, class_prop)
                values = self.stringArray2values(props[name], type)
                self.set_property_values(name, class_prop, values)
            else:
                print(name + " property NOT found in database")

    def get_device_properties(self, dev, class_prop, dev_prop):
        """
            get_device_properties(self, dev, class_prop, dev_prop) -> None

                    Returns the device properties

                Parameters :
                    - dev : (DeviceImpl) the device object
                    - class_prop : (dict<str, obj>) the class properties
                    - dev_prop : [in,out] (dict<str, None>) the device property names

                Return     : None"""
        #    initialize default properties
        if dev_prop == {} or not Util._UseDb:
            return

        # Call database to get properties
        props = self.db.get_device_property(dev.get_name(), list(dev_prop.keys()))
        #    if value defined in database, store it
        for name in dev_prop:
            prop_value = props[name]
            if len(prop_value):
                data_type = self.get_property_type(name, dev_prop)
                values = self.stringArray2values(prop_value, data_type)
                if not self.is_empty_seq(values):
                    self.set_property_values(name, dev_prop, values)
                else:
                    #    Try to get it from class property
                    values = self.get_property_values(name, class_prop)
                    if not self.is_empty_seq(values):
                        if not self.is_seq(values):
                            values = [values]
                        data_type = self.get_property_type(name, class_prop)
                        values = self.stringArray2values(values, data_type)
                        if not self.is_empty_seq(values):
                            self.set_property_values(name, dev_prop, values)
            else:
                #    Try to get it from class property
                values = self.get_property_values(name, class_prop)
                if not self.is_empty_seq(values):
                    if not self.is_seq(values):
                        values = [values]
                    data_type = self.get_property_type(name, class_prop)
                    values = self.stringArray2values(values, data_type)
                    if not self.is_empty_seq(values):
                        self.set_property_values(name, dev_prop, values)

    def is_seq(self, v):
        """
            is_seq(self, v) -> bool

                    Helper method. Determines if the object is a sequence

                Parameters :
                    - v : (object) the object to be analysed

                Return     : (bool) True if the object is a sequence or False otherwise"""
        return isinstance(v, collections.Sequence)

    def is_empty_seq(self, v):
        """
            is_empty_seq(self, v) -> bool

                    Helper method. Determines if the object is an empty sequence

                Parameters :
                    - v : (object) the object to be analysed

                Return     : (bool) True if the object is a sequence which is empty or False otherwise"""
        return self.is_seq(v) and not len(v)

    def get_property_type(self, prop_name, properties):
        """
            get_property_type(self, prop_name, properties) -> CmdArgType

                    Gets the property type for the given property name using the
                    information given in properties

                Parameters :
                    - prop_name : (str) property name
                    - properties : (dict<str,data>) property data

                Return     : (CmdArgType) the tango type for the given property"""
        try:
            tg_type = properties[prop_name][0]
        except:
            tg_type = CmdArgType.DevVoid
        return tg_type

    def set_property_values(self, prop_name, properties, values):
        """
            set_property_values(self, prop_name, properties, values) -> None

                    Sets the property value in the properties

                Parameters :
                    - prop_name : (str) property name
                    - properties : (dict<str,obj>) [in,out] dict which will contain the value
                    - values : (seq) the new property value

                Return     : None"""

        properties[prop_name][2] = values

    def get_property_values(self, prop_name, properties):
        """
            get_property_values(self, prop_name, properties) -> obj

                    Gets the property value

                Parameters :
                    - prop_name : (str) property name
                    - properties : (dict<str,obj>) properties
                Return     : (obj) the value for the given property name"""
        try:
            tg_type = self.get_property_type(prop_name, properties)
            val = properties[prop_name][2]
        except:
            val = []

        if is_array(tg_type) or (isinstance(val, collections.Sequence) and not len(val)):
            return val
        else:
            if is_non_str_seq(val):
                return val[0]
            else:
                return val

    def get_property_description(self, prop_name, properties):
        """
            get_property_description(self, prop_name, properties) -> obj

                    Gets the property description

                Parameters :
                    - prop_name : (str) property name
                    - properties : (dict<str,obj>) properties
                Return     : (str) the description for the given property name"""
        return properties[prop_name][1]

    def stringArray2values(self, argin, argout_type):
        """internal helper method"""
        return seqStr_2_obj(argin, argout_type)

    def values2string(self, argin, argout_type):
        """internal helper method"""
        return obj_2_str(argin, argout_type)


def __DeviceClass__init__(self, name):
    DeviceClass.__init_orig__(self, name)
    self.dyn_att_added_methods = []
    self.dyn_cmd_added_methods = []
    try:
        pu = self.prop_util = PropUtil()
        self.py_dev_list = []
        pu.set_default_property_values(self, self.class_property_list,
                                       self.device_property_list)
        pu.get_class_properties(self, self.class_property_list)
        for prop_name in self.class_property_list:
            if not hasattr(self, prop_name):
                setattr(self, prop_name, pu.get_property_values(prop_name,
                                                                self.class_property_list))
    except DevFailed as df:
        print("PyDS: %s: A Tango error occured in the constructor:" % name)
        Except.print_exception(df)
    except Exception as e:
        print("PyDS: %s: An error occured in the constructor:" % name)
        print(str(e))


def __DeviceClass__str__(self):
    return '%s(%s)' % (self.__class__.__name__, self.get_name())


def __DeviceClass__repr__(self):
    return '%s(%s)' % (self.__class__.__name__, self.get_name())


def __throw_create_attribute_exception(msg):
    """
    Helper method to throw DevFailed exception when inside
    create_attribute
    """
    Except.throw_exception("PyDs_WrongAttributeDefinition", msg,
                           "create_attribute()")


def __throw_create_command_exception(msg):
    """
    Helper method to throw DevFailed exception when inside
    create_command
    """
    Except.throw_exception("PyDs_WrongCommandDefinition", msg,
                           "create_command()")


def __DeviceClass__create_user_default_attr_prop(self, attr_name, extra_info):
    """for internal usage only"""
    p = UserDefaultAttrProp()
    for k, v in extra_info.items():
        k_lower = k.lower()
        method_name = "set_%s" % k_lower.replace(' ', '_')
        if hasattr(p, method_name):
            method = getattr(p, method_name)
            method(str(v))
        elif k == 'delta_time':
            p.set_delta_t(str(v))
        elif k_lower not in ('display level', 'polling period', 'memorized'):
            name = self.get_name()
            msg = "Wrong definition of attribute %s in " \
                  "class %s\nThe object extra information '%s' " \
                  "is not recognized!" % (attr_name, name, k)
            self.__throw_create_attribute_exception(msg)
    return p


def __DeviceClass__attribute_factory(self, attr_list):
    """for internal usage only"""
    for attr_name, attr_info in self.attr_list.items():
        if isinstance(attr_info, AttrData):
            attr_data = attr_info
        else:
            attr_data = AttrData(attr_name, self.get_name(), attr_info)
        if attr_data.forward:
            self._create_fwd_attribute(attr_list, attr_data.name, attr_data.att_prop)
        else:
            self._create_attribute(attr_list, attr_data.attr_name,
                                   attr_data.attr_type,
                                   attr_data.attr_format,
                                   attr_data.attr_write,
                                   attr_data.dim_x, attr_data.dim_y,
                                   attr_data.display_level,
                                   attr_data.polling_period,
                                   attr_data.memorized,
                                   attr_data.hw_memorized,
                                   attr_data.read_method_name,
                                   attr_data.write_method_name,
                                   attr_data.is_allowed_name,
                                   attr_data.att_prop)


def __DeviceClass__pipe_factory(self, pipe_list):
    """for internal usage only"""
    for pipe_name, pipe_info in self.pipe_list.items():
        if isinstance(pipe_info, PipeData):
            pipe_data = pipe_info
        else:
            pipe_data = PipeData(pipe_name, self.get_name(), pipe_info)
        self._create_pipe(pipe_list, pipe_data.pipe_name,
                          pipe_data.pipe_write,
                          pipe_data.display_level,
                          pipe_data.read_method_name,
                          pipe_data.write_method_name,
                          pipe_data.is_allowed_name,
                          pipe_data.pipe_prop)


def __DeviceClass__command_factory(self):
    """for internal usage only"""
    name = self.get_name()
    class_info = get_class(name)
    deviceimpl_class = class_info[1]

    if not hasattr(deviceimpl_class, "init_device"):
        msg = "Wrong definition of class %s\n" \
              "The init_device() method does not exist!" % name
        Except.throw_exception("PyDs_WrongCommandDefinition", msg, "command_factory()")

    for cmd_name, cmd_info in self.cmd_list.items():
        __create_command(self, deviceimpl_class, cmd_name, cmd_info)


def __create_command(self, deviceimpl_class, cmd_name, cmd_info):
    """for internal usage only"""
    name = self.get_name()

    # check for well defined command info

    # check parameter
    if not isinstance(cmd_info, collections.Sequence):
        msg = "Wrong data type for value for describing command %s in " \
              "class %s\nMust be a sequence with 2 or 3 elements" % (cmd_name, name)
        __throw_create_command_exception(msg)

    if len(cmd_info) < 2 or len(cmd_info) > 3:
        msg = "Wrong number of argument for describing command %s in " \
              "class %s\nMust be a sequence with 2 or 3 elements" % (cmd_name, name)
        __throw_create_command_exception(msg)

    param_info, result_info = cmd_info[0], cmd_info[1]

    if not isinstance(param_info, collections.Sequence):
        msg = "Wrong data type in command argument for command %s in " \
              "class %s\nCommand parameter (first element) must be a sequence" % (cmd_name, name)
        __throw_create_command_exception(msg)

    if len(param_info) < 1 or len(param_info) > 2:
        msg = "Wrong data type in command argument for command %s in " \
              "class %s\nSequence describing command parameters must contain " \
              "1 or 2 elements"
        __throw_create_command_exception(msg)

    param_type = CmdArgType.DevVoid
    try:
        param_type = CmdArgType(param_info[0])
    except:
        msg = "Wrong data type in command argument for command %s in " \
              "class %s\nCommand parameter type (first element in first " \
              "sequence) must be a tango.CmdArgType"
        __throw_create_command_exception(msg)

    param_desc = ""
    if len(param_info) > 1:
        param_desc = param_info[1]
        if not is_pure_str(param_desc):
            msg = "Wrong data type in command parameter for command %s in " \
                  "class %s\nCommand parameter description (second element " \
                  "in first sequence), when given, must be a string"
            __throw_create_command_exception(msg)

    # Check result
    if not isinstance(result_info, collections.Sequence):
        msg = "Wrong data type in command result for command %s in " \
              "class %s\nCommand result (second element) must be a sequence" % (cmd_name, name)
        __throw_create_command_exception(msg)

    if len(result_info) < 1 or len(result_info) > 2:
        msg = "Wrong data type in command result for command %s in " \
              "class %s\nSequence describing command result must contain " \
              "1 or 2 elements" % (cmd_name, name)
        __throw_create_command_exception(msg)

    result_type = CmdArgType.DevVoid
    try:
        result_type = CmdArgType(result_info[0])
    except:
        msg = "Wrong data type in command result for command %s in " \
              "class %s\nCommand result type (first element in second " \
              "sequence) must be a tango.CmdArgType" % (cmd_name, name)
        __throw_create_command_exception(msg)

    result_desc = ""
    if len(result_info) > 1:
        result_desc = result_info[1]
        if not is_pure_str(result_desc):
            msg = "Wrong data type in command result for command %s in " \
                  "class %s\nCommand parameter description (second element " \
                  "in second sequence), when given, must be a string" % (cmd_name, name)
            __throw_create_command_exception(msg)

    # If it is defined, get addictional dictionnary used for optional parameters
    display_level, default_command, polling_period = DispLevel.OPERATOR, False, -1

    if len(cmd_info) == 3:
        extra_info = cmd_info[2]
        if not isinstance(extra_info, collections.Mapping):
            msg = "Wrong data type in command information for command %s in " \
                  "class %s\nCommand information (third element in sequence), " \
                  "when given, must be a dictionary" % (cmd_name, name)
            __throw_create_command_exception(msg)

        if len(extra_info) > 3:
            msg = "Wrong data type in command information for command %s in " \
                  "class %s\nThe optional dictionary can not have more than " \
                  "three elements" % (cmd_name, name)
            __throw_create_command_exception(msg)

        for info_name, info_value in extra_info.items():
            info_name_lower = info_name.lower()
            if info_name_lower == "display level":
                try:
                    display_level = DispLevel(info_value)
                except:
                    msg = "Wrong data type in command information for command %s in " \
                          "class %s\nCommand information for display level is not a " \
                          "tango.DispLevel" % (cmd_name, name)
                    __throw_create_command_exception(msg)
            elif info_name_lower == "default command":
                if not is_pure_str(info_value):
                    msg = "Wrong data type in command information for command %s in " \
                          "class %s\nCommand information for default command is not a " \
                          "string" % (cmd_name, name)
                    __throw_create_command_exception(msg)
                v = info_value.lower()
                default_command = v == 'true'
            elif info_name_lower == "polling period":
                try:
                    polling_period = int(info_value)
                except:
                    msg = "Wrong data type in command information for command %s in " \
                          "class %s\nCommand information for polling period is not an " \
                          "integer" % (cmd_name, name)
                    __throw_create_command_exception(msg)
            else:
                msg = "Wrong data type in command information for command %s in " \
                      "class %s\nCommand information has unknown key " \
                      "%s" % (cmd_name, name, info_name)
                __throw_create_command_exception(msg)

    # check that the method to be executed exists
    try:
        cmd = getattr(deviceimpl_class, cmd_name)
        if not isinstance(cmd, collections.Callable):
            msg = "Wrong definition of command %s in " \
                  "class %s\nThe object exists in class but is not " \
                  "a method!" % (cmd_name, name)
            __throw_create_command_exception(msg)
    except AttributeError:
        msg = "Wrong definition of command %s in " \
              "class %s\nThe command method does not exist!" % (cmd_name, name)
        __throw_create_command_exception(msg)

    is_allowed_name = "is_%s_allowed" % cmd_name
    try:
        is_allowed = getattr(deviceimpl_class, is_allowed_name)
        if not isinstance(is_allowed, collections.Callable):
            msg = "Wrong definition of command %s in " \
                  "class %s\nThe object '%s' exists in class but is " \
                  "not a method!" % (cmd_name, name, is_allowed_name)
            __throw_create_command_exception(msg)
    except:
        is_allowed_name = ""

    self._create_command(cmd_name, param_type, result_type,
                         param_desc, result_desc,
                         display_level, default_command,
                         polling_period, is_allowed_name)


def __DeviceClass__new_device(self, klass, dev_class, dev_name):
    return klass(dev_class, dev_name)


def __DeviceClass__device_factory(self, device_list):
    """for internal usage only"""

    klass = self.__class__
    klass_name = klass.__name__
    info, klass = get_class_by_class(klass), get_constructed_class_by_class(klass)

    if info is None:
        raise RuntimeError("Device class '%s' is not registered" % klass_name)

    if klass is None:
        raise RuntimeError("Device class '%s' as not been constructed" % klass_name)

    deviceClassClass, deviceImplClass, deviceImplName = info
    deviceImplClass._device_class_instance = klass

    tmp_dev_list = []
    for dev_name in device_list:
        device = self._new_device(deviceImplClass, klass, dev_name)
        self._add_device(device)
        tmp_dev_list.append(device)

    self.dyn_attr(tmp_dev_list)

    for dev in tmp_dev_list:
        if Util._UseDb and not Util._FileDb:
            self.export_device(dev)
        else:
            self.export_device(dev, dev.get_name())
    self.py_dev_list += tmp_dev_list


def __DeviceClass__create_device(self, device_name, alias=None, cb=None):
    """
        create_device(self, device_name, alias=None, cb=None) -> None

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
            - device_name : (str) the device name
            - alias : (str) optional alias. Default value is None meaning do not create device alias
            - cb : (callable) a callback that is called AFTER the device is registered
                   in the database and BEFORE the init_device for the newly created
                   device is called. Typically you may want to put device and/or attribute
                   properties in the database here. The callback must receive a parameter:
                   device name (str). Default value is None meaning no callback

        Return     : None"""
    util = Util.instance()
    util.create_device(self.get_name(), device_name, alias=alias, cb=cb)


def __DeviceClass__delete_device(self, device_name):
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
    util = Util.instance()
    util.delete_device(self.get_name(), device_name)


def __DeviceClass__dyn_attr(self, device_list):
    """
        dyn_attr(self,device_list) -> None

            Default implementation does not do anything
            Overwrite in order to provide dynamic attributes

        Parameters :
            - device_list : (seq<DeviceImpl>) sequence of devices of this class

        Return     : None"""
    pass


def __DeviceClass__device_destroyer(self, name):
    """for internal usage only"""
    name = name.lower()
    for d in self.py_dev_list:
        dname = d.get_name().lower()
        if dname == name:
            dev_cl = d.get_device_class()
            # the internal C++ device_destroyer isn't case sensitive so we
            # use the internal DeviceImpl name to make sure the DeviceClass
            # finds it
            dev_cl._device_destroyer(d.get_name())
            self.py_dev_list.remove(d)
            return
    err_mess = "Device " + name + " not in Tango class device list!"
    Except.throw_exception("PyAPI_CantDestroyDevice", err_mess, "DeviceClass.device_destroyer")


def __DeviceClass__device_name_factory(self, dev_name_list):
    """
        device_name_factory(self, dev_name_list) ->  None

            Create device(s) name list (for no database device server).
            This method can be re-defined in DeviceClass sub-class for
            device server started without database. Its rule is to
            initialise class device name. The default method does nothing.

        Parameters :
            - dev_name_list : (seq<str>) sequence of devices to be filled

        Return     : None"""
    pass


def __init_DeviceClass():
    DeviceClass.class_property_list = {}
    DeviceClass.device_property_list = {}
    DeviceClass.cmd_list = {}
    DeviceClass.attr_list = {}
    DeviceClass.pipe_list = {}
    DeviceClass.__init_orig__ = DeviceClass.__init__
    DeviceClass.__init__ = __DeviceClass__init__
    DeviceClass.__str__ = __DeviceClass__str__
    DeviceClass.__repr__ = __DeviceClass__repr__
    DeviceClass._create_user_default_attr_prop = __DeviceClass__create_user_default_attr_prop
    DeviceClass._attribute_factory = __DeviceClass__attribute_factory
    DeviceClass._pipe_factory = __DeviceClass__pipe_factory
    DeviceClass._command_factory = __DeviceClass__command_factory
    DeviceClass._new_device = __DeviceClass__new_device

    DeviceClass.device_factory = __DeviceClass__device_factory
    DeviceClass.create_device = __DeviceClass__create_device
    DeviceClass.delete_device = __DeviceClass__delete_device
    DeviceClass.dyn_attr = __DeviceClass__dyn_attr
    DeviceClass.device_destroyer = __DeviceClass__device_destroyer
    DeviceClass.device_name_factory = __DeviceClass__device_name_factory


def __doc_DeviceClass():
    DeviceClass.__doc__ = """
    Base class for all TANGO device-class class.
    A TANGO device-class class is a class where is stored all
    data/method common to all devices of a TANGO device class
    """

    def document_method(method_name, desc, append=True):
        return __document_method(DeviceClass, method_name, desc, append)

    document_method("export_device", """
    export_device(self, dev, corba_dev_name = 'Unused') -> None

            For internal usage only

        Parameters :
            - dev : (DeviceImpl) device object
            - corba_dev_name : (str) CORBA device name. Default value is 'Unused'

        Return     : None
    """)

    document_method("register_signal", """
    register_signal(self, signo) -> None
    register_signal(self, signo, own_handler=false) -> None

            Register a signal.
            Register this class as class to be informed when signal signo
            is sent to to the device server process.
            The second version of the method is available only under Linux.

        Throws tango.DevFailed:
            - if the signal number is out of range
            - if the operating system failed to register a signal for the process.

        Parameters :
            - signo : (int) signal identifier
            - own_handler : (bool) true if you want the device signal handler
                            to be executed in its own handler instead of being
                            executed by the signal thread. If this parameter
                            is set to true, care should be taken on how the
                            handler is written. A default false value is provided

        Return     : None
    """)

    document_method("unregister_signal", """
    unregister_signal(self, signo) -> None

            Unregister a signal.
            Unregister this class as class to be informed when signal signo
            is sent to to the device server process

        Parameters :
            - signo : (int) signal identifier
        Return     : None
    """)

    document_method("signal_handler", """
    signal_handler(self, signo) -> None

            Signal handler.

            The method executed when the signal arrived in the device server process.
            This method is defined as virtual and then, can be redefined following
            device class needs.

        Parameters :
            - signo : (int) signal identifier
        Return     : None
    """)

    document_method("get_name", """
    get_name(self) -> str

            Get the TANGO device class name.

        Parameters : None
        Return     : (str) the TANGO device class name.
    """)

    document_method("get_type", """
    get_type(self) -> str

            Gets the TANGO device type name.

        Parameters : None
        Return     : (str) the TANGO device type name
    """)

    document_method("get_doc_url", """
    get_doc_url(self) -> str

            Get the TANGO device class documentation URL.

        Parameters : None
        Return     : (str) the TANGO device type name
    """)

    document_method("set_type", """
    set_type(self, dev_type) -> None

            Set the TANGO device type name.

        Parameters :
            - dev_type : (str) the new TANGO device type name
        Return     : None
    """)

    document_method("get_cvs_tag", """
    get_cvs_tag(self) -> str

            Gets the cvs tag

        Parameters : None
        Return     : (str) cvs tag
    """)

    document_method("get_cvs_location", """
    get_cvs_location(self) -> None

            Gets the cvs localtion

        Parameters : None
        Return     : (str) cvs location
    """)

    document_method("get_device_list", """
    get_device_list(self) -> sequence<tango.DeviceImpl>

            Gets the list of tango.DeviceImpl objects for this class

        Parameters : None
        Return     : (sequence<tango.DeviceImpl>) list of tango.DeviceImpl objects for this class
    """)

    document_method("get_command_list", """
    get_command_list(self) -> sequence<tango.Command>

            Gets the list of tango.Command objects for this class

        Parameters : None
        Return     : (sequence<tango.Command>) list of tango.Command objects for this class

        New in PyTango 8.0.0
    """)

    document_method("get_cmd_by_name", """
    get_cmd_by_name(self, (str)cmd_name) -> tango.Command

            Get a reference to a command object.

        Parameters :
            - cmd_name : (str) command name
        Return     : (tango.Command) tango.Command object

        New in PyTango 8.0.0
    """)

    document_method("add_wiz_dev_prop", """
    add_wiz_dev_prop(self, str, str) -> None
    add_wiz_dev_prop(self, str, str, str) -> None

            For internal usage only

        Parameters : None
        Return     : None
    """)

    document_method("add_wiz_class_prop", """
    add_wiz_class_prop(self, str, str) -> None
    add_wiz_class_prop(self, str, str, str) -> None

            For internal usage only

        Parameters : None
        Return     : None
    """)


def device_class_init(doc=True):
    __init_DeviceClass()
    if doc:
        __doc_DeviceClass()
