#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This class manage the TANGO database."""

import sys
import time
import logging
import functools

try:
    import argparse
except ImportError:
    argparse = None
    from optparse import OptionParser

import tango
from tango import AttrWriteType, GreenMode
from tango.server import Device
from tango.server import attribute, command
from tango.server import run

from tango.globals import get_class_by_class, get_constructed_class_by_class


READ_ONLY = AttrWriteType.READ
WRITE_ONLY = AttrWriteType.WRITE
READ_WRITE = AttrWriteType.READ_WRITE
READ_WITH_WRITE = AttrWriteType.READ_WITH_WRITE

# Argument Options
global options
global WILDCARD_REPLACEMENT
WILDCARD_REPLACEMENT = True


class DbInter(tango.Interceptors):

    def create_thread(self):
        pass

    def delete_thread(self):
        pass


DB_NAME = None


def set_db_name(db_name):
    global DB_NAME
    DB_NAME = db_name


def get_db_name():
    return DB_NAME


from . import db_access

th_exc = tango.Except.throw_exception

from .db_errors import *


def check_device_name(dev_name):
    if '*' in dev_name:
        return False, None, None

    if dev_name.startswith("tango:"):
        dev_name = dev_name[6:]
    elif dev_name.startswith("taco:"):
        dev_name = dev_name[5:]
    if dev_name.startswith("//"):
        dev_name = dev_name[2:]
        if '/' not in dev_name or dev_name.startswith("/"):
            return False, None, None
    dfm = dev_name.split("/")
    if len(dfm) != 3:
        return False, None, None
    # check that each element has at least one character
    if not all(map(len, dfm)):
        return False, None, None
    return True, dev_name, dfm


def replace_wildcard(text):
    if not WILDCARD_REPLACEMENT:
        return text
    # escape '%' with '\'
    text = text.replace("%", "\\%")
    # escape '_' with '\'
    text = text.replace("_", "\\_")
    # escape '"' with '\'
    text = text.replace('"', '\\"')
    # escape ''' with '\'
    text = text.replace("'", "\\'")
    # replace '*' with '%'
    text = text.replace("*", "%")
    return text


class TimeStructure:
    def __init__(self):
        self.average = 0
        self.minimum = 0
        self.maximum = 0
        self.maximum = 0
        self.total_elapsed = 0
        self.calls = 0
        self.index = ''


def stats(f):
    fname = f.__name__

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        try:
            return f(self, *args, **kwargs)
        finally:
            end = time.time()
            update_timing_stats(self, start, end, fname)
    return wrapper


def update_timing_stats(dev, time_before, time_after, cmd_name):
    tmp_time = dev.timing_maps[cmd_name]
    time_elapsed = (time_after - time_before) * 1000.
    tmp_time.total_elapsed = tmp_time.total_elapsed + time_elapsed
    if time_elapsed > tmp_time.maximum:
        tmp_time.maximum = time_elapsed
    if time_elapsed < tmp_time.minimum or tmp_time.minimum == 0:
        tmp_time.minimum = time_elapsed
    tmp_time.calls = tmp_time.calls + 1
    tmp_time.average = tmp_time.total_elapsed / tmp_time.calls


def get_plugin(name):
    fullname = '%s.%s' % (db_access.__package__, name)
    return __import__(fullname, None, None, fullname)


class DataBase(Device):
    """
    DataBase TANGO device class
    """

    # --- attributes ---------------------------------------

    Timing_maximum = attribute(dtype=('float64',), max_dim_x=128, access=READ_ONLY)

    Timing_average = attribute(dtype=('float64',), max_dim_x=128, access=READ_ONLY)

    Timing_index = attribute(dtype=('str',), max_dim_x=128, access=READ_ONLY)

    Timing_calls = attribute(dtype=('float64',), max_dim_x=128, access=READ_ONLY)

    Timing_info = attribute(dtype=('str',), max_dim_x=128, access=READ_ONLY)

    StoredProcedureRelease = attribute(dtype='str', access=READ_ONLY)

    Timing_minimum = attribute(dtype=('float64',), max_dim_x=128, access=READ_ONLY)

    def init_device(self):
        self._log = logging.getLogger(self.get_name())
        self._log.debug("In init_device()")
        self.attr_StoredProcedureRelease_read = ''
        self.init_timing_stats()
        m = get_plugin(options.db_access)
        self.db = m.get_db(personal_name = options.argv[1])
        try:
            global WILDCARD_REPLACEMENT
            WILDCARD_REPLACEMENT = m.get_wildcard_replacement()
        except AttributeError:
            pass
        self.set_state(tango.DevState.ON)

    def init_timing_stats(self):
        self.timing_maps = {}
        for cmd in dir(self):
            if cmd.startswith('Db'):
                self.timing_maps[cmd] = TimeStructure()
                self.timing_maps[cmd].index = cmd

    # --- attribute methods --------------------------------

    def read_Timing_maximum(self):
        self._log.debug("In read_Timing_maximum()")
        return [x.maximum for x in self.timing_maps.values()]

    def read_Timing_average(self):
        self._log.debug("In read_Timing_average()")

        return [x.average for x in self.timing_maps.values()]

    def read_Timing_index(self):
        self._log.debug("In read_Timing_index()")
        return [x.index for x in self.timing_maps.values()]

    def read_Timing_calls(self):
        self._log.debug("In read_Timing_calls()")
        return [x.calls for x in self.timing_maps.values()]

    def read_Timing_info(self):
        self._log.debug("In read_Timing_info()")
        util = tango.Util.instance()
        attr_Timing_info_read = []
        attr_Timing_info_read.append("TANGO Database Timing info on host " + util.get_host_name())
        attr_Timing_info_read.append(" ")
        attr_Timing_info_read.append("command	average	minimum	maximum	calls")
        attr_Timing_info_read.append(" ")
        for tmp_name in sorted(self.timing_maps.keys()):
            tmp_info = "%41s\t%6.3f\t%6.3f\t%6.3f\t%.0f"%(tmp_name, self.timing_maps[tmp_name].average, self.timing_maps[tmp_name].minimum, self.timing_maps[tmp_name].maximum, self.timing_maps[tmp_name].calls)
            attr_Timing_info_read.append(tmp_info)
        return attr_Timing_info_read

    def read_StoredProcedureRelease(self):
        self._log.debug("In read_StoredProcedureRelease()")
        self.attr_StoredProcedureRelease_read = self.db.get_stored_procedure_release()
        return self.attr_StoredProcedureRelease_read

    def read_Timing_minimum(self):
        self._log.debug("In read_Timing_minimum()")
        return [x.minimum for x in self.timing_maps.values()]

    # --- commands -----------------------------------------

    @stats
    @command(dtype_in='str', doc_in='The wildcard', dtype_out=('str',), doc_out='Device name domain list')
    def DbGetDeviceDomainList(self, argin):
        """ Get list of device domain name matching the specified

        :param argin: The wildcard
        :type: tango.DevString
        :return: Device name domain list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceDomainList()")
        return self.db.get_device_domain_list(replace_wildcard(argin))

    @stats
    @command(dtype_in='str', doc_in='Device server name (executable/instance)', doc_out='none')
    def DbUnExportServer(self, argin):
        """ Mark all devices belonging to a specified device server
        process as non exported

        :param argin: Device server name (executable/instance)
        :type: tango.DevString
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbUnExportServer()")
        self.db.unexport_server(argin)

    @command(dtype_in=('str',), doc_in='str[0] = device name\nStr[1]...str[n] = attribute name(s)', doc_out='none')
    def DbDeleteAllDeviceAttributeProperty(self, argin):
        """ Delete all attribute properties for the specified device attribute(s)

        :param argin: str[0] = device name
        Str[1]...str[n] = attribute name(s)
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteAllDeviceAttributeProperty()")

        if len(argin) < 2:
            self.warn_stream("DataBase::DbDeleteAllDeviceAttributeProperty(): insufficient number of arguments ")
            th_exc(DB_IncorrectArguments,
                   "insufficient number of arguments to delete all device attribute(s) property",
                   "DataBase::DbDeleteAllDeviceAttributeProperty()")

        dev_name = argin[0]

        ret, d_name, dfm = check_device_name(dev_name)

        if not ret:
            th_exc(DB_IncorrectDeviceName,
                  "device name (" + argin + ") syntax error (should be [tango:][//instance/]domain/family/member)",
                  "DataBase::DbDeleteAllDeviceAttributeProperty()")

        self.db.delete_all_device_attribute_property(dev_name, argin[1:])

    @command(dtype_in='str', doc_in='Attriibute alias name.', doc_out='none')
    def DbDeleteAttributeAlias(self, argin):
        """ Delete an attribute alias.

        :param argin: Attriibute alias name.
        :type: tango.DevString
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteAttributeAlias()")
        self.db.delete_attribute_alias(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class\nStr[1] = Attribute name\nStr[2] = Property name', dtype_out=('str',), doc_out='Str[0] = Attribute name\nStr[1] = Property name\nStr[2] = date\nStr[3] = Property value number (array case)\nStr[4] = Property value 1\nStr[n] = Property value n')
    def DbGetClassAttributePropertyHist(self, argin):
        """ Retrieve Tango class attribute property history

        :param argin: Str[0] = Tango class
        Str[1] = Attribute name
        Str[2] = Property name
        :type: tango.DevVarStringArray
        :return: Str[0] = Attribute name
        Str[1] = Property name
        Str[2] = date
        Str[3] = Property value number (array case)
        Str[4] = Property value 1
        Str[n] = Property value n
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetClassAttributePropertyHist()")
        class_name = argin[0]
        attribute = replace_wildcard(argin[1])
        prop_name = replace_wildcard(argin[2])
        return self.db.get_class_attribute_property_hist(class_name, attribute, prop_name)

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Attribute number\nStr[2] = Attribute name\nStr[3] = Property number\nStr[4] = Property name\nStr[5] = Property value number (array case)\nStr[5] = Property value 1\nStr[n] = Property value n (array case)\n.....', doc_out='none')
    def DbPutDeviceAttributeProperty2(self, argin):
        """ Put device attribute property. This command adds the possibility to have attribute property
        which are arrays. Not possible with the old DbPutDeviceAttributeProperty command.
        This old command is not deleted for compatibility reasons.

        :param argin: Str[0] = Device name
        Str[1] = Attribute number
        Str[2] = Attribute name
        Str[3] = Property number
        Str[4] = Property name
        Str[5] = Property value number (array case)
        Str[5] = Property value 1
        Str[n] = Property value n (array case)
        .....
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutDeviceAttributeProperty2()")
        device_name = argin[0]
        nb_attributes = int(argin[1])
        self.db.put_device_attribute_property2(device_name, nb_attributes, argin[2:])

    @command(dtype_in='str', doc_in='attribute alias filter string (eg: att*)', dtype_out=('str',), doc_out='attribute aliases')
    def DbGetAttributeAliasList(self, argin):
        """ Get attribute alias list for a specified filter

        :param argin: attribute alias filter string (eg: att*)
        :type: tango.DevString
        :return: attribute aliases
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetAttributeAliasList()")
        if not argin:
            argin = "%"
        else:
            argin = replace_wildcard(argin)
        return self.db.get_attribute_alias_list(argin)

    @command(dtype_in='str', doc_in='Class name', dtype_out=('str',), doc_out='Device exported list')
    def DbGetExportdDeviceListForClass(self, argin):
        """ Query the database for device exported for the specified class.

        :param argin: Class name
        :type: tango.DevString
        :return: Device exported list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetExportdDeviceListForClass()")
        argin = replace_wildcard(argin)
        return self.db.get_exported_device_list_for_class(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = attribute name\nStr[1] = attribute alias', doc_out='none')
    def DbPutAttributeAlias(self, argin):
        """ Define an alias for an attribute

        :param argin: Str[0] = attribute name
        Str[1] = attribute alias
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutAttributeAlias()")

        if len(argin) < 2:
            self.warn_stream("DataBase::DbPutAttributeAlias(): insufficient number of arguments ")
            th_exc(DB_IncorrectArguments,
                   "insufficient number of arguments to put attribute alias",
                   "DataBase::DbPutAttributeAlias()")

        attribute_name = argin[0]
        attribute_alias = argin[1]
        self.db.put_attribute_alias(attribute_name, attribute_alias)

    @stats
    @command(dtype_in='str', doc_in='The filter', dtype_out=('str',), doc_out='Device server process name list')
    def DbGetServerList(self, argin):
        """ Get list of device server process defined in database
        with name matching the specified filter

        :param argin: The filter
        :type: tango.DevString
        :return: Device server process name list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetServerList()")
        argin = replace_wildcard(argin)
        return self.db.get_server_list(argin)

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = CORBA IOR\nStr[2] = Device server process host name\nStr[3] = Device server process PID or string ``null``\nStr[4] = Device server process version', doc_out='none')
    def DbExportDevice(self, argin):
        """ Export a device to the database

        :param argin: Str[0] = Device name
        Str[1] = CORBA IOR
        Str[2] = Device server process host name
        Str[3] = Device server process PID or string ``null``
        Str[4] = Device server process version
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        DbExportDevice(self, argin)

    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Attribute name\nStr[2] = Property name\nStr[n] = Property name', doc_out='none')
    def DbDeleteDeviceAttributeProperty(self, argin):
        """ delete a device attribute property from the database

        :param argin: Str[0] = Device name
        Str[1] = Attribute name
        Str[2] = Property name
        Str[n] = Property name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteDeviceAttributeProperty()")

        if len(argin) < 3:
            self.warn_stream("DataBase::db_delete_device_attribute_property(): insufficient number of arguments ")
            th_exc(DB_IncorrectArguments,
                   "insufficient number of arguments to delete device attribute property",
                   "DataBase::DeleteDeviceAttributeProperty()")

        dev_name, attr_name = argin[:2]

        ret, dev_name, dfm = check_device_name(argin)
        if not ret:
            self.warn_stream("DataBase::db_delete_device_attribute_property(): device name " + argin + " incorrect ")
            th_exc(DB_IncorrectDeviceName,
                   "failed to delete device attribute property, device name incorrect",
                   "DataBase::DeleteDeviceAttributeProperty()")

        for prop_name in argin[2:]:
            self.db.delete_device_attribute_property(dev_name, attr_name, prop_name)

    @stats
    @command(dtype_in='str', doc_in='The wildcard', dtype_out=('str',), doc_out='Family list')
    def DbGetDeviceFamilyList(self, argin):
        """ Get a list of device name families for device name matching the
        specified wildcard

        :param argin: The wildcard
        :type: tango.DevString
        :return: Family list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceFamilyList()")
        argin = replace_wildcard(argin)
        return self.db.get_device_family_list(argin)

    @command(dtype_in='str', doc_in='filter', dtype_out=('str',), doc_out='list of exported devices')
    def DbGetDeviceWideList(self, argin):
        """ Get a list of devices whose names satisfy the filter.

        :param argin: filter
        :type: tango.DevString
        :return: list of exported devices
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceWideList()")
        argin = replace_wildcard(argin)
        return self.db.get_device_wide_list(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = Object name\nStr[1] = Property number\nStr[2] = Property name\nStr[3] = Property value number\nStr[4] = Property value 1\nStr[n] = Property value n\n....', doc_out='none')
    def DbPutProperty(self, argin):
        """ Create / Update free object property(ies)

        :param argin: Str[0] = Object name
        Str[1] = Property number
        Str[2] = Property name
        Str[3] = Property value number
        Str[4] = Property value 1
        Str[n] = Property value n
        ....
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutProperty()")
        object_name = argin[0]
        nb_properties = int(argin[1])
        self.db.put_property(object_name, properties, argin[2:])

    @command(dtype_in=('str',), doc_in='Str[0]  = Object name\nStr[1] = Property name\nStr[n] = Property name', doc_out='none')
    def DbDeleteProperty(self, argin):
        """ Delete free property from database

        :param argin: Str[0]  = Object name
        Str[1] = Property name
        Str[n] = Property name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteProperty()")
        obj_name = argin[0]
        for prop_name in argin[1:]:
            self.db.delete_property(obj_name, prop_name)

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class name\nStr[1] = Attribute name\nStr[n] = Attribute name', dtype_out=('str',), doc_out='Str[0] = Tango class name\nStr[1] = Attribute property  number\nStr[2] = Attribute property 1 name\nStr[3] = Attribute property 1 value number (array case)\nStr[4] = Attribute property 1 value\nStr[n] = Attribute property 1 value (array case)\nStr[n + 1] = Attribute property 2 name\nStr[n + 2] = Attribute property 2 value number (array case)\nStr[n + 3] = Attribute property 2 value\nStr[n + m] = Attribute property 2 value (array case)')
    def DbGetClassAttributeProperty2(self, argin):
        """ This command supports array property compared to the old command called
        DbGetClassAttributeProperty. The old command has not been deleted from the
        server for compatibility reasons.

        :param argin: Str[0] = Tango class name
        Str[1] = Attribute name
        Str[n] = Attribute name
        :type: tango.DevVarStringArray
        :return: Str[0] = Tango class name
        Str[1] = Attribute property  number
        Str[2] = Attribute property 1 name
        Str[3] = Attribute property 1 value number (array case)
        Str[4] = Attribute property 1 value
        Str[n] = Attribute property 1 value (array case)
        Str[n + 1] = Attribute property 2 name
        Str[n + 2] = Attribute property 2 value number (array case)
        Str[n + 3] = Attribute property 2 value
        Str[n + m] = Attribute property 2 value (array case)
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetClassAttributeProperty2()")
        class_name = argin[0]
        return self.db.get_class_attribute_property2(class_name, argin[1:])

    @stats
    @command(dtype_in='str', doc_in='filter', dtype_out=('str',), doc_out='list of exported devices')
    def DbGetDeviceExportedList(self, argin):
        """ Get a list of exported devices whose names satisfy the filter (wildcard is

        :param argin: filter
        :type: tango.DevString
        :return: list of exported devices
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceExportedList()")
        argin = replace_wildcard(argin)
        return self.db.get_device_exported_list(argin)

    @command(dtype_in='str', doc_in='The device name', dtype_out='str', doc_out='The alias found')
    def DbGetDeviceAlias(self, argin):
        """ Return alias for device name if found.

        :param argin: The device name
        :type: tango.DevString
        :return: The alias found
        :rtype: tango.DevString """
        self._log.debug("In DbGetDeviceAlias()")
        ret, dev_name, dfm = check_device_name(argin)
        if not ret:
            th_exc(DB_IncorrectDeviceName,
                  "device name (" + argin + ") syntax error (should be [tango:][//instance/]domain/family/member)",
                  "DataBase::DbGetDeviceAlias()")

        return self.db.get_device_alias(dev_name)

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class name\nStr[1] = Attribute number\nStr[2] = Attribute name\nStr[3] = Property number\nStr[4] = Property name\nStr[5] = Property value\n.....', doc_out='none')
    def DbPutClassAttributeProperty(self, argin):
        """ Create/Update class attribute property(ies) in database

        :param argin: Str[0] = Tango class name
        Str[1] = Attribute number
        Str[2] = Attribute name
        Str[3] = Property number
        Str[4] = Property name
        Str[5] = Property value
        .....
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutClassAttributeProperty()")
        class_name = argin[0]
        nb_attributes = int(argin[1])
        self.db.put_class_attribute_property(class_name, nb_attributes, argin[2:])

    @stats
    @command(dtype_in='str', doc_in='The filter', dtype_out=('str',), doc_out='Property name list')
    def DbGetClassPropertyList(self, argin):
        """ Get property list for a given Tango class with a specified filter

        :param argin: The filter
        :type: tango.DevString
        :return: Property name list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetClassPropertyList()")
        if not argin:
            argin = "%"
        else:
            argin = replace_wildcard(argin)
        return self.db.get_class_property_list(argin)

    @command(dtype_in='str', doc_in='The filter', dtype_out=('str',), doc_out='Device alias list')
    def DbGetDeviceAliasList(self, argin):
        """ Get device alias name with a specific filter

        :param argin: The filter
        :type: tango.DevString
        :return: Device alias list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceAliasList()")
        if not argin:
            argin = "%"
        else:
            argin = replace_wildcard(argin)

        return self.db.get_device_alias_list(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class name\nStr[1] = Attribute name', doc_out='none')
    def DbDeleteClassAttribute(self, argin):
        """ delete a class attribute and all its properties from database

        :param argin: Str[0] = Tango class name
        Str[1] = Attribute name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteClassAttribute()")

        if len(argin) < 2:
            self.warn_stream("DataBase::db_delete_class_attribute(): insufficient number of arguments ")
            th_exc(DB_IncorrectArguments,
                   "insufficient number of arguments to delete class attribute",
                   "DataBase::DeleteClassAttribute()")

        klass_name, attr_name = argin[:2]

        self.db.delete_class_attribute(klass_name, attr_name)

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class\nStr[1] = Property name', dtype_out=('str',), doc_out='Str[0] = Property name\nStr[1] = date\nStr[2] = Property value number (array case)\nStr[3] = Property value 1\nStr[n] = Property value n')
    def DbGetClassPropertyHist(self, argin):
        """ Retrieve Tango class property history

        :param argin: Str[0] = Tango class
        Str[1] = Property name
        :type: tango.DevVarStringArray
        :return: Str[0] = Property name
        Str[1] = date
        Str[2] = Property value number (array case)
        Str[3] = Property value 1
        Str[n] = Property value n
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetClassPropertyHist()")
        class_name = argin[0]
        prop_name = argin[1]
        return self.db.get_class_property_hist(class_name, prop_name)

    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Attribute name', doc_out='none')
    def DbDeleteDeviceAttribute(self, argin):
        """ Delete  device attribute properties from database

        :param argin: Str[0] = Device name
        Str[1] = Attribute name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteDeviceAttribute()")

        if len(argin) < 2:
            self.warn_stream("DataBase::db_delete_device_attribute(): insufficient number of arguments ")
            th_exc(DB_IncorrectArguments,
                   "insufficient number of arguments to delete device attribute",
                   "DataBase::DeleteDeviceAttribute()")

        dev_name, attr_name = argin[:2]

        ret, dev_name, dfm = check_device_name(argin)
        if not ret:
            self.warn_stream("DataBase::db_delete_device_attribute(): device name " + argin + " incorrect ")
            th_exc(DB_IncorrectDeviceName,
                   "failed to delete device attribute, device name incorrect",
                   "DataBase::DeleteDeviceAttribute()")

        self.db.delete_device_attribute(dev_name, attr_name)

    @stats
    @command(dtype_in='str', doc_in='MySql Select command', dtype_out='DevVarLongStringArray', doc_out='MySql Select command result\n - svalues : select results\n - lvalue[n] : =0 if svalue[n] is null else =1\n (last lvalue -1) is number of rows, (last lvalue) is number of fields')
    def DbMySqlSelect(self, argin):
        """ This is a very low level command.
        It executes the specified  SELECT command on TANGO database and returns its result without filter.

        :param argin: MySql Select command
        :type: tango.DevString
        :return: MySql Select command result
         - svalues : select results
         - lvalue[n] : =0 if svalue[n] is null else =1
         (last lvalue -1) is number of rows, (last lvalue) is number of fields
        :rtype: tango.DevVarLongStringArray """
        self._log.debug("In DbMySqlSelect()")
        tmp_argin = argin.lower()

        #  Check if SELECT key is alread inside command

        cmd = argin
        tmp_argin = argin.lower()
        pos = tmp_argin.find('select')
        if pos == -1:
            cmd = "SELECT " + cmd

        pos = tmp_argin.find(';')
        if pos != -1 and len(tmp_argin) > (pos + 1):
            th_exc(DB_IncorrectArguments,
                   "SQL command not valid: " + argin,
                   "DataBase::ExportDevice()")
        return self.db.my_sql_select(cmd)

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Attribute number\nStr[2] = Attribute name\nStr[3] = Property number\nStr[4] = Property name\nStr[5] = Property value\n.....', doc_out='none')
    def DbPutDeviceAttributeProperty(self, argin):
        """ Create/Update device attribute property(ies) in database

        :param argin: Str[0] = Device name
        Str[1] = Attribute number
        Str[2] = Attribute name
        Str[3] = Property number
        Str[4] = Property name
        Str[5] = Property value
        .....
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutDeviceAttributeProperty()")
        device_name = argin[0]
        nb_attributes = int(argin[1])
        self.db.put_device_attribute_property(device_name, nb_attributes, argin[2:])

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Attribute name\nStr[n] = Attribute name', dtype_out=('str',), doc_out='Str[0] = Device name\nStr[1] = Attribute property  number\nStr[2] = Attribute property 1 name\nStr[3] = Attribute property 1 value\nStr[n + 1] = Attribute property 2 name\nStr[n + 2] = Attribute property 2 value')
    def DbGetDeviceAttributeProperty(self, argin):
        """ Get device attribute property(ies) value

        :param argin: Str[0] = Device name
        Str[1] = Attribute name
        Str[n] = Attribute name
        :type: tango.DevVarStringArray
        :return: Str[0] = Device name
        Str[1] = Attribute property  number
        Str[2] = Attribute property 1 name
        Str[3] = Attribute property 1 value
        Str[n + 1] = Attribute property 2 name
        Str[n + 2] = Attribute property 2 value
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceAttributeProperty()")
        dev_name = argin[0]
        return self.db.get_device_attribute_property(dev_name, argin[1:])

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Object name\nStr[1] = filter', dtype_out=('str',), doc_out='Property name list')
    def DbGetPropertyList(self, argin):
        """ Get list of property defined for a free object and matching the
        specified filter

        :param argin: Str[0] = Object name
        Str[1] = filter
        :type: tango.DevVarStringArray
        :return: Property name list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetPropertyList()")
        object_name = argin[0]
        wildcard = replace_wildcard(argin[1])
        return self.db.get_property_list(object_name, wildcard)

    @stats
    @command(dtype_in='str', doc_in='Device server process name', dtype_out=('str',), doc_out='Str[0] = Device name\nStr[1] = Tango class\nStr[n] = Device name\nStr[n + 1] = Tango class')
    def DbGetDeviceClassList(self, argin):
        """ Get Tango classes/device list embedded in a specific device server

        :param argin: Device server process name
        :type: tango.DevString
        :return: Str[0] = Device name
        Str[1] = Tango class
        Str[n] = Device name
        Str[n + 1] = Tango class
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceClassList()")
        return self.db.get_device_class_list(argin)

    @command(dtype_in='str', doc_in='Device name', doc_out='none')
    def DbUnExportDevice(self, argin):
        """ Mark a device as non exported in database

        :param argin: Device name
        :type: tango.DevString
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbUnExportDevice()")
        dev_name = argin[0].lower()
        self.db.unexport_device(dev_name)

    @command(dtype_in='str', doc_in='Alias name', dtype_out='str', doc_out='Device name')
    def DbGetAliasDevice(self, argin):
        """ Get device name from its alias.

        :param argin: Alias name
        :type: tango.DevString
        :return: Device name
        :rtype: tango.DevString """
        self._log.debug("In DbGetAliasDevice()")
        if not argin:
            argin = "%"
        else:
            argin = replace_wildcard(argin)
        return self.db.get_alias_device(argin)

    @command(dtype_in='str', doc_in='device name', doc_out='none')
    def DbDeleteDevice(self, argin):
        """ Delete a devcie from database

        :param argin: device name
        :type: tango.DevString
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteDevice()")

        ret, dev_name, dfm = check_device_name(argin)
        if not ret:
            self.warn_stream("DataBase::db_delete_device(): device name " + argin + " incorrect ")
            th_exc(DB_IncorrectDeviceName,
                   "failed to delete device, device name incorrect",
                   "DataBase::DeleteDevice()")
        self.db.delete_device(dev_name)

    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Wildcard', dtype_out=('str',), doc_out='attribute name list')
    def DbGetDeviceAttributeList(self, argin):
        """ Return list of attributes matching the wildcard
         for the specified device

        :param argin: Str[0] = Device name
        Str[1] = Wildcard
        :type: tango.DevVarStringArray
        :return: attribute name list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceAttributeList()")
        dev_name = argin[0]
        wildcard = argin[1]
        if not wildcard:
            wildcard = "%"
        else:
            wildcard = replace_wildcard(wildcard)
        return self.db.get_device_attribute_list(dev_name, wildcard)

    @command(dtype_in='str', doc_in='Host name', dtype_out=('str',), doc_out='Server info for all servers running on specified host')
    def DbGetHostServersInfo(self, argin):
        """ Get info about all servers running on specified host, name, mode and level

        :param argin: Host name
        :type: tango.DevString
        :return: Server info for all servers running on specified host
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetHostServersInfo()")
        argin = replace_wildcard(argin)
        return self.db.get_host_servers_info(argin)

    @command(dtype_in=('str',), doc_in='s[0] = old device server name (exec/instance)\ns[1] = new device server name (exec/instance)')
    def DbRenameServer(self, argin):
        """ Rename a device server process

        :param argin: str[0] = old device server name (exec/instance)
        str[1] =  new device server name (exec/instance)
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbRenameServer()")

        if len(argin) < 2:
            self.warn_stream("DataBase::DbRenameServer(): insufficient number of arguments ")
            th_exc(DB_IncorrectArguments,
                   "insufficient number of arguments (two required: old name and new name",
                   "DataBase::DbRenameServer")

        old_name = argin[0]
        new_name = argin[1]

        if ('/' not in argin[0]) or ('/' not in argin[1]):
            self.warn_stream("DataBase::DbRenameServer(): wrong syntax in command args ")
            th_exc(DB_IncorrectArguments,
                   "Wrong syntax in command args (ds_exec_name/inst_name)",
                   "DataBase::DbRenameServer")

        self.db.rename_server(old_name, new_name)

    @stats
    @command(dtype_in='str', doc_in='The filter', dtype_out=('str',), doc_out='Host name list')
    def DbGetHostList(self, argin):
        """ Get host list with name matching the specified filter

        :param argin: The filter
        :type: tango.DevString
        :return: Host name list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetHostList()")
        argin = replace_wildcard(argin)
        return self.db.get_host_list(argin)

    @command(dtype_in='str', doc_in='Device name', dtype_out=('str',), doc_out='Classes off the specified device.\n[0] - is the class of the device.\n[1] - is the class from the device class is inherited.\n........and so on')
    def DbGetClassInheritanceForDevice(self, argin):
        """ Get class inheritance for the specified device.

        :param argin: Device name
        :type: tango.DevString
        :return: Classes off the specified device.
        [0] - is the class of the device.
        [1] - is the class from the device class is inherited.
        ........and so on
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetClassInheritanceForDevice()")
        return self.db.get_class_inheritance_for_device(argin)

    @stats
    @command(dtype_in='str', doc_in='Device server name', doc_out='none')
    def DbDeleteServer(self, argin):
        """ Delete server from the database but dont delete device properties

        :param argin: Device server name
        :type: tango.DevString
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteServer()")

        if '*' in argin or '%' in argin or not '/' in argin:
            self.warn_stream("DataBase::db_delete_server(): server name " + argin + " incorrect ")
            th_exc(DB_IncorrectServerName,
                   "failed to delete server, server name incorrect",
                   "DataBase::DeleteServer()")

        self.db.delete_server(argin)

    @command(dtype_in='str', doc_in='The attribute alias name', dtype_out='str', doc_out='The attribute name (device/attribute)')
    def DbGetAttributeAlias(self, argin):
        """ Get the attribute name for the given alias.
        If alias not found in database, returns an empty string.

        :param argin: The attribute alias name
        :type: tango.DevString
        :return: The attribute name (device/attribute)
        :rtype: tango.DevString """
        self._log.debug("In DbGetAttributeAlias()")
        return self.db.get_attribute_alias(argin)

    @command(dtype_in=('str',), doc_in='Elt[0] = DS name (exec_name/inst_name), Elt[1] = Host name', dtype_out=('str',), doc_out='All the data needed by the device server during its startup sequence. Precise list depend on the device server')
    def DbGetDataForServerCache(self, argin):
        """ This command returns all the data needed by a device server process during its
        startup sequence. The aim of this command is to minimize database access during
        device server startup sequence.

        :param argin: Elt[0] = DS name (exec_name/inst_name), Elt[1] = Host name
        :type: tango.DevVarStringArray
        :return: All the data needed by the device server during its startup sequence. Precise list depend on the device server
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDataForServerCache()")
        ##  TODO
        return ['']

    @command(dtype_in=('str',), doc_in='Str[0] = Object name\nStr[1] = Property name\nStr[n] = Property name', dtype_out=('str',), doc_out='Str[0] = Object name\nStr[1] = Property number\nStr[2] = Property name\nStr[3] = Property value number (array case)\nStr[4] = Property value 1\nStr[n] = Property value n (array case)\nStr[n + 1] = Property name\nStr[n + 2] = Property value number (array case)\nStr[n + 3] = Property value 1\nStr[n + m] = Property value m')
    def DbGetProperty(self, argin):
        """ Get free object property

        :param argin: Str[0] = Object name
        Str[1] = Property name
        Str[n] = Property name
        :type: tango.DevVarStringArray
        :return: Str[0] = Object name
        Str[1] = Property number
        Str[2] = Property name
        Str[3] = Property value number (array case)
        Str[4] = Property value 1
        Str[n] = Property value n (array case)
        Str[n + 1] = Property name
        Str[n + 2] = Property value number (array case)
        Str[n + 3] = Property value 1
        Str[n + m] = Property value m
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetProperty()")
        object_name = argin[0]
        return self.db.get_property(object_name, argin[1:])

    @command(dtype_in='str', doc_in='device server process name', dtype_out=('str',), doc_out='list of classes for this device server')
    def DbGetDeviceServerClassList(self, argin):
        """ Get list of Tango classes for a device server

        :param argin: device server process name
        :type: tango.DevString
        :return: list of classes for this device server
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceServerClassList()")
        argin = replace_wildcard(argin)
        return self.db.get_server_class_list(argin)

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Tango device name\nStr[1] = Property number\nStr[2] = Property name\nStr[3] = Property value number\nStr[4] = Property value 1\nStr[n] = Property value n\n....', doc_out='none')
    def DbPutDeviceProperty(self, argin):
        """ Create / Update device property(ies)

        :param argin: Str[0] = Tango device name
        Str[1] = Property number
        Str[2] = Property name
        Str[3] = Property value number
        Str[4] = Property value 1
        Str[n] = Property value n
        ....
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutDeviceProperty()")
        device_name = argin[0]
        nb_properties = int(argin[1])
        self.db.put_device_property(device_name, nb_properties, argin[2:])

    @command(doc_in='none', doc_out='none')
    def ResetTimingValues(self):
        """ Reset the timing attribute values.

        :param :
        :type: tango.DevVoid
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In ResetTimingValues()")
        for tmp_timing in self.timing_maps.itervalues():
            tmp_timing.average = 0.
            tmp_timing.minimum = 0.
            tmp_timing.maximum = 0.
            tmp_timing.total_elapsed = 0.
            tmp_timing.calls = 0.

    @command(doc_in='none', dtype_out=('str',), doc_out='List of host:port with one element for each database server')
    def DbGetCSDbServerList(self):
        """ Get a list of host:port for all database server defined in the control system

        :param :
        :type: tango.DevVoid
        :return: List of host:port with one element for each database server
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetCSDbServerList()")
        return self.db.get_csdb_server_list()

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Tango class name\nStr[1] = Property number\nStr[2] = Property name\nStr[3] = Property value number\nStr[4] = Property value 1\nStr[n] = Property value n\n....', doc_out='none')
    def DbPutClassProperty(self, argin):
        """ Create / Update class property(ies)

        :param argin: Str[0] = Tango class name
        Str[1] = Property number
        Str[2] = Property name
        Str[3] = Property value number
        Str[4] = Property value 1
        Str[n] = Property value n
        ....
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutClassProperty()")
        class_name = argin[0]
        nb_properties = int(argin[1])
        self.db.put_class_property(class_name, nb_properties, argin[2:])

    @stats
    @command(dtype_in='str', doc_in='Device name (or alias)', dtype_out='DevVarLongStringArray', doc_out='Str[0] = device name\nStr[1] = CORBA IOR\nStr[2] = device version\nStr[3] = device server process name\nStr[4] = host name\nStr[5] = Tango class name\n\nLg[0] = Exported flag\nLg[1] = Device server process PID')
    def DbImportDevice(self, argin):
        """ Import a device from the database

        :param argin: Device name (or alias)
        :type: tango.DevString
        :return: Str[0] = device name
        Str[1] = CORBA IOR
        Str[2] = device version
        Str[3] = device server process name
        Str[4] = host name
        Str[5] = Tango class name

        Lg[0] = Exported flag
        Lg[1] = Device server process PID
        :rtype: tango.DevVarLongStringArray """
        self._log.debug("In DbImportDevice()")
        return self.db.import_device(argin.lower())

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Property name\nStr[n] = Property name', doc_out='none')
    def DbDeleteDeviceProperty(self, argin):
        """ Delete device property(ies)

        :param argin: Str[0] = Device name
        Str[1] = Property name
        Str[n] = Property name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteDeviceProperty()")
        dev_name = argin[0]
        for prop_name in argin[1:]:
            self.db.delete_device_property(dev_name, prop_name)

    @command(dtype_in='str', doc_in='Device name', dtype_out='str', doc_out='Device Tango class')
    def DbGetClassForDevice(self, argin):
        """ Get Tango class for the specified device.

        :param argin: Device name
        :type: tango.DevString
        :return: Device Tango class
        :rtype: tango.DevString """
        self._log.debug("In DbGetClassForDevice()")
        return self.db.get_class_for_device(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Attribute name\nStr[2] = Property name', dtype_out=('str',), doc_out='Str[0] = Attribute name\nStr[1] = Property name\nStr[2] = date\nStr[3] = Property value number (array case)\nStr[4] = Property value 1\nStr[n] = Property value n')
    def DbGetDeviceAttributePropertyHist(self, argin):
        """ Retrieve device attribute property history

        :param argin: Str[0] = Device name
        Str[1] = Attribute name
        Str[2] = Property name
        :type: tango.DevVarStringArray
        :return: Str[0] = Attribute name
        Str[1] = Property name
        Str[2] = date
        Str[3] = Property value number (array case)
        Str[4] = Property value 1
        Str[n] = Property value n
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceAttributePropertyHist()")
        dev_name = argin[0]
        attribute = replace_wildcard(argin[1])
        prop_name = replace_wildcard(argin[2])
        return self.db.get_device_attribute_property_hist(dev_name, attribute, prop_name)

    @command(dtype_in='str', doc_in='server name', dtype_out=('str',), doc_out='server info')
    def DbGetServerInfo(self, argin):
        """ Get info about host, mode and level for specified server

        :param argin: server name
        :type: tango.DevString
        :return: server info
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetServerInfo()")
        return self.db.get_server_info(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = device name\nStr[1] = alias name', doc_out='none')
    def DbPutDeviceAlias(self, argin):
        """ Define alias for  a given device name

        :param argin: Str[0] = device name
        Str[1] = alias name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutDeviceAlias()")

        if len(argin) < 2:
            self.warn_stream("DataBase::DbPutDeviceAlias(): insufficient number of arguments ")
            th_exc(DB_IncorrectArguments,
                   "insufficient number of arguments to put device alias",
                   "DataBase::DbPutDeviceAlias()")

        device_name = argin[0]
        device_alias = argin[1]
        self.db.put_device_alias(device_name, device_alias)

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = device name\nStr[1] = Filter', dtype_out=('str',), doc_out='Property name list')
    def DbGetDevicePropertyList(self, argin):
        """ Get property list belonging to the specified device and with
        name matching the specified filter

        :param argin: Str[0] = device name
        Str[1] = Filter
        :type: tango.DevVarStringArray
        :return: Property name list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDevicePropertyList()")
        device_name = argin[0]
        prop_filter = argin[1]
        prop_filter = replace_wildcard(prop_filter)
        return self.db.get_device_property_list(device_name, prop_filter)

    @stats
    @command(dtype_in='str', doc_in='The filter', dtype_out=('str',), doc_out='Device server process name list')
    def DbGetHostServerList(self, argin):
        """ Get list of device server process name running on host with name matching
        the specified filter

        :param argin: The filter
        :type: tango.DevString
        :return: Device server process name list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetHostServerList()")
        argin = replace_wildcard(argin)
        return self.db.get_host_server_list(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class\nStr[1] = Property name\nStr[2] = Property name', dtype_out=('str',), doc_out='Str[0] = Tango class\nStr[1] = Property number\nStr[2] = Property name\nStr[3] = Property value number (array case)\nStr[4] = Property value\nStr[n] = Propery value (array case)\n....')
    def DbGetClassProperty(self, argin):
        """

        :param argin: Str[0] = Tango class
        Str[1] = Property name
        Str[2] = Property name
        :type: tango.DevVarStringArray
        :return: Str[0] = Tango class
        Str[1] = Property number
        Str[2] = Property name
        Str[3] = Property value number (array case)
        Str[4] = Property value
        Str[n] = Propery value (array case)
        ....
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetClassProperty()")
        class_name = argin[0]
        return self.db.get_class_property(class_name,argin[1:])

    @command(dtype_in='str', doc_in='The filter', dtype_out=('str',), doc_out='Object name list')
    def DbGetObjectList(self, argin):
        """ Get list of free object defined in database with name
        matching the specified filter

        :param argin: The filter
        :type: tango.DevString
        :return: Object name list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetObjectList()")
        argin = replace_wildcard(argin)
        return self.db.get_object_list(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class name\nStr[1] = Attribute name\nStr[2] = Property name\nStr[n] = Property name', doc_out='none')
    def DbDeleteClassAttributeProperty(self, argin):
        """ delete class attribute properties from database

        :param argin: Str[0] = Tango class name
        Str[1] = Attribute name
        Str[2] = Property name
        Str[n] = Property name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteClassAttributeProperty()")

        if len(argin) < 3:
            self.warn_stream("DataBase::db_delete_class_attribute_property(): insufficient number of arguments ")
            th_exc(DB_IncorrectArguments,
                   "insufficient number of arguments to delete class attribute property",
                   "DataBase::DeleteClassAttributeProperty()")

        klass_name, attr_name = argin[:2]

        for prop_name in argin[2:]:
            self.db.delete_class_attribute_property(klass_name, attr_name, prop_name)

    @command(dtype_in='str', doc_in='Server name', dtype_out=('str',), doc_out='The instance names found for specified server.')
    def DbGetInstanceNameList(self, argin):
        """ Returns the instance names found for specified server.

        :param argin: Server name
        :type: tango.DevString
        :return: The instance names found for specified server.
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetInstanceNameList()")
        return self.db.get_instance_name_list(argin)

    @command(dtype_in='str', doc_in='The attribute name (dev_name/att_name)', dtype_out='str', doc_out='The attribute alias name (or empty string)')
    def DbGetAttributeAlias2(self, argin):
        """ Get the attribute alias from the attribute name.
        Returns one empty string if nothing found in database

        :param argin: The attribute name (dev_name/att_name)
        :type: tango.DevString
        :return: The attribute alias name (or empty string)
        :rtype: tango.DevString """
        self._log.debug("In DbGetAttributeAlias2()")
        attr_name = argin[0]
        return self.db.get_attribute_alias2(attr_name)

    @command(dtype_in=('str',), doc_in='Str[0] = Full device server name\nStr[1] = Device(s) name\nStr[2] = Tango class name\nStr[n] = Device name\nStr[n + 1] = Tango class name', doc_out='none')
    def DbAddServer(self, argin):
        """ Create a device server process entry in database

        :param argin: Str[0] = Full device server name
        Str[1] = Device(s) name
        Str[2] = Tango class name
        Str[n] = Device name
        Str[n + 1] = Tango class name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbAddServer()")

        if len(argin) < 3 or not len(argin) % 2:
            self.warn_stream("DataBase::AddServer(): incorrect number of input arguments ")
            th_exc(DB_IncorrectArguments,
                   "incorrect no. of input arguments, needs at least 3 (server,device,class)",
                   "DataBase::AddServer()")
        server_name = argin[0]

        for i in range((len(argin) - 1) / 2):
            d_name, klass_name = argin[i * 2 + 1], argin[i * 2 + 2]
            ret, dev_name, dfm = check_device_name(d_name)
            if not ret:
                th_exc(DB_IncorrectDeviceName,
                      "device name (" + d_name + ") syntax error (should be [tango:][//instance/]domain/family/member)",
                      "DataBase::AddServer()")
            self.db.add_device(server_name, (dev_name, dfm) , klass_name)

    @stats
    @command(dtype_in='str', doc_in='name of event channel or factory', dtype_out='DevVarLongStringArray', doc_out='export information e.g. IOR')
    def DbImportEvent(self, argin):
        """ Get event channel info from database

        :param argin: name of event channel or factory
        :type: tango.DevString
        :return: export information e.g. IOR
        :rtype: tango.DevVarLongStringArray """
        self._log.debug("In DbImportEvent()")
        argin = replace_wildcard(argin.lower())
        return self.db.import_event(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[2] = Property name', dtype_out=('str',), doc_out='Str[0] = Property name\nStr[1] = date\nStr[2] = Property value number (array case)\nStr[3] = Property value 1\nStr[n] = Property value n')
    def DbGetDevicePropertyHist(self, argin):
        """ Retrieve device  property history

        :param argin: Str[0] = Device name
        Str[1] = Property name
        :type: tango.DevVarStringArray
        :return: Str[0] = Property name
        Str[1] = date
        Str[2] = Property value number (array case)
        Str[3] = Property value 1
        Str[n] = Property value n
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDevicePropertyHist()")
        device_name = argin[0]
        prop_name = argin[1]
        return self.db.get_device_property_hist(device_name, prop_name)

    @command(dtype_in='str', doc_in='wildcard for server names.', dtype_out=('str',), doc_out='server names found.')
    def DbGetServerNameList(self, argin):
        """ Returns the list of server names found for the wildcard specified.
        It returns only the server executable name without instance name as DbGetServerList.

        :param argin: wildcard for server names.
        :type: tango.DevString
        :return: server names found.
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetServerNameList()")
        argin = replace_wildcard(argin)
        return self.db.get_server_name_list(argin)

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Attribute name\nStr[n] = Attribute name', dtype_out=('str',), doc_out='Str[0] = Device name\nStr[1] = Attribute property  number\nStr[2] = Attribute property 1 name\nStr[3] = Attribute property 1 value number (array case)\nStr[4] = Attribute property 1 value\nStr[n] = Attribute property 1 value (array case)\nStr[n + 1] = Attribute property 2 name\nStr[n + 2] = Attribute property 2 value number (array case)\nStr[n + 3] = Attribute property 2 value\nStr[n + m] = Attribute property 2 value (array case)')
    def DbGetDeviceAttributeProperty2(self, argin):
        """ Retrieve device attribute properties. This command has the possibility to retrieve
        device attribute properties which are arrays. It is not possible with the old
        DbGetDeviceAttributeProperty command. Nevertheless, the old command has not been
        deleted for compatibility reason

        :param argin: Str[0] = Device name
        Str[1] = Attribute name
        Str[n] = Attribute name
        :type: tango.DevVarStringArray
        :return: Str[0] = Device name
        Str[1] = Attribute property  number
        Str[2] = Attribute property 1 name
        Str[3] = Attribute property 1 value number (array case)
        Str[4] = Attribute property 1 value
        Str[n] = Attribute property 1 value (array case)
        Str[n + 1] = Attribute property 2 name
        Str[n + 2] = Attribute property 2 value number (array case)
        Str[n + 3] = Attribute property 2 value
        Str[n + m] = Attribute property 2 value (array case)
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceAttributeProperty2()")
        dev_name = argin[0]
        return self.db.get_device_attribute_property2(dev_name, argin[1:])

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class name\nStr[1] = Property name\nStr[n] = Property name', doc_out='none')
    def DbDeleteClassProperty(self, argin):
        """ Delete class properties from database

        :param argin: Str[0] = Tango class name
        Str[1] = Property name
        Str[n] = Property name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteClassProperty()")
        klass_name = argin[0]
        for prop_name in argin[1:]:
            self.db.delete_class_property(prop_name)

    @command(dtype_in='str', doc_in='name of event channel or factory to unexport', doc_out='none')
    def DbUnExportEvent(self, argin):
        """ Mark one event channel as non exported in database

        :param argin: name of event channel or factory to unexport
        :type: tango.DevString
        :return: none
        :rtype: tango.DevVoid """
        self._log.debug("In DbUnExportEvent()")
        event_name = argin[0].lower()
        self.db.unexport_event(event_name)

    @stats
    @command(doc_in='none', dtype_out=('str',), doc_out='Miscellaneous info like:\n- Device defined in database\n- Device marked as exported in database\n- Device server process defined in database\n- Device server process marked as exported in database\n- Device properties defined in database\n- Class properties defined in database\n- Device attribute properties defined in database\n- Class attribute properties defined in database\n- Object properties defined in database')
    def DbInfo(self):
        """ Get miscellaneous numbers on information
        stored in database

        :param :
        :type: tango.DevVoid
        :return: Miscellaneous info like:
        - Device defined in database
        - Device marked as exported in database
        - Device server process defined in database
        - Device server process marked as exported in database
        - Device properties defined in database
        - Class properties defined in database
        - Device attribute properties defined in database
        - Class attribute properties defined in database
        - Object properties defined in database
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbInfo()")
        return self.db.info()

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class name\nStr[1] = Attribute name\nStr[n] = Attribute name', dtype_out=('str',), doc_out='Str[0] = Tango class name\nStr[1] = Attribute property  number\nStr[2] = Attribute property 1 name\nStr[3] = Attribute property 1 value\nStr[n + 1] = Attribute property 2 name\nStr[n + 2] = Attribute property 2 value')
    def DbGetClassAttributeProperty(self, argin):
        """ Get Tango class property(ies) value

        :param argin: Str[0] = Tango class name
        Str[1] = Attribute name
        Str[n] = Attribute name
        :type: tango.DevVarStringArray
        :return: Str[0] = Tango class name
        Str[1] = Attribute property  number
        Str[2] = Attribute property 1 name
        Str[3] = Attribute property 1 value
        Str[n + 1] = Attribute property 2 name
        Str[n + 2] = Attribute property 2 value
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetClassAttributeProperty()")
        class_name = argin[0]
        return self.db.get_class_attribute_property(class_name, argin[1:])

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class name\nStr[1] = Attribute number\nStr[2] = Attribute name\nStr[3] = Property number\nStr[4] = Property name\nStr[5] = Property value number (array case)\nStr[5] = Property value 1\nStr[n] = Property value n (array case)\n.....', doc_out='none')
    def DbPutClassAttributeProperty2(self, argin):
        """ This command adds support for array properties compared to the previous one
        called DbPutClassAttributeProperty. The old comman is still there for compatibility reason

        :param argin: Str[0] = Tango class name
        Str[1] = Attribute number
        Str[2] = Attribute name
        Str[3] = Property number
        Str[4] = Property name
        Str[5] = Property value number (array case)
        Str[5] = Property value 1
        Str[n] = Property value n (array case)
        .....
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutClassAttributeProperty2()")
        class_name = argin[0]
        nb_attributes = int(argin[1])
        self.db.put_class_attribute_property2(class_name, nb_attributes, argin[2:])

    @command(dtype_in=('str',), doc_in='server info', doc_out='none')
    def DbPutServerInfo(self, argin):
        """ Update server info including host, mode and level

        :param argin: server info
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbPutServerInfo()")

        if len(argin) < 4:
            self.warn_stream("DataBase::DbPutServerInfo(): insufficient number of arguments ")
            th_exc(DB_IncorrectArguments,
                   "insufficient server info",
                   "DataBase::DbPutServerInfo()")

        tmp_server = argin[0].lower()
        tmp_host = argin[1]
        tmp_mode = argin[2]
        tmp_level = argin[3]
        tmp_extra = []
        if len(argin) > 4:
            tmp_extra = argin[4:]

        tmp_len = len(argin) - 1
        self.db.put_server_info(tmp_server, tmp_host, tmp_mode, tmp_level, tmp_extra)

    @command(dtype_in='str', doc_in='device alias name', doc_out='none')
    def DbDeleteDeviceAlias(self, argin):
        """ Delete a device alias.

        :param argin: device alias name
        :type: tango.DevString
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteDeviceAlias()")
        self.db.delete_device_alias(argin)

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = event channel name (or factory name)\nStr[1] = CORBA IOR\nStr[2] = Notifd host name\nStr[3] = Notifd pid\nStr[4] = Notifd version', doc_out='none')
    def DbExportEvent(self, argin):
        """ Export Event channel to database

        :param argin: Str[0] = event channel name (or factory name)
        Str[1] = CORBA IOR
        Str[2] = Notifd host name
        Str[3] = Notifd pid
        Str[4] = Notifd version
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbExportEvent()")

        if len(argin) < 5:
            self.warn_stream("DataBase::db_export_event(): insufficient export info for event ")
            th_exc(DB_IncorrectArguments,
                   "insufficient export info for event",
                   "DataBase::ExportEvent()")

        event, IOR, host, pid, version = argin[:5]
        event = replace_wildcard(event.lower())
        self.db.export_event(event, IOR, host, pid, version)

    @stats
    @command(dtype_in=('str',), doc_in='Str[0] = Device name\nStr[1] = Property name\nStr[n] = Property name', dtype_out=('str',), doc_out='Str[0] = Device name\nStr[1] = Property number\nStr[2] = Property name\nStr[3] = Property value number (array case)\nStr[4] = Property value 1\nStr[n] = Property value n (array case)\nStr[n + 1] = Property name\nStr[n + 2] = Property value number (array case)\nStr[n + 3] = Property value 1\nStr[n + m] = Property value m')
    def DbGetDeviceProperty(self, argin):
        """

        :param argin: Str[0] = Device name
        Str[1] = Property name
        Str[n] = Property name
        :type: tango.DevVarStringArray
        :return: Str[0] = Device name
        Str[1] = Property number
        Str[2] = Property name
        Str[3] = Property value number (array case)
        Str[4] = Property value 1
        Str[n] = Property value n (array case)
        Str[n + 1] = Property name
        Str[n + 2] = Property value number (array case)
        Str[n + 3] = Property value 1
        Str[n + m] = Property value m
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceProperty()")
        device_name = argin[0]
        return self.db.get_device_property(device_name, argin[1:])

    @command(dtype_in='str', doc_in='Device name', dtype_out='DevVarLongStringArray', doc_out='Str[0] = Device name\nStr[1] = CORBA IOR\nStr[2] = Device version\nStr[3] = Device Server name\nStr[4] = Device Server process host name\nStr[5] = Started date (or ? if not set)\nStr[6] = Stopped date (or ? if not set)\nStr[7] = Device class\n\nLg[0] = Device exported flag\nLg[1] = Device Server process PID (or -1 if not set)')
    def DbGetDeviceInfo(self, argin):
        """ Returns info from DbImportDevice and started/stopped dates.

        :param argin: Device name
        :type: tango.DevString
        :return: Str[0] = Device name
        Str[1] = CORBA IOR
        Str[2] = Device version
        Str[3] = Device Server name
        Str[4] = Device Server process host name
        Str[5] = Started date (or ? if not set)
        Str[6] = Stopped date (or ? if not set)
        Str[7] = Device class

        Lg[0] = Device exported flag
        Lg[1] = Device Server process PID (or -1 if not set)
        :rtype: tango.DevVarLongStringArray """
        self._log.debug("In DbGetDeviceInfo()")
        ret, dev_name, dfm = check_device_name(argin)
        if not ret:
            th_exc(DB_IncorrectDeviceName,
                  "device name (" + argin + ") syntax error (should be [tango:][//instance/]domain/family/member)",
                  "DataBase::DbGetDeviceAlias()")

        return self.db.get_device_info(dev_name)

    @command(dtype_in=('str',), doc_in='Str[0] = Object name\nStr[2] = Property name', dtype_out=('str',), doc_out='Str[0] = Property name\nStr[1] = date\nStr[2] = Property value number (array case)\nStr[3] = Property value 1\nStr[n] = Property value n')
    def DbGetPropertyHist(self, argin):
        """ Retrieve object  property history

        :param argin: Str[0] = Object name
        Str[2] = Property name
        :type: tango.DevVarStringArray
        :return: Str[0] = Property name
        Str[1] = date
        Str[2] = Property value number (array case)
        Str[3] = Property value 1
        Str[n] = Property value n
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetPropertyHist()")
        object_name = argin[0]
        prop_name = argin[1]
        return self.db.get_property_hist(object_name, prop_name)

    @stats
    @command(dtype_in='str', doc_in='The filter', dtype_out=('str',), doc_out='Device names member list')
    def DbGetDeviceMemberList(self, argin):
        """ Get a list of device name members for device name matching the
        specified filter

        :param argin: The filter
        :type: tango.DevString
        :return: Device names member list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceMemberList()")
        argin = replace_wildcard(argin)
        return self.db.get_device_member_list(argin)

    @command(dtype_in='str', doc_in='Filter', dtype_out=('str',), doc_out='Class list')
    def DbGetClassList(self, argin):
        """ Get Tango class list with a specified filter

        :param argin: Filter
        :type: tango.DevString
        :return: Class list
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetClassList()")
        server = replace_wildcard(argin)
        return self.db.get_class_list(server)

    @command(dtype_in='str', doc_in='The attribute alias', dtype_out='str', doc_out='The attribute name (dev_name/att_name)')
    def DbGetAliasAttribute(self, argin):
        """ Get the attribute name from the given alias.
        If the given alias is not found in database, returns an empty string

        :param argin: The attribute alias
        :type: tango.DevString
        :return: The attribute name (dev_name/att_name)
        :rtype: tango.DevString """
        self._log.debug("In DbGetAliasAttribute()")
        alias_name = argin[0]
        return self.db.get_alias_attribute(alias_name)

    @command(dtype_in='str', doc_in='Device server name', doc_out='none')
    def DbDeleteServerInfo(self, argin):
        """ delete info related to a Tango devvice server process

        :param argin: Device server name
        :type: tango.DevString
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbDeleteServerInfo()")
        self.db.delete_server_info(argin)

    @command(dtype_in=('str',), doc_in='Str[0] = Tango class name\nStr[1] = Attribute name filter (eg: att*)', dtype_out=('str',), doc_out='Str[0] = Class attribute name\nStr[n] = Class attribute name')
    def DbGetClassAttributeList(self, argin):
        """ Get attrilute list for a given Tango class with a specified filter

        :param argin: Str[0] = Tango class name
        Str[1] = Attribute name filter (eg: att*)
        :type: tango.DevVarStringArray
        :return: Str[0] = Class attribute name
        Str[n] = Class attribute name
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetClassAttributeList()")
        class_name = argin[0]
        wildcard = replace_wildcard(argin[1])
        return self.db.get_class_attribute_list(class_name, wildcard)

    @command(dtype_in=('str',), doc_in='Str[0] = Full device server process name\nStr[1] = Device name\nStr[2] = Tango class name', doc_out='none')
    def DbAddDevice(self, argin):
        """ Add a Tango class device to a specific device server

        :param argin: Str[0] = Full device server process name
        Str[1] = Device name
        Str[2] = Tango class name
        :type: tango.DevVarStringArray
        :return:
        :rtype: tango.DevVoid """
        self._log.debug("In DbAddDevice()")

        if len(argin) < 3:
            self.warn_stream("DataBase::AddDevice(): incorrect number of input arguments ")
            th_exc(DB_IncorrectArguments,
                   "incorrect no. of input arguments, needs at least 3 (server,device,class)",
                   "DataBase::AddDevice()")

        self.info_stream("DataBase::AddDevice(): insert %s server with device %s",argin[0],argin[1])
        server_name, d_name, klass_name = argin[:3]
        if len(argin) > 3:
            alias = argin[3]
        else:
            alias = None

        ret, dev_name, dfm = check_device_name(d_name)
        if not ret:
            th_exc(DB_IncorrectDeviceName,
                  "device name (" + d_name + ") syntax error (should be [tango:][//instance/]domain/family/member)",
                  "DataBase::AddDevice()")
        # Lock table
        self.db.add_device(server_name, (dev_name, dfm) , klass_name, alias=alias)

    @command(dtype_in=('str',), doc_in='argin[0] : server name\nargin[1] : class name', dtype_out=('str',), doc_out='The list of devices for specified server and class.')
    def DbGetDeviceList(self, argin):
        """ Get a list of devices for specified server and class.

        :param argin: argin[0] : server name
        argin[1] : class name
        :type: tango.DevVarStringArray
        :return: The list of devices for specified server and class.
        :rtype: tango.DevVarStringArray """
        self._log.debug("In DbGetDeviceList()")
        server_name = replace_wildcard(argin[0])
        class_name = replace_wildcard(argin[1])
        return self.db.get_device_list(server_name, class_name)


# DbExportDevice is executed in the post_init_cb function below.
# It needs to be separated from the actual device to prevent the device in
# gevent mode to queue the request to the gevent thread and waitting for it.
def DbExportDevice(self, argin):
    """ Export a device to the database

    :param argin: Str[0] = Device name
    Str[1] = CORBA IOR
    Str[2] = Device server process host name
    Str[3] = Device server process PID or string ``null``
    Str[4] = Device server process version
    :type: tango.DevVarStringArray
    :return:
    :rtype: tango.DevVoid """
    self._log.debug("In DbExportDevice()")
    if len(argin) < 5:
        self.warn_stream("DataBase::DbExportDevice(): insufficient export info for device ")
        th_exc(DB_IncorrectArguments,
               "insufficient export info for device",
               "DataBase::ExportDevice()")

    dev_name, IOR, host, pid, version = argin[:5]
    dev_name = dev_name.lower()
    if pid.lower() == 'null':
        pid = "-1"
    self.db.export_device(dev_name, IOR, host, pid, version)


def main(argv = None):
    #Parameters management
    global options
    if argparse:
        parser = argparse.ArgumentParser()
        parser.add_argument("--db_access",dest="db_access",default="sqlite3",
                            help="database type")
        parser.add_argument("-e", "--embedded",dest="embedded",default=False,
                            action="store_true")
        parser.add_argument("--logging_level","-l",dest="logging_level",type=int,
                            default=0,help="logging_level 0:WARNING,1:INFO,2:DEBUG")
        parser.add_argument("--port",dest="port",default=None, type=int,
                            help="database port")
        parser.add_argument('argv',nargs=argparse.REMAINDER)
        options = parser.parse_args(argv)
        options.argv = ["DataBaseds"] + options.argv
    else:
        parser = OptionParser()
        parser.add_option("--db_access",dest="db_access",default="sqlite3",
                          help="database type")
        parser.add_option("-l", "--logging_level",dest="logging_level",default=0,
                          help="logging_level 0:WARNING,1:INFO,2:DEBUG")
        parser.add_option("-e","--embedded",dest="embedded",default=False,
                          action="store_true")
        parser.add_option("--port",dest="port",default=10000, type=int,
                          help="database port")
        (options,args) = parser.parse_args(argv)
        options.argv = ["DataBaseds"] + args

    # Check plugin availability
    get_plugin(options.db_access)

    port = options.port
    if port is None:
        try:
            _, port = tango.ApiUtil.get_env_var("TANGO_HOST").split(":")
        except:
            port = 10000

    options.argv += ["-ORBendPoint", "giop:tcp::{0}".format(port)]

    log_fmt = '%(threadName)-14s %(levelname)-8s %(asctime)s %(name)s: %(message)s'
    if options.logging_level == 1:
        logging_level = logging.INFO
    elif options.logging_level == 2:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.WARNING
    logging.basicConfig(format=log_fmt, stream=sys.stdout,level=logging_level)
    try:
        db_name = "sys/database/" + options.argv[1]
        set_db_name(db_name)
        if options.embedded:
            __run_embedded(db_name, options.argv)
        else:
            __run(db_name, options.argv)
    except Exception as e:
        import traceback
        traceback.print_exc()


def __monkey_patch_database_class():
    DataBaseClass = DataBase.TangoClassClass

    def device_factory(self, device_list):
        """for internal usage only"""

        dev_name = get_db_name()

        klass = self.__class__
        klass_name = klass.__name__
        info = get_class_by_class(klass)
        klass = get_constructed_class_by_class(klass)

        if info is None:
            raise RuntimeError("Device class '%s' is not " \
                               "registered" % klass_name)

        if klass is None:
            raise RuntimeError("Device class '%s' as not been " \
                               "constructed" % klass_name)

        deviceClassClass, deviceImplClass, deviceImplName = info
        deviceImplClass._device_class_instance = klass

        device = self._new_device(deviceImplClass, klass, dev_name)
        self._add_device(device)
        tmp_dev_list = [device]

        self.dyn_attr(tmp_dev_list)

        self.export_device(device, "database")
        self.py_dev_list += tmp_dev_list

    DataBaseClass.device_factory = device_factory


def __monkey_patch_util(util):
    # trick util to execute orb_run instead of the usual server_run
    util._original_server_run = util.server_run
    util.server_run = util.orb_run


def __run(db_name,argv):
    """
    Runs the Database DS as a standalone database. Run it with::

        ./DataBaseds pydb-test -ORBendPoint giop:tcp::11000
    """
    tango.Util.set_use_db(False)
    util = tango.Util(argv)
    __monkey_patch_util(util)
    __monkey_patch_database_class()

    dbi = DbInter()
    util.set_interceptors(dbi)

    def post_init_cb():
        logging.debug("post_init_cb()")
        util = tango.Util.instance()
        dserver = util.get_dserver_device()
        dserver_name = dserver.get_name()
        dserver_ior = util.get_dserver_ior(dserver)
        dbase = util.get_device_by_name(db_name)
        dbase_name = dbase.get_name()
        dbase_ior = util.get_device_ior(dbase)
        host = util.get_host_name()
        pid = util.get_pid_str()
        version = util.get_version_str()
        DbExportDevice(dbase, [dserver_name, dserver_ior, host, pid, version])
        DbExportDevice(dbase, [dbase_name, dbase_ior, host, pid, version])

    run((DataBase,), args=argv, util=util, post_init_callback=post_init_cb,
        green_mode=GreenMode.Gevent, verbose=True)


def __run_embedded(db_name,argv):
    """Runs the Database device server embeded in another TANGO Database
    (just like any other TANGO device server)"""
    __monkey_patch_database_class()

    run((DataBase,), args=argv, util=util, green_mode=GreenMode.Gevent)


if __name__ == '__main__':
    main()
