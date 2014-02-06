# -----------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

"""An IPython profile designed to provide a user friendly interface to Tango"""

from __future__ import print_function

__all__ = ["load_config", "load_ipython_extension", "unload_ipython_extension",
           "run"]

import os
import re
import io
import sys
import operator
import textwrap

from IPython.core.error import UsageError
from IPython.utils.ipstruct import Struct
from IPython.core.page import page
from IPython.core.interactiveshell import InteractiveShell
from IPython.config.application import Application
from IPython.terminal.ipapp import launch_new_instance

import PyTango
import PyTango.utils


_TG_EXCEPTIONS = PyTango.DevFailed, PyTango.ConnectionFailed, \
    PyTango.CommunicationFailed, \
    PyTango.NamedDevFailed, PyTango.NamedDevFailedList, \
    PyTango.WrongNameSyntax, PyTango.NonDbDevice, PyTango.WrongData, \
    PyTango.NonSupportedFeature, PyTango.AsynCall, \
    PyTango.AsynReplyNotArrived, PyTango.EventSystemFailed, \
    PyTango.DeviceUnlocked, PyTango.NotAllowed

_DB_SYMB = "db"
_DFT_TANGO_HOST = None
_TANGO_STORE = "__tango_store"
_TANGO_ERR = "__tango_error"
_PYTHON_ERR = "__python_error"
_tango_init = False

#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
# IPython utilities
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

def get_pylab_mode():
    return get_app().pylab

def get_color_mode():
    return get_config().InteractiveShell.colors

def get_app():
    #return TerminalIPythonApp.instance()
    return Application.instance()

def get_shell():
    """Get the global InteractiveShell instance."""
    return get_app().shell

def get_ipapi():
    """Get the global InteractiveShell instance."""
    return InteractiveShell.instance()

def get_config():
    return get_app().config

def get_editor():
    return get_ipapi().editor

def get_user_ns():
    return get_shell().user_ns

class DeviceClassCompleter(object):
    """Completer class that returns the list of devices of some class when
    called. """
    
    def __init__(self, klass, devices):
        self._klass = klass
        self._devices = devices
    
    def __call__(self, ip, evt):
        return self._devices


# Rewrite DeviceProxy constructor because the database context that the user is
# using may be different than the default TANGO_HOST. What we do is always append
# the name of the database in usage to the device name given by the user (in case 
# he doesn't give a database name him(her)self, of course.
#__DeviceProxy_init_orig__ = PyTango.DeviceProxy.__init__
#def __DeviceProxy__init__(self, dev_name):
#    db = __get_db()
#    if db is None: return
#    if dev_name.count(":") == 0:
#        db_name = "%s:%s" % (db.get_db_host(), db.get_db_port())
#        dev_name = "%s/%s" % (db_name, dev_name)
#    __DeviceProxy_init_orig__(self, dev_name)
#PyTango.DeviceProxy.__init__ = __DeviceProxy__init__

#-------------------------------------------------------------------------------
# Completers
#-------------------------------------------------------------------------------

def __DeviceProxy_completer(ip, evt):
    db = __get_db()
    if db is None: return
    ret = list(db._db_cache.devices.keys())
    ret.extend(db._db_cache.aliases.keys())
    return ret

def __DeviceClass_completer(ip, evt):
    db = __get_db()
    if db is None: return
    return list(db._db_cache.klasses.keys())

def __DeviceAlias_completer(ip, evt):
    db = __get_db()
    if db is None: return
    return list(db._db_cache.aliases.keys())

def __AttributeAlias_completer(ip, evt):
    db = __get_db()
    if db is None: return
    return list(db._db_cache.attr_aliases.keys())

def __PureDeviceProxy_completer(ip, evt):
    db = __get_db()
    if db is None: return
    return list(db._db_cache.devices.keys())

def __AttributeProxy_completer(ip, evt):
    db = __get_db()
    if db is None: return
    cache = db._db_cache
    
    symb = evt.symbol
    n = symb.count("/")
    ret, devs, dev_aliases = None, cache.devices, cache.aliases
    # dev_list and dev_alias_list are case insensitive. They should only be used
    # to search for elements. Their elements are the same as the keys of the 
    # dictionaries devs and dev_aliases respectively
    dev_list, dev_alias_list = cache.device_list, cache.alias_list
    dev_name = None
    if n == 0:
        # means it can be an attr alias, a device name has alias or as full device name
        ret = list(cache.get("attr_aliases").keys())
        ret.extend([ d+"/" for d in devs ])
        ret.extend([ d+"/" for d in dev_aliases ])
        # device alias found!
        if symb in dev_alias_list:
            dev_name = symb
    elif n == 1:
        # it can still be a full device name
        ret = [ d+"/" for d in devs ]
        # it can also be devalias/attr_name
        dev, attr = symb.split("/")
        if dev in dev_alias_list:
            dev_name = dev
    elif n == 2:
        # it is for sure NOT an attribute alias or a device alias
        ret = [ d+"/" for d in devs ]
        # device found!
        if symb in dev_list:
            dev_name = symb
    elif n == 3:
        # it is for sure a full attribute name
        dev, sep, attr = symb.rpartition("/")
        if dev in dev_list:
            dev_name = dev

    if dev_name is None:
        return ret
    
    try:
        d = __get_device_proxy(dev_name)
        # first check in cache for the attribute list
        if not hasattr(d, "_attr_list"):
            d._attr_list = d.get_attribute_list()
        if ret is None:
            ret = []
        ret.extend([ "%s/%s" % (dev_name, a) for a in d._attr_list ])
    except:
        # either DeviceProxy could not be created or device is not online
        pass

    return ret

def __get_device_proxy(dev_name):
    db = __get_db()
    if db is None: return
    cache = db._db_cache
    from_alias = cache.aliases.get(dev_name)
    
    if from_alias is not None:
        dev_name = from_alias

    data = cache.devices.get(dev_name)
    if data is not None:
        d = data[3]
        if d is None:
            try:
                d = data[3] = PyTango.DeviceProxy(dev_name)
            except:
                pass
        return d

def __get_device_subscriptions(dev_name):
    db = __get_db()
    if db is None: return
    cache = db._db_cache
    from_alias = cache.aliases.get(dev_name)
    
    if from_alias is not None:
        dev_name = from_alias

    data = cache.devices.get(dev_name)
    if data is not None:
        return data[4]

__monitor_completer = __AttributeProxy_completer

#-------------------------------------------------------------------------------
# Magic commands
#-------------------------------------------------------------------------------

def refreshdb(self, parameter_s=''):
    init_db(parameter_s)

def switchdb(self, parameter_s=''):
    """Switches the active tango Database.
    
    Usage: switchdb <host>[(:| )<port>]

    <port> is optional. If not given it defaults to 10000.
    
    Examples:
    In [1]: switchdb homer:10005
    In [2]: switchdb homer 10005
    In [3]: switchdb homer"""
    
    if parameter_s == '':
        raise UsageError("%switchdb: Must specify a tango database name. "\
                         "See '%switchdb?'")
    return init_db(parameter_s)

def lsdev(self, parameter_s=''):
    """Lists all known tango devices.
    
    Usage: lsdev [<device name filter(regular expression)]
    
    Examples:
    In [1]: lsdev
    In [2]: lsdev sys.*"""
    
    if parameter_s:
        reg_exp = re.compile(parameter_s, re.IGNORECASE)
    else:
        reg_exp = None
    
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database. Device list is empty")
        return
    data = db._db_cache.devices

    s = io.BytesIO()
    lengths = 40, 25, 25, 20
    title = "Device", "Alias", "Server", "Class"
    templ = "{0:{l[0]}} {1:{l[1]}} {2:{l[2]}} {3:{l[3]}}"
    msg = templ.format(*title, l=lengths)
    print(msg, file=s)
    print(*map(operator.mul, lengths, len(lengths)*"-"), file=s)
    for d, v in data.items():
        if reg_exp and not reg_exp.match(d):
            continue
        print(templ.format(d, v[0], v[1], v[2], l=lengths), file=s)
    s.seek(0)
    page(s.read())

def lsdevclass(self, parameter_s=''):
    """Lists all known tango device classes.
    
    Usage: lsdevclass [<class name filter(regular expression)]
    
    Examples:
    In [1]: lsdevclass
    In [2]: lsdevclass Motor.*"""
    
    if parameter_s:
        reg_exp = re.compile(parameter_s, re.IGNORECASE)
    else:
        reg_exp = None
    
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database. Device class list is empty")
        return
    data = db._db_cache.klasses

    s = io.BytesIO()
    data = [ "%-030s" % klass for klass in data.keys() if not reg_exp or reg_exp.match(klass) ]
    s = textwrap.fill(" ".join(data), 80)
    page(s)

def lsserv(self, parameter_s=''):
    """Lists all known tango servers.
    
    Usage: lsserv [<class name filter(regular expression)]
    
    Examples:
    In [1]: lsserv
    In [2]: lsserv Motor/.*"""
    
    if parameter_s:
        reg_exp = re.compile(parameter_s, re.IGNORECASE)
    else:
        reg_exp = None
    
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database. Device class list is empty")
        return
    data = db._db_cache.servers

    s = io.BytesIO()
    data = [ "%-030s" % server for server in data.keys() if not reg_exp or reg_exp.match(server) ]
    s = textwrap.fill(" ".join(data), 80)
    page(s)

def tango_error(self, parameter_s=''):
    """Displays detailed information about the last tango error"""
    global _TANGO_ERR
    err_info = get_user_ns().get(_TANGO_ERR)
    if err_info is None:
        print("No tango error reported so far.")
        return
    print("Last tango error:")
    print(err_info[1])

def python_error(self, parameter_s=''):
    """Displays detailed information about the last python error"""
    global _PYTHON_ERR
    err_info = get_user_ns().get(_PYTHON_ERR)
    if err_info is None:
        print("No error reported so far.")
        return
    ip = get_ipapi()
    etype, evalue, etb = err_info[:3]
    ip.InteractiveTB(etype=etype, evalue=evalue, etb=etb, tb_offset=None)

_EVT_LOG = None
def __get_event_log():
    global _EVT_LOG
    if _EVT_LOG is None:
#        qthreads = get_config().q4thread
#        if qthreads:
#            import ipy_qt
#            model = ipy_qt.EventLoggerTableModel(capacity=10000)
#            _EVT_LOG = ipy_qt.EventLogger(model=model)
#            _EVT_LOG.setWindowTitle("ITango - Event Logger Table")
#        else:
        import PyTango.ipython.eventlogger
        _EVT_LOG = PyTango.ipython.eventlogger.EventLogger(capacity=10000, pager=page)
    return _EVT_LOG

def mon(self, parameter_s=''):
    """Monitor a given attribute.
    
    %mon -a <attribute name>           - activates monitoring of given attribute
    %mon -d <attribute name>           - deactivates monitoring of given attribute
    %mon -r                            - deactivates monitoring of all attributes
    %mon -i <id>                       - displays detailed information for the event with given id
    %mon -l <dev filter> <attr filter> - shows event table filtered with the regular expression for attribute name
    %mon                               - shows event table (= %mon -i .* .*)"""
    
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database.")
        return
    
    # make sure parameter_s is a str and not a unicode
    parameter_s = str(parameter_s)
    opts, args = self.parse_options(parameter_s,'adril', mode='list')
    if len(args) > 3:
        raise UsageError("%mon: too many arguments")
    if 'd' in opts:
        try:
            todel = args[0]
        except IndexError:
            raise UsageError("%mon -d: must provide an attribute to unmonitor")
        else:
            try:
                dev, _, attr = todel.rpartition("/")
                subscriptions = __get_device_subscriptions(dev)
                attr_id = subscriptions[attr.lower()]
                del subscriptions[attr.lower()]
                d = __get_device_proxy(dev)
                d.unsubscribe_event(attr_id)
                print("Stopped monitoring '%s'" % todel)
            except KeyError:
                raise UsageError("%%mon -d: Not monitoring '%s'" % todel)
                    
    elif 'a' in opts:
        try:
            toadd = args[0]
        except IndexError:
            raise UsageError("%mon -a: must provide an attribute to monitor")
        dev, _, attr = toadd.rpartition("/")
        subscriptions = __get_device_subscriptions(dev)
        attr_id = subscriptions.get(attr.lower())
        if attr_id is not None:
            raise UsageError("%%mon -a: Already monitoring '%s'" % toadd)
        d = __get_device_proxy(dev)
        w = __get_event_log()
        model = w.model()
        attr_id = d.subscribe_event(attr, PyTango.EventType.CHANGE_EVENT,
                                    model, [])
        subscriptions[attr.lower()] = attr_id
        print("'%s' is now being monitored. Type 'mon' to see all events" % toadd)
    elif 'r' in opts:
        for d, v in db._db_cache.devices.items():
            d, subs = v[3], v[4]
            for _id in subs.values():
                d.unsubscribe_event(_id)
            v[4] = {}
    elif 'i' in opts:
        try:
            evtid = int(args[0])
        except IndexError:
            raise UsageError("%mon -i: must provide an event ID")
        except ValueError:
            raise UsageError("%mon -i: must provide a valid event ID")
        try:
            w = __get_event_log()
            e = w.getEvents()[evtid]
            if e.err:
                print(str(PyTango.DevFailed(*e.errors)))
            else:
                print(str(e))
        except IndexError:
            raise UsageError("%mon -i: must provide a valid event ID")
    elif 'l' in opts:
        try:
            dexpr = args[0]
            aexpr = args[1]
        except IndexError:
            raise UsageError("%mon -l: must provide valid device and " \
                             "attribute filters")
        w = __get_event_log()
        w.show(dexpr, aexpr)
    else:
        w = __get_event_log()
        w.show()

#-------------------------------------------------------------------------------
# Useful functions (not magic commands but accessible from CLI as normal python
# functions)
#-------------------------------------------------------------------------------

def get_device_map():
    """Returns a dictionary where keys are device names and value is a sequence
    of 4 elements: 
        - alias name (empty string if no alias is defined)
        - tango server name (full tango server name <name>/<instance>)
        - tango class name
        - DeviceProxy to the device or None if it hasn't been initialized yet
          (this last element is for internal tango usage only. If you need a 
           DeviceProxy to this device, create your own)"""
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database.")
        return
    return db._db_cache.devices

def get_server_map():
    """Returns a dictionary where keys are server names (<name>/<instance>)
    and value is a sequence of device names that belong to the server"""
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database.")
        return
    return db._db_cache.servers

def get_class_map():
    """Returns a dictionary where keys are the tango classes and value is a 
    sequence of device names that belong to the tango class"""
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database.")
        return
    return db._db_cache.klasses

def get_alias_map():
    """Returns a dictionary where keys are the tango device aliases and value 
    is a the tango device name"""
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database.")
        return
    return db._db_cache.aliases

def get_device_list():
    """Returns a case insensitive list of device names for the current 
    database"""
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database.")
        return
    return db._db_cache.device_list

def get_alias_list():
    """Returns a case insensitive list of device aliases for the current 
    database"""
    db = __get_db()
    if db is None:
        print("You are not connected to any Tango Database.")
        return
    return db._db_cache.alias_list    
    
#-------------------------------------------------------------------------------
# Private helper methods
#-------------------------------------------------------------------------------

def __exc_handler(ip, etype, value, tb, tb_offset=None):
    global _TG_EXCEPTIONS
    user_ns = get_user_ns()
    if etype in _TG_EXCEPTIONS:
        global _TANGO_ERR
        user_ns[_TANGO_ERR] = etype, value, tb, tb_offset
        if len(value.args):
            v = value.args[0]
            print("%s: %s" % (v.reason ,v.desc))
        else:
            print("Empty DevFailed")
        print("(For more detailed information type: tango_error)")
    else:
        global _PYTHON_ERR
        user_ns[_PYTHON_ERR] = etype, value, tb, tb_offset
        print(etype.__name__ + ": " + str(value))
        print("(For more detailed information type: python_error)")

def __get_default_tango_host():
    global _DFT_TANGO_HOST
    if _DFT_TANGO_HOST is None:
        try:
            db = PyTango.Database()
            _DFT_TANGO_HOST = "%s:%s" % (db.get_db_host(), db.get_db_port())
        except:
            pass
    return _DFT_TANGO_HOST

def __get_db(host_port=None):
    """host_port == None: Use current DB whatever it is or create
                          default if doesn't exist
       host_port == '' : use default db. If it is not the current db, switch
                         current db to it and return it
       host_port == ... : if ... is not the current db, switch current db to it
                          and return it
    """
    
    ip = get_ipapi()
    user_ns = get_user_ns()

    global _DB_SYMB
    db = user_ns.get(_DB_SYMB)
    
    if host_port is None:
        if db is None:
            host_port = __get_default_tango_host()
    elif host_port == '':
        host_port = __get_default_tango_host()
    else:
        host_port = host_port.strip().replace(" ",":")
        if host_port.count(":") == 0:
            host_port += ":10000"
    
    if host_port is not None:
        host_port = str(host_port)
    
    if db is None:
        create_db = True
    elif host_port is None:
        create_db = False
    else:
        old_host_port = "%s:%s" % (db.get_db_host(), db.get_db_port())
        create_db = old_host_port != host_port

    if create_db:
        try:
            db = PyTango.Database(*host_port.split(":"))
            
            user_ns["DB_NAME"] = host_port
        except Exception as e:
            if db:
                print("\nCould not access Database %s:" % host_port)
                print(str(e))
                old_host_port = "%s:%s" % (db.get_db_host(), db.get_db_port())
                print("Maintaining connection to Database", old_host_port)
                user_ns["DB_NAME"] = old_host_port
            else:
                print("\nCould not access any Database. Make sure:")
                print("\t- .tangorc, /etc/tangorc or TANGO_HOST environment is defined.")
                print("\t- the Database DS is running")
                user_ns["DB_NAME"] = "OFFLINE"
                
        # register the 'db' in the user namespace
        user_ns.update({ _DB_SYMB : db })
        
    return db

def __get_obj_name(o):
    try:
        n = o.__name__
    except:
        try:
            n = o.__class__.__name__
        except:
            n = "<unknown>"
    return n

def __completer_wrapper(f):
    def wrapper(ip, evt):
        try:
            return f(ip, evt)
        except Exception as e:
            print()
            print("An unexpected exception ocorred during ITango command completer.")
            print("Please send a bug report to the PyTango team with the following information:")
            print(80*"-")
            print("Completer:",__get_obj_name(f))
            print(80*"-")
            import traceback
            traceback.print_exc()
            print(80*"-")
            raise e
    return wrapper

def __expose_magic(ip, name, fn, completer_func=None):
    ip.define_magic(name, fn)
    
    if completer_func is None:
        return
    
    # enable macro param completion
    ip.set_hook('complete_command', completer_func, re_key = ".*" + name)

def __unexpose_magic(ip, name):
    delattr(ip, name)

def __build_color_scheme(ip, name):
    import IPython.Prompts
    import IPython.PyColorize
    import IPython.excolors
    from IPython.utils.coloransi import TermColors, InputTermColors

    # make some schemes as instances so we can copy them for modification easily:
    PromptColors = IPython.Prompts.PromptColors
    ANSICodeColors = IPython.PyColorize.ANSICodeColors
    ExceptionColors = IPython.excolors.ExceptionColors
    TBColors = ip.IP.InteractiveTB.color_scheme_table
    SyntaxColors = ip.IP.SyntaxTB.color_scheme_table
    InspectColors = IPython.OInspect.InspectColors
    
    promptTangoColors = PromptColors['Linux'].copy(name)
    ANSITangoColors = ANSICodeColors['Linux'].copy(name)
    exceptionTangoColors = ExceptionColors['Linux'].copy(name)
    TBTangoColors = TBColors['Linux'].copy(name)
    syntaxTangoColors = SyntaxColors['Linux'].copy(name)
    inspectTangoColors = InspectColors['Linux'].copy(name)
    
    # initialize prompt with default tango colors
    promptTangoColors.colors.in_prompt  = InputTermColors.Purple
    promptTangoColors.colors.in_number  = InputTermColors.LightPurple
    promptTangoColors.colors.in_prompt2 = InputTermColors.Purple
    promptTangoColors.colors.out_prompt = TermColors.Blue
    promptTangoColors.colors.out_number = TermColors.LightBlue

    ret= { "prompt" : (PromptColors, promptTangoColors),
          "ANSI"   : (ANSICodeColors, ANSITangoColors),
          "except" : (ExceptionColors, exceptionTangoColors),
          "TB"     : (TBColors, TBTangoColors),
          "Syntax" : (SyntaxColors, syntaxTangoColors),
          "Inspect": (InspectColors, inspectTangoColors) }

    if ip.IP.isthreaded:
        TBThreadedColors = ip.IP.sys_excepthook.color_scheme_table
        TBThreadedTangoColors = TBThreadedColors['Linux'].copy(name)
        ret["TBThreaded"] = TBThreadedColors, TBThreadedTangoColors
    return ret

#-------------------------------------------------------------------------------
# Initialization methods
#-------------------------------------------------------------------------------

def init_pytango(ip):
    """Initializes the IPython environment with PyTango elements"""

    # export symbols to IPython namepspace
    ip.ex("import PyTango")
    ip.ex("from PyTango import DeviceProxy, AttributeProxy, Database, Group")
    ip.ex("Device = DeviceProxy")
    ip.ex("Attribute = AttributeProxy")

    # add completers
    dp_completer = __completer_wrapper(__DeviceProxy_completer)
    attr_completer = __completer_wrapper(__AttributeProxy_completer)
    ip.set_hook('complete_command', dp_completer, re_key = ".*DeviceProxy[^\w\.]+")
    ip.set_hook('complete_command', dp_completer, re_key = ".*Device[^\w\.]+")
    ip.set_hook('complete_command', attr_completer, re_key = ".*AttributeProxy[^\w\.]+")
    ip.set_hook('complete_command', attr_completer, re_key = ".*Attribute[^\w\.]+")
    
    ip.set_custom_exc((Exception,), __exc_handler)

def init_db(parameter_s=''):
    ip = get_ipapi()
    user_ns = get_user_ns()
    global _DB_SYMB
    old_db = user_ns.get(_DB_SYMB)
    
    db = __get_db(parameter_s)
    
    if old_db is not None and hasattr(old_db, "_db_cache"):
        old_junk = old_db._db_cache["junk"].keys()
        for e in old_junk:
            del user_ns[e]
    else:
        old_junk = ()
        
    if db is None: return
    
    os.environ["TANGO_HOST"] = "%s:%s" % (db.get_db_host(), db.get_db_port())
    
    # Initialize device and server information
    query = "SELECT name, alias, server, class FROM device order by name"
    
    r = db.command_inout("DbMySqlSelect", query)
    row_nb, column_nb = r[0][-2], r[0][-1]
    data = r[1]
    assert row_nb == len(data) / column_nb
    devices, aliases, servers, klasses = data[0::4], data[1::4], data[2::4], data[3::4]

    #CD = PyTango.utils.CaselessDict
    CD = dict
    dev_dict, serv_dict, klass_dict, alias_dict = CD(), CD(), CD(), CD()
    
    for device, alias, server, klass in zip(devices, aliases, servers, klasses):
        dev_lower = device.lower()

        # hide dserver devices
        if dev_lower.startswith("dserver/"): continue
        
        # hide alias that start with "_"
        if alias and alias[0] == "_": alias = ''
        
        # last None below is to be filled by DeviceProxy object on demand
        # last empty dict<str, int> where keys is attribute name and value is 
        # the subscription id
        dev_dict[device] = [alias, server, klass, None, {}]
        serv_devs = serv_dict.get(server)
        if serv_devs is None:
            serv_dict[server] = serv_devs = []
        serv_devs.append(device)
        klass_devs = klass_dict.get(klass)
        if klass_devs is None:
            klass_dict[klass] = klass_devs = []
        klass_devs.append(device)
        if len(alias):
            alias_dict[alias] = device
            serv_devs.append(alias)
            klass_devs.append(alias)

    exposed_klasses = {}
    excluded_klasses = "DServer",
    for klass, devices in klass_dict.items():
        if klass in excluded_klasses:
            continue
        exists = klass in user_ns
        if not exists or klass in old_junk:
            c = DeviceClassCompleter(klass, devices)
            ip.set_hook('complete_command', c, re_key = ".*" + klass + "[^\w\.]+")
            exposed_klasses[klass] = PyTango.DeviceProxy
    
    # expose classes no user namespace
    user_ns.update(exposed_klasses)
    
    # Initialize attribute information
    query = "SELECT name, alias FROM attribute_alias order by alias"

    r = db.command_inout("DbMySqlSelect", query)
    row_nb, column_nb = r[0][-2], r[0][-1]
    data = r[1]
    assert row_nb == len(data) / column_nb
    attributes, aliases = data[0::2], data[1::2]
    
    attr_alias_dict = {}
    for attribute, alias in zip(attributes, aliases):
        if len(alias):
            attr_alias_dict[alias] = attribute
    
    device_list = PyTango.utils.CaselessList(dev_dict.keys())
    alias_list = PyTango.utils.CaselessList(alias_dict.keys())
    attr_alias_list = PyTango.utils.CaselessList(attr_alias_dict.keys())
    
    # Build cache
    db_cache = Struct(devices=dev_dict, aliases=alias_dict,
        servers=serv_dict, klasses=klass_dict, junk=exposed_klasses,
        attr_aliases=attr_alias_dict, device_list=device_list,
        alias_list=alias_list, attr_alias_list=attr_alias_list)
    
    db._db_cache = db_cache

    # Add this DB to the list of known DBs (for possible use in magic commands)
    if db.get_db_port_num() == 10000:
        db_name = db.get_db_host()
    else:
        db_name = "%s:%s" % (db.get_db_host(), db.get_db_port())
    return db

def init_magic(ip):

    import IPython.core.magic

    new_style_magics = hasattr(IPython.core.magic, 'Magics') and hasattr(IPython.core.magic, 'magics_class')

    if new_style_magics:
        @IPython.core.magic.magics_class
        class Tango(IPython.core.magic.Magics):
            
            refreshdb = IPython.core.magic.line_magic(refreshdb)
            switchdb = IPython.core.magic.line_magic(switchdb)
            lsdev = IPython.core.magic.line_magic(lsdev)
            lsdevclass = IPython.core.magic.line_magic(lsdevclass)
            lsserv = IPython.core.magic.line_magic(lsserv)
            tango_error = IPython.core.magic.line_magic(tango_error)
            python_error = IPython.core.magic.line_magic(python_error)
            mon = IPython.core.magic.line_magic(mon)

        ip.register_magics(Tango)
        ip.set_hook('complete_command', __monitor_completer, re_key = ".*" + "mon")
    else:
        __expose_magic(ip, "refreshdb", refreshdb)
        __expose_magic(ip, "switchdb", switchdb)
        __expose_magic(ip, "lsdev", lsdev)
        __expose_magic(ip, "lsdevclass", lsdevclass)
        __expose_magic(ip, "lsserv", lsserv)
        __expose_magic(ip, "tango_error", tango_error)
        __expose_magic(ip, "python_error", python_error)
        __expose_magic(ip, "mon", mon, __monitor_completer)
    
    get_user_ns().update({"get_device_map"   : get_device_map,
                   "get_server_map"  : get_server_map,
                   "get_class_map"   : get_class_map,
                   "get_alias_map"   : get_alias_map,
                   "get_device_list" : get_device_list,
                   "get_alias_list"  : get_alias_list})

def complete(text):
    """a super complete!!!!"""
    self = get_ipapi().IP
    complete = self.Completer.complete
    state = 0
    comps = set()
    while True:
        newcomp = complete("", state, line_buffer=text)
        if newcomp is None:
            break
        comps.add(newcomp)
        state += 1
    outcomps = sorted(comps)
    return outcomps

__DIRNAME = os.path.dirname(os.path.abspath(__file__))
__RES_DIR = os.path.join(__DIRNAME, os.path.pardir, 'resource')

class __TangoDeviceInfo(object):
    """Helper class for when DeviceProxy.info() is not available"""
    
    def __init__(self, dev):
        try:
            db = dev.get_device_db()
            klass = db.get_class_for_device(dev.dev_name())
            self.dev_class = self.dev_type = klass
        except:
            self.dev_class = self.dev_type = 'Device'
        self.doc_url = 'http://www.esrf.eu/computing/cs/tango/tango_doc/ds_doc/index.html'
        self.server_host = 'Unknown'
        self.server_id = 'Unknown'
        self.server_version = 1
    
        
def __get_device_class_icon(klass="Device"):
    icon_prop = "__icon"
    db = __get_db()
    try:
        icon_filename = db.get_class_property(klass, icon_prop)[icon_prop]
        if icon_filename:
            icon_filename = icon_filename[0]
        else:            
            icon_filename = klass.lower() + os.path.extsep + "png"
    except:
        icon_filename = klass.lower() + os.path.extsep + "png"
    
    if os.path.isabs(icon_filename):
        icon = icon_filename
    else:
        icon = os.path.join(__RES_DIR, icon_filename)
    if not os.path.isfile(icon):
        icon = os.path.join(__RES_DIR, "_class.png")
    return icon


__DEV_CLASS_HTML_TEMPLATE = """\
<table border="0" cellpadding="2" width="100%">
<tr><td width="140" rowspan="7" valign="middle" align="center"><img src="{icon}" height="128"/></td>
    <td width="140">Name:</td><td><b>{name}</b></td></tr>
<tr><td width="140">Super class:</td><td>{super_class}</td></tr>
<tr><td width="140">Database:</td><td>{database}</td></tr>
<tr><td width="140">Description:</td><td>{description}</td></tr>
<tr><td width="140">Documentation:</td><td><a target="_blank" href="{doc_url}">{doc_url}</a></td></tr>
</table>"""

def __get_class_property_str(dev_class, prop_name, default=""):
    data = __get_db().get_class_property(dev_class, prop_name)[prop_name]
    if len(data):
        return data[0]
    else:
        return default

def display_deviceclass_html(dev_class):
    """displayhook function for PyTango.DeviceProxy, rendered as HTML"""
    fmt = dict(name=dev_class)
    db = __get_db()
    try:
        fmt["database"] = db.get_db_host() + ":" + db.get_db_port()
    except:
        try:
            fmt["database"] = db.get_file_name()
        except:
            fmt["database"]  = "Unknown"

    doc_url = __get_class_property_str(dev_class, "doc_url", "www.tango-controls.org")
    try:
        fmt["doc_url"] = doc_url[doc_url.index("http"):]
    except ValueError:
        fmt["doc_url"] = doc_url
    
    fmt['icon'] = __get_device_class_icon(dev_class)
    fmt['super_class'] = __get_class_property_str(dev_class, "InheritedFrom", "DeviceImpl")
    fmt['description'] = __get_class_property_str(dev_class, "Description", "A Tango device class")
    return __DEV_CLASS_HTML_TEMPLATE.format(**fmt)


def __get_device_icon(dev_proxy, klass="Device"):
    icon_prop = "__icon"
    db = dev_proxy.get_device_db()
    try:
        icon_filename = dev_proxy.get_property(icon_prop)[icon_prop]
        if icon_filename:
            icon_filename = icon_filename[0]
        else:
            icon_filename = db.get_class_property(klass, icon_prop)[icon_prop]
            if icon_filename:
                icon_filename = icon_filename[0]
            else:            
                icon_filename = klass.lower() + os.path.extsep + "png"
    except:
        icon_filename = klass.lower() + os.path.extsep + "png"
    
    if os.path.isabs(icon_filename):
        icon = icon_filename
    else:
        icon = os.path.join(__RES_DIR, icon_filename)
    if not os.path.isfile(icon):
        icon = os.path.join(__RES_DIR, "device.png")
    return icon

__DEV_HTML_TEMPLATE = """\
<table border="0" cellpadding="2" width="100%">
<tr><td width="140" rowspan="7" valign="middle" align="center"><img src="{icon}" height="128"/></td>
    <td width="140">Name:</td><td><b>{name}</b></td></tr>
<tr><td width="140">Alias:</td><td>{alias}</td></tr>
<tr><td width="140">Database:</td><td>{database}</td></tr>
<tr><td width="140">Type:</td><td>{dev_class}</td></tr>
<tr><td width="140">Server:</td><td>{server_id}</td></tr>
<tr><td width="140">Server host:</td><td>{server_host}</td></tr>
<tr><td width="140">Documentation:</td><td><a target="_blank" href="{doc_url}">{doc_url}</a></td></tr>
</table>"""

def display_deviceproxy_html(dev_proxy):
    """displayhook function for PyTango.DeviceProxy, rendered as HTML"""
    try:
        info = dev_proxy.info()
    except:
        info = __TangoDeviceInfo(dev_proxy)
    name = dev_proxy.dev_name()
    fmt = dict(dev_class=info.dev_class, server_id=info.server_id,
               server_host=info.server_host, name=name)
    
    try:
        fmt["alias"] = dev_proxy.alias()
    except:
        fmt["alias"] = "-----"

    db = dev_proxy.get_device_db()
    try:
        fmt["database"] = db.get_db_host() + ":" + db.get_db_port()
    except:
        try:
            fmt["database"] = db.get_file_name()
        except:
            fmt["database"]  = "Unknown"

    doc_url = info.doc_url.split("\n")[0]
    try:
        fmt["doc_url"] = doc_url[doc_url.index("http"):]
    except ValueError:
        fmt["doc_url"] = doc_url

    fmt['icon'] = __get_device_icon(dev_proxy, info.dev_class)

    return __DEV_HTML_TEMPLATE.format(**fmt)

__DB_HTML_TEMPLATE = """\
<table border="0" cellpadding="2" width="100%">
<tr><td width="140" rowspan="2" valign="middle" align="center"><img src="{icon}" height="128"/></td>
    <td><b>{name}</b></td></tr>
<tr><td>{info}</td></tr>
</table>"""

def display_database_html(db):
    """displayhook function for PyTango.Database, rendered as HTML"""
    fmt = dict()

    try:
        fmt["name"] = db.get_db_host() + ":" + db.get_db_port()
    except:
        try:
            fmt["name"] = db.get_file_name()
        except:
            fmt["name"]  = "Unknown"

    try:
        fmt["info"] = db.get_info().replace("\n", "<BR/>")
    except:
        fmt["info"] = "Unknown"
    
    fmt['icon'] = os.path.join(__RES_DIR, "database.png")

    return __DB_HTML_TEMPLATE.format(**fmt)

__DEV_ATTR_RW_HTML_TEMPLATE = """\
<table border="0" cellpadding="2" width="100%">
<tr><td colspan="2" bgcolor="{bgcolor}">{name} ({type}, {data_format}, {quality}) at {time}</td></tr>
<tr><td bgcolor="#EEEEEE" width="140">value{r_dim}:</td>
    <td bgcolor="#EEEEEE">{value}</td></tr>
<tr><td bgcolor="#EEEEEE" width="140">w_value{w_dim}:</td>
    <td bgcolor="#EEEEEE">{w_value}</td></tr>
</table>"""

__DEV_ATTR_RO_HTML_TEMPLATE = """\
<table border="0" cellpadding="2" width="100%">
<tr><td colspan="2" bgcolor="{bgcolor}">{name} ({type}, {data_format}, {quality}) at {time}</td></tr>
<tr><td bgcolor="#EEEEEE" width="140">value{r_dim}:</td>
    <td bgcolor="#EEEEEE">{value}</td></tr>
</table>"""

__DEV_ATTR_ERR_HTML_TEMPLATE = """\
<table border="0" cellpadding="2" width="100%">
<tr><td bgcolor="#FF0000">{name} ({type}, {data_format}, {quality}) at {time}</td></tr>
<tr><td bgcolor="#EEEEEE">{error}</td></tr>
</table>"""

QUALITY_TO_HEXCOLOR_STR = {
    PyTango.AttrQuality.ATTR_VALID : ("#00FF00", "#000000"),
    PyTango.AttrQuality.ATTR_INVALID : ("#808080", "#FFFFFF"),    
    PyTango.AttrQuality.ATTR_ALARM : ("#FF8C00", "#FFFFFF"),    
    PyTango.AttrQuality.ATTR_WARNING : ("#FF8C00", "#FFFFFF"),    
    PyTango.AttrQuality.ATTR_CHANGING : ("#80A0FF", "#000000"),
    None : ("#808080", "#000000"), 
}

def display_deviceattribute_html(da):
    """displayhook function for PyTango.DeviceAttribute, rendered as HTML"""
    fmt = dict(name=da.name, type=da.type, data_format=da.data_format)
    template = None
    if da.has_failed:
        fmt['error'] = "\n".join(map(str, da.get_err_stack())).replace("\n", "<br/>")
        
        template = __DEV_ATTR_ERR_HTML_TEMPLATE
    else:
        rd, wd = da.r_dimension, da.w_dimension
        if wd.dim_x == 0 and wd.dim_y == 0 and da.w_value is None:
            template = __DEV_ATTR_RO_HTML_TEMPLATE
        else:
            template = __DEV_ATTR_RW_HTML_TEMPLATE
            fmt['w_value'] = str(da.w_value)
            if da.data_format == PyTango.AttrDataFormat.SCALAR:
                fmt['w_dim'] = ""
            else:
                fmt['w_dim'] = "<br/>(%d, %d)" % (wd.dim_x, wd.dim_y)
        fmt['bgcolor'], fmt['fgcolor'] = QUALITY_TO_HEXCOLOR_STR[da.quality]
        fmt['quality'] = str(da.quality)
        if da.data_format == PyTango.AttrDataFormat.SCALAR:
            fmt['r_dim'] = ""
        else:
            fmt['r_dim'] = "<br/>(%d, %d)" % (rd.dim_x, rd.dim_y)
        fmt['value'] = str(da.value)
        fmt['time'] = str(da.time.todatetime())
    return template.format(**fmt)

__GROUP_HTML_TEMPLATE = """\
<table border="0" cellpadding="2" width="100%">
<tr><td width="100" bgcolor="#EEEEEE">Name:</td><td bgcolor="#EEEEEE">{name}</td></tr>
<tr><td width="100" bgcolor="#EEEEEE">Size:</td><td bgcolor="#EEEEEE">{size}</td></tr>
<tr><td width="100" bgcolor="#EEEEEE">Devices:</td><td bgcolor="#EEEEEE">{devices}</td></tr>
</table>"""

def display_group_html(group):
    devices = group.get_device_list()
    devices = ", ".join(devices)
    fmt=dict(name=group.get_name(), size=group.get_size(), devices=devices) 
    return __GROUP_HTML_TEMPLATE.format(**fmt)

__GROUP_REPLY_HTML_TEMPLATE = """\
<table border="0" cellpadding="2" width="100%">
<tr><td bgcolor="#EEEEEE">{name}</td></tr>
<tr><td>{data}</td></tr>    
"""

__GROUP_REPLY_ERR_HTML_TEMPLATE = """\
<table border="0" cellpadding="2" width="100%">
<tr><td bgcolor="#FF0000">{name}</td></tr>
<tr><td bgcolor="#EEEEEE">{error}</td></tr>
</table>"""
    
def display_groupreply_html(gr):
    fmt = dict(name="%s/%s" % (gr.dev_name(), gr.obj_name()))
    template = None
    if gr.has_failed():
        fmt['error'] = "\n".join(map(str, gr.get_err_stack())).replace("\n", "<br/>")
        template = __GROUP_REPLY_ERR_HTML_TEMPLATE
    else:
        template = __GROUP_REPLY_HTML_TEMPLATE
        data = gr.get_data()
        if isinstance(data, PyTango.DeviceAttribute):
            data = display_deviceattribute_html(data)
        fmt["data"] = data
        
    ret = template.format(**fmt)
    return ret

def init_display(ip):
    html_formatter = ip.display_formatter.formatters["text/html"]
    html_formatter.for_type(PyTango.DeviceProxy, display_deviceproxy_html)
    html_formatter.for_type(PyTango.Database, display_database_html)
    html_formatter.for_type(PyTango.DeviceAttribute, display_deviceattribute_html)
    html_formatter.for_type(PyTango.Group, display_group_html)
    html_formatter.for_type(PyTango.GroupAttrReply, display_groupreply_html)
    html_formatter.for_type(PyTango.GroupCmdReply, display_groupreply_html)

    
def init_ipython(ip=None, store=True, pytango=True, colors=True, console=True,
                 magic=True):

    if ip is None:
        ip = get_ipapi()
    
    global _tango_init
    if _tango_init is True: return

    init_display(ip)
    
    if pytango:
        init_pytango(ip)
    
    init_db()

    if magic:
        init_magic(ip)
    
    _tango_init = True

def load_config(config):
    import PyTango.ipython
    import IPython.utils.coloransi
    
    d = { "version" : str(PyTango.ipython.get_pytango_version()),
          "pyver" : str(PyTango.ipython.get_python_version()),
          "ipyver" : str(PyTango.ipython.get_ipython_version()),
          "pytangover" : str(PyTango.ipython.get_pytango_version()), }
    d.update(IPython.utils.coloransi.TermColors.__dict__)

    so = Struct(
        tango_banner="""%(Blue)shint: Try typing: mydev = Device("%(LightBlue)s<tab>%(Normal)s\n""")

    so = config.get("tango_options", so)

    ipy_ver = PyTango.ipython.get_ipython_version()
    
    # ------------------------------------
    # Application
    # ------------------------------------
    app = config.Application
    app.log_level = 30

    # ------------------------------------
    # InteractiveShell
    # ------------------------------------
    i_shell = config.InteractiveShell
    i_shell.colors = 'Linux'

    # ------------------------------------
    # PromptManager (ipython >= 0.12)
    # ------------------------------------
    prompt = config.PromptManager
    prompt.in_template = 'ITango [\\#]: '
    prompt.out_template = 'Result [\\#]: '
    
    # ------------------------------------
    # InteractiveShellApp
    # ------------------------------------
    i_shell_app = config.InteractiveShellApp
    extensions = getattr(i_shell_app, 'extensions', [])
    extensions.append('PyTango.ipython')
    i_shell_app.extensions = extensions
    
    # ------------------------------------
    # TerminalIPythonApp: options for the IPython terminal (and not Qt Console)
    # ------------------------------------
    term_app = config.TerminalIPythonApp
    term_app.display_banner = True
    #term_app.nosep = False
    #term_app.classic = True
    
    # ------------------------------------
    # IPKernelApp: options for the  Qt Console
    # ------------------------------------
    #kernel_app = config.IPKernelApp
    ipython_widget = config.IPythonWidget
    ipython_widget.in_prompt  = 'ITango [<span class="in-prompt-number">%i</span>]: '
    ipython_widget.out_prompt = 'Result [<span class="out-prompt-number">%i</span>]: '
    
    #zmq_i_shell = config.ZMQInteractiveShell
    
    # ------------------------------------
    # TerminalInteractiveShell
    # ------------------------------------
    term_i_shell = config.TerminalInteractiveShell
    banner = """\
%(Purple)sITango %(version)s%(Normal)s -- An interactive %(Purple)sTango%(Normal)s client.

Running on top of Python %(pyver)s, IPython %(ipyver)s and PyTango %(pytangover)s

help      -> ITango's help system.
object?   -> Details about 'object'. ?object also works, ?? prints more.
"""
    
    banner = banner % d
    banner = banner.format(**d)
    tango_banner = so.tango_banner % d
    tango_banner = tango_banner.format(**d)
    all_banner = "\n".join((banner, tango_banner))

    term_i_shell.banner1 = banner
    term_i_shell.banner2 = tango_banner

    # ------------------------------------
    # FrontendWidget
    # ------------------------------------
    frontend_widget = config.ITangoConsole
    frontend_widget.banner = all_banner
    
def load_ipython_extension(ipython):
    # The ``ipython`` argument is the currently active
    # :class:`InteractiveShell` instance that can be used in any way.
    # This allows you do to things like register new magics, plugins or
    # aliases.
    init_ipython(ip=ipython, store=False, colors=False)

def unload_ipython_extension(ipython):
    # If you want your extension to be unloadable, put that logic here.
    #print "Unloading PyTango IPython extension"
    pass

def run(qt=False):

    # overwrite the original IPython Qt widget with our own so we can put a
    # customized banner. IPython may have been installed without Qt support so we
    # protect this code against an import error
    try:
        from IPython.utils.traitlets import Unicode
        from IPython.qt.console.rich_ipython_widget import RichIPythonWidget

        class ITangoConsole(RichIPythonWidget):
            
            banner = Unicode(config=True)

            def _banner_default(self):
                config = get_config()
                return config.ITangoConsole.banner

        import IPython.qt.console.qtconsoleapp
        IPythonQtConsoleApp = IPython.qt.console.qtconsoleapp.IPythonQtConsoleApp
        IPythonQtConsoleApp.widget_factory = ITangoConsole      
    except ImportError:
        pass

    argv = sys.argv

    try:
        for i, arg in enumerate(argv[:1]):
            if arg.startswith('--profile='):
                break
        else:
            argv.append("--profile=tango")
    except:
        pass    
    
    if qt:
        if not 'qtconsole' in argv:
            argv.insert(1, 'qtconsole')
            argv.append('--pylab=inline')
    
    launch_new_instance()
