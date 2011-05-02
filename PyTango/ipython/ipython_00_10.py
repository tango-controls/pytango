#!/usr/bin/env python

################################################################################
##
## This file is part of PyTango, a python binding for Tango
## 
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
## 
## PyTango is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## PyTango is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################

"""An IPython profile designed to provide a user friendly interface to Tango"""

import sys
import os
import re
import StringIO
import textwrap
import IPython.ipapi
import IPython.ColorANSI
import IPython.Prompts
import IPython.PyColorize
import IPython.excolors
import IPython.ipstruct
import IPython.genutils
import PyTango
import PyTango.utils

_DB_SYMB = "db"
_DFT_TANGO_HOST = None
_SPOCK_STORE = "__spock_store"
_SPOCK_ERR = "__spock_error"
_spock_init = False

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
    ret = db._db_cache.devices.keys()
    ret.extend(db._db_cache.aliases.keys())
    return ret

def __DeviceClass_completer(ip, evt):
    db = __get_db()
    if db is None: return
    return db._db_cache.klasses.keys()

def __DeviceAlias_completer(ip, evt):
    db = __get_db()
    if db is None: return
    return db._db_cache.aliases.keys()

def __AttributeAlias_completer(ip, evt):
    db = __get_db()
    if db is None: return
    return db._db_cache.attr_aliases.keys()

def __PureDeviceProxy_completer(ip, evt):
    db = __get_db()
    if db is None: return
    return db._db_cache.devices.keys()

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
        ret = cache.get("attr_aliases").keys()
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

def __switchdb_completer(ip, evt):
    return __get_store(ip, "database_list").keys()

__monitor_completer = __AttributeProxy_completer

#-------------------------------------------------------------------------------
# Magic commands
#-------------------------------------------------------------------------------

def magic_refreshdb(self, parameter_s=''):
    init_db(IPython.ipapi.get(), parameter_s)

def magic_switchdb(self, parameter_s=''):
    """Switches the active tango Database.
    
    Usage: switchdb <host>[(:| )<port>]

    <port> is optional. If not given it defaults to 10000.
    
    Examples:
    In [1]: switchdb homer:10005
    In [2]: switchdb homer 10005
    In [3]: switchdb homer"""
    
    if parameter_s == '':
        raise IPython.ipapi.UsageError("%switchdb: Must specify a tango database name. See '%switchdb?'")
    return init_db(IPython.ipapi.get(), parameter_s)

def magic_lsdev(self, parameter_s=''):
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
        print "You are not connected to any Tango Database. Device list is empty"
        return
    data = db._db_cache.devices

    s = StringIO.StringIO()
    cols = 40, 25, 25, 20
    l = "%{0}s %{1}s %{2}s %{3}s".format(*cols)
    print >>s, l % ("Device", "Alias", "Server", "Class")
    print >>s, l % (cols[0]*"-", cols[1]*"-", cols[2]*"-", cols[3]*"-")
    for d, v in data.items():
        if reg_exp and not reg_exp.match(d): continue
        print >>s, l % (d, v[0], v[1], v[2])
    s.seek(0)
    IPython.genutils.page(s.read())

def magic_lsdevclass(self, parameter_s=''):
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
        print "You are not connected to any Tango Database. Device class list is empty"
        return
    data = db._db_cache.klasses

    s = StringIO.StringIO()
    data = [ "%-030s" % klass for klass in data.keys() if not reg_exp or reg_exp.match(klass) ]
    s = textwrap.fill(" ".join(data), 80)
    IPython.genutils.page(s)

def magic_lsserv(self, parameter_s=''):
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
        print "You are not connected to any Tango Database. Device class list is empty"
        return
    data = db._db_cache.servers

    s = StringIO.StringIO()
    data = [ "%-030s" % server for server in data.keys() if not reg_exp or reg_exp.match(server) ]
    s = textwrap.fill(" ".join(data), 80)
    IPython.genutils.page(s)

def magic_tango_error(self, parameter_s=''):
    """Displays detailed information about the last tango error"""
    
    global _SPOCK_ERR
    err_info = self.user_ns.get(_SPOCK_ERR)
    if err_info is None:
        print "No tango error reported so far."
        return
    print "Last tango error:"
    print err_info[1]
    
_EVT_LOG = None
def __get_event_log():
    global _EVT_LOG
    if _EVT_LOG is None:
        qthreads = IPython.ipapi.get().options.q4thread
        if qthreads:
            import ipy_qt
            model = ipy_qt.EventLoggerTableModel(capacity=10000)
            _EVT_LOG = ipy_qt.EventLogger(model=model)
            _EVT_LOG.setWindowTitle("Spock - Event Logger Table")
        else:
            import ipy_cli
            _EVT_LOG = ipy_cli.EventLogger(capacity=10000)
    return _EVT_LOG

def magic_mon(self, parameter_s=''):
    """Monitor a given attribute.
    
    %mon -a <attribute name>           - activates monitoring of given attribute
    %mon -d <attribute name>           - deactivates monitoring of given attribute
    %mon -r                            - deactivates monitoring of all attributes
    %mon -i <id>                       - displays detailed information for the event with given id
    %mon -l <dev filter> <attr filter> - shows event table filtered with the regular expression for attribute name
    %mon                               - shows event table (= %mon -i .* .*)"""
    
    db = __get_db()
    if db is None:
        print "You are not connected to any Tango Database."
        return
    opts, args = self.parse_options(parameter_s,'adril', mode='list')
    if len(args) > 3:
        raise IPython.ipapi.UsageError("%mon: too many arguments")
    if opts.has_key('d'):
        try:
            todel = args[0]
        except IndexError:
            raise IPython.ipapi.UsageError(
                "%mon -d: must provide an attribute to unmonitor")
        else:
            try:
                dev, sep, attr = todel.rpartition("/")
                subscriptions = __get_device_subscriptions(dev)
                id = subscriptions[attr.lower()]
                del subscriptions[attr.lower()]
                d = __get_device_proxy(dev)
                d.unsubscribe_event(id)
                print "Stopped monitoring '%s'" % todel
            except KeyError:
                raise IPython.ipapi.UsageError(
                    "%%mon -d: Not monitoring '%s'" % todel)
                    
    elif opts.has_key('a'):
        try:
            toadd = args[0]
        except IndexError:
            raise IPython.ipapi.UsageError(
                "%mon -a: must provide an attribute to monitor")
        dev, sep, attr = toadd.rpartition("/")
        subscriptions = __get_device_subscriptions(dev)
        id = subscriptions.get(attr.lower())
        if id is not None:
            raise IPython.ipapi.UsageError(
                "%%mon -a: Already monitoring '%s'" % toadd)
        d = __get_device_proxy(dev)
        w = __get_event_log()
        model = w.model()
        id = d.subscribe_event(attr, PyTango.EventType.CHANGE_EVENT, model, [])
        subscriptions[attr.lower()] = id
        print "'%s' is now being monitored. Type 'mon' to see all events" % toadd
    elif opts.has_key('r'):
        for d, v in db._db_cache.devices.items():
            d, subs = v[3], v[4]
            for id in subs.values():
                d.unsubscribe_event(id)
            v[4] = {}
    elif opts.has_key('i'):
        try:
            evtid = int(args[0])
        except IndexError:
            raise IPython.ipapi.UsageError(
                "%mon -i: must provide an event ID")
        except ValueError:
            raise IPython.ipapi.UsageError(
                "%mon -i: must provide a valid event ID")
        try:
            w = __get_event_log()
            e = w.getEvents()[evtid]
            if e.err:
                print str(PyTango.DevFailed(*e.errors))
            else:
                print str(e)
        except IndexError:
            raise IPython.ipapi.UsageError(
                "%mon -i: must provide a valid event ID")
    elif opts.has_key('l'):
        try:
            dexpr = args[0]
            aexpr = args[1]
        except IndexError:
            raise IPython.ipapi.UsageError(
                "%mon -l: must provide valid device and attribute filters")
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
          (this last element is for internal spock usage only. If you need a 
           DeviceProxy to this device, create your own)"""
    db = __get_db()
    if db is None:
        print "You are not connected to any Tango Database."
        return
    return db._db_cache.devices

def get_server_map():
    """Returns a dictionary where keys are server names (<name>/<instance>)
    and value is a sequence of device names that belong to the server"""
    db = __get_db()
    if db is None:
        print "You are not connected to any Tango Database."
        return
    return db._db_cache.servers

def get_class_map():
    """Returns a dictionary where keys are the tango classes and value is a 
    sequence of device names that belong to the tango class"""
    db = __get_db()
    if db is None:
        print "You are not connected to any Tango Database."
        return
    return db._db_cache.klasses

def get_alias_map():
    """Returns a dictionary where keys are the tango device aliases and value 
    is a the tango device name"""
    db = __get_db()
    if db is None:
        print "You are not connected to any Tango Database."
        return
    return db._db_cache.aliases

def get_device_list():
    """Returns a case insensitive list of device names for the current 
    database"""
    db = __get_db()
    if db is None:
        print "You are not connected to any Tango Database."
        return
    return db._db_cache.device_list

def get_alias_list():
    """Returns a case insensitive list of device aliases for the current 
    database"""
    db = __get_db()
    if db is None:
        print "You are not connected to any Tango Database."
        return
    return db._db_cache.alias_list    
    
#-------------------------------------------------------------------------------
# Private helper methods
#-------------------------------------------------------------------------------

def __tango_exc_handler(ip, etype, value, tb):
    global _SPOCK_ERR
    ip.user_ns[_SPOCK_ERR] = etype, value, tb
    if etype == PyTango.DevFailed:
        if len(value.args):
            v = value[0]
            print v.reason,":",v.desc
        else:
            print "Empty DevFailed"
        print "For more detailed information type: tango_error"
        
def __safe_tango_exec(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except PyTango.DevFailed, df:
        print df[0].reason,":",df[0].desc
        print "For more information type: get_last_tango_error"

def __get_default_tango_host():
    global _DFT_TANGO_HOST
    if _DFT_TANGO_HOST is None:
        _DFT_TANGO_HOST = os.environ.get("TANGO_HOST")
        if _DFT_TANGO_HOST is None:
            # ok, it must have been defined in the tangorc way. Since the
            # method Tango::Connection::get_env_var is protected we do a hack to
            # get the tango_host: Create a temporary Database object. It is not
            #very nice but is done only once in the lifetime of the application 
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
    
    ip = IPython.ipapi.get()
    global _DB_SYMB
    db = ip.user_ns.get(_DB_SYMB)
    
    if host_port is None:
        if db is None:
            host_port = __get_default_tango_host()
    elif host_port == '':
        host_port = __get_default_tango_host()
    else:
        host_port = host_port.strip().replace(" ",":")
        if host_port.count(":") == 0:
            host_port += ":10000"
        
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
            
            ip.user_ns["DB_NAME"] = host_port
        except Exception:
            print
            if db:
                print "Could not access Database", host_port
                old_host_port = "%s:%s" % (db.get_db_host(), db.get_db_port())
                print "Maintaining connection to Database", old_host_port
                ip.user_ns["DB_NAME"] = old_host_port
            else:
                print "Could not access any Database."
                print "Make sure .tangorc, /etc/tangorc or TANGO_HOST environment is defined."
                ip.user_ns["DB_NAME"] = "OFFLINE"
                
        # register the 'db' in the user namespace
        ip.to_user_ns({ _DB_SYMB : db })
        
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
        except Exception, e:
            print
            print "An unexpected exception ocorred during Spock command completer."
            print "Please send a bug report to the PyTango team with the following informantion:"
            print IPython.ipapi.get().options.banner
            print 80*"-"
            print "Completer:",__get_obj_name(f)
            print 80*"-"
            print str(e)
            print 80*"-"
            raise e
    return wrapper

def __get_python_version():
    return '.'.join(map(str,sys.version_info[:3]))

def __get_ipython_version():
    """Returns the current IPython version"""
    v = None
    try:
        try:
            v = IPython.Release.version
        except Exception:
            try:
                v = IPython.release.version
            except Exception:
                pass
    except Exception:
        pass
    return v

def __get_pytango_version():
    vi = PyTango.Release.version_info
    return ".".join(map(str,vi[:3]))+vi[3]

def __get_ipapi():
    return IPython.ipapi.get()

def __expose_magic(ip, name, fn, completer_func=None):
    ip.expose_magic(name, fn)
    
    if completer_func is None:
        return
    
    # enable macro param completion
    ip.set_hook('complete_command', completer_func, re_key = ".*" + name)

def __unexpose_magic(ip, name):
    mg = 'magic_%s' % name
    delattr(ip.IP, mg)

def __build_color_scheme(ip, name):
    
    # make some schemes as instances so we can copy them for modification easily:
    ColorANSI = IPython.ColorANSI
    InputColors = ColorANSI.InputTermColors
    TermColors = ColorANSI.TermColors
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
    promptTangoColors.colors.in_prompt  = InputColors.Purple
    promptTangoColors.colors.in_number  = InputColors.LightPurple
    promptTangoColors.colors.in_prompt2 = InputColors.Purple
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

def __set_store(ip, key=None, value=None):
    if key is not None:
        spock_store = ip.user_ns.get(_SPOCK_STORE)
        spock_store[key] = value
    __store(ip, _SPOCK_STORE)

def __get_store(ip, key, nvalue=None):
    spock_store = ip.user_ns.get(_SPOCK_STORE)
    v = spock_store.get(key)
    if v is None and nvalue is not None:
        spock_store[key] = nvalue
        v = nvalue
    return v

def __store(ip, var):
    # this executes the magic command store which prints a lot of info. So, first
    # we hide the standard output 
    stdout = sys.stdout
    try:
        sys.stdout = StringIO.StringIO()
        ip.magic("store %s" % var)
    finally:
        sys.stdout = stdout
        
#-------------------------------------------------------------------------------
# Initialization methods
#-------------------------------------------------------------------------------

def init_colors(ip):
    ColorANSI = IPython.ColorANSI
    InputColors = ColorANSI.InputTermColors
    TermColors = ColorANSI.TermColors
    
    name = "Tango"
    scheme = __build_color_scheme(ip, name)
    for k, v in scheme.items():
        v[0].add_scheme(v[1])

    name = "PurpleTango"
    scheme = __build_color_scheme(ip, name)
    for k, v in scheme.items():
        v[0].add_scheme(v[1])

    name = "BlueTango"
    scheme = __build_color_scheme(ip, name)
    prompt = scheme["prompt"][1]
    prompt.colors.in_prompt  = InputColors.Blue
    prompt.colors.in_number  = InputColors.LightBlue
    prompt.colors.in_prompt2 = InputColors.Blue
    prompt.colors.out_prompt = TermColors.Cyan
    prompt.colors.out_number = TermColors.LightCyan
    for k, v in scheme.items():
        v[0].add_scheme(v[1])

    name = "GreenTango"
    scheme = __build_color_scheme(ip, name)
    prompt = scheme["prompt"][1]
    prompt.colors.in_prompt  = InputColors.Green
    prompt.colors.in_number  = InputColors.LightGreen
    prompt.colors.in_prompt2 = InputColors.Green
    prompt.colors.out_prompt = TermColors.Red
    prompt.colors.out_number = TermColors.LightRed
    for k, v in scheme.items():
        v[0].add_scheme(v[1])

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
    
    ip
    ip.set_custom_exc((PyTango.DevFailed,), __tango_exc_handler)

def init_db(ip, parameter_s=''):
    global _DB_SYMB
    old_db = ip.user_ns.get(_DB_SYMB)
    
    db = __get_db(parameter_s)
    
    if old_db is not None and hasattr(old_db, "_db_cache"):
        old_junk = old_db._db_cache["junk"].keys()
        for e in old_junk:
            del ip.user_ns[e]
    else:
        old_junk = ()
        
    if db is None: return
    
    os.environ["TANGO_HOST"] = "%s:%s" % (db.get_db_host(), db.get_db_port())
    
    # Initialize device and server information
    query = "SELECT name, alias, server, class FROM device order by name"
    
    r = db.command_inout("DbMySqlSelect", query)
    row_nb, column_nb = r[0][-2], r[0][-1]
    results, data = r[0][:-2], r[1]
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
        if klass in excluded_klasses: continue
        if not ip.user_ns.has_key(klass) or klass in old_junk:
            c = DeviceClassCompleter(klass, devices)
            ip.set_hook('complete_command', c, re_key = ".*" + klass + "[^\w\.]+")
            exposed_klasses[klass] = PyTango.DeviceProxy
        else:
            print "Failed to add completer for DeviceClass",klass
    
    # expose classes no user namespace
    ip.to_user_ns(exposed_klasses)
    
    # Initialize attribute information
    query = "SELECT name, alias FROM attribute_alias order by alias"

    r = db.command_inout("DbMySqlSelect", query)
    row_nb, column_nb = r[0][-2], r[0][-1]
    results, data = r[0][:-2], r[1]
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
    db_cache = IPython.ipstruct.Struct(devices=dev_dict, aliases=alias_dict,
                                       servers=serv_dict, klasses=klass_dict,
                                       junk=exposed_klasses,
                                       attr_aliases=attr_alias_dict,
                                       device_list=device_list,
                                       alias_list=alias_list,
                                       attr_alias_list=attr_alias_list)
    
    db._db_cache = db_cache

    # Add this DB to the list of known DBs (for possible use in magic commands)
    valid_dbs = __get_store(ip, "database_list", {})
    if db.get_db_port_num() == 10000:
        db_name = db.get_db_host()
    else:
        db_name = "%s:%s" % (db.get_db_host(), db.get_db_port())
    valid_dbs[db_name] = None
    __set_store(ip)
    
    return db

def init_store(ip):
    # recover the environment
    ip.magic("store -r")
    spock_store = ip.user_ns.get(_SPOCK_STORE)
    
    if spock_store is None:
        print "Initializing spock store (should only happen once)"
        spock_store = {}
        ip.to_user_ns( { _SPOCK_STORE : spock_store} )
        __store(ip, _SPOCK_STORE)
        
def init_console(ip):
    
    TermColors = IPython.ColorANSI.TermColors
    
    d = { "version" : PyTango.Release.version,
          "pyver" : __get_python_version(),
          "ipyver" : __get_ipython_version(),
          "pytangover" : __get_pytango_version() }
    d.update(TermColors.__dict__)

    o = ip.options

    so = IPython.ipstruct.Struct(
        spock_banner = """%(Blue)shint: Try typing: mydev = Device("%(LightBlue)s<tab>%(Normal)s\n""")

    so = ip.user_ns.get("spock_options", so)
    
    o.colors = "Tango"
    o.prompt_in1 = "Spock <$DB_NAME> [\\#]: "
    o.prompt_out = "Result [\\#]: "
    banner = """
%(Purple)sSpock %(version)s%(Normal)s -- An interactive %(Purple)sTango%(Normal)s client.

Running on top of Python %(pyver)s, IPython %(ipyver)s and PyTango %(pytangover)s

help      -> Spock's help system.
object?   -> Details about 'object'. ?object also works, ?? prints more.

""" + so.spock_banner
    o.banner = banner % d
    if hasattr(o.banner, "format"):
        o.banner = o.banner.format(**d)
    
def init_magic(ip):
    __expose_magic(ip, "refreshdb", magic_refreshdb)
    __expose_magic(ip, "reloaddb", magic_refreshdb)
    __expose_magic(ip, "switchdb", magic_switchdb, __switchdb_completer)
    __expose_magic(ip, "lsdev", magic_lsdev)
    __expose_magic(ip, "lsdevclass", magic_lsdevclass)
    __expose_magic(ip, "lsserv", magic_lsserv)
    __expose_magic(ip, "tango_error", magic_tango_error)
    __expose_magic(ip, "mon", magic_mon, __monitor_completer)
    #__expose_magic(ip, "umon", magic_umon, __monitor_completer)
    
    ip.to_user_ns({"get_device_map"   : get_device_map,
                   "get_server_map"  : get_server_map,
                   "get_class_map"   : get_class_map,
                   "get_alias_map"   : get_alias_map,
                   "get_device_list" : get_device_list,
                   "get_alias_list"  : get_alias_list})
    
    #__expose_magic(ip, "get_device_map", get_device_map)
    #__expose_magic(ip, "get_server_map", get_server_map)
    #__expose_magic(ip, "get_class_map", get_class_map)
    #__expose_magic(ip, "get_alias_map", get_alias_map)
    #__expose_magic(ip, "get_device_list", get_device_list)
    #__expose_magic(ip, "get_alias_list", get_alias_list)

def complete(text):
    """a super complete!!!!"""
    self = IPython.ipapi.get().IP
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

def init_ipython(ip, store=True, pytango=True, colors=True, console=True, magic=True):
    
    if ip is None:
        raise Exception("Spock's init_ipython must be called from inside IPython")
    
    global _spock_init
    if _spock_init is True: return
    
    #ip.IP._orig_complete = ip.IP.complete
    #ip.IP.complete = complete
    
    if colors:  init_colors(ip)
    if store:   init_store(ip)
    if pytango: init_pytango(ip)
    init_db(ip)
    if console: init_console(ip)
    if magic:   init_magic(ip)
    
    _spock_init = True
