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

"""Server for tango objects."""

import os
import sys
import six
import logging
import weakref
import inspect
import functools

from .codec import loads, dumps
from .attr_data import AttrData
from .utils import TO_TANGO_TYPE
from ._tango import AttrDataFormat, CmdArgType, GreenMode
from ._tango import DbDevInfo, Database, DevState, constants
from .server import Device, _to_classes, _add_classes
from .server import get_worker, set_worker
from .green import get_executor

__all__ = ('Server',)

_CLEAN_UP_TEMPLATE = """
import sys
from tango import Database

db = Database()
server_instance = '{server_instance}'
try:
    devices = db.get_device_class_list(server_instance)[::2]
    for device in devices:
        db.delete_device(device)
        try:
            db.delete_device_alias(db.get_alias(device))
        except:
            pass
except:
    print ('Failed to cleanup!')
"""


def __to_tango_type_fmt(value):
    dfmt = AttrDataFormat.SCALAR
    value_t = type(value)
    dtype = TO_TANGO_TYPE.get(value_t)
    max_dim_x, max_dim_y = 1, 0
    if dtype is None:
        if constants.NUMPY_SUPPORT:
            import numpy
        else:
            numpy = None
        if numpy and isinstance(value, numpy.ndarray):
            dtype = TO_TANGO_TYPE.get(value.dtype.name)
            shape_l = len(value.shape)
            if shape_l == 1:
                dfmt = AttrDataFormat.SPECTRUM
                max_dim_x = max(2 ** 16, value.shape[0])
            elif shape_l == 2:
                dfmt = AttrDataFormat.IMAGE
                max_dim_x = max(2 ** 16, value.shape[0])
                max_dim_y = max(2 ** 16, value.shape[1])
        else:
            dtype = CmdArgType.DevEncoded
    return dtype, dfmt, max_dim_x, max_dim_y


def create_tango_class(server, obj, tango_class_name=None, member_filter=None):
    slog = server.server_instance.replace("/", ".")
    log = logging.getLogger("tango.Server." + slog)

    obj_klass = obj.__class__
    obj_klass_name = obj_klass.__name__

    if tango_class_name is None:
        tango_class_name = obj_klass_name

    class DeviceDispatcher(Device):

        TangoClassName = tango_class_name

        def __init__(self, tango_class_obj, name):
            tango_object = server.get_tango_object(name)
            self.__tango_object = weakref.ref(tango_object)
            Device.__init__(self, tango_class_obj, name)

        def init_device(self):
            Device.init_device(self)
            self.set_state(DevState.ON)

        @property
        def _tango_object(self):
            return self.__tango_object()

        @property
        def _object(self):
            return self._tango_object._object

    DeviceDispatcher.__name__ = tango_class_name
    DeviceDispatcher.TangoClassName = tango_class_name
    DeviceDispatcherClass = DeviceDispatcher.TangoClassClass

    for name in dir(obj):
        if name.startswith("_"):
            continue
        log.debug("inspecting %s.%s", obj_klass_name, name)
        try:
            member = getattr(obj, name)
        except:
            log.info(
                "failed to inspect member '%s.%s'", obj_klass_name, name)
            log.debug("Details:", exc_info=1)
        if inspect.isclass(member) or inspect.ismodule(member):
            continue
        if member_filter and not member_filter(obj, tango_class_name,
                                               name, member):
            log.debug("filtered out %s from %s", name, tango_class_name)
            continue
        if inspect.isroutine(member):
            # try to find out if there are any parameters
            in_type = CmdArgType.DevEncoded
            out_type = CmdArgType.DevEncoded
            try:
                arg_spec = inspect.getargspec(member)
                if not arg_spec.args:
                    in_type = CmdArgType.DevVoid
            except TypeError:
                pass

            if in_type == CmdArgType.DevVoid:
                def _command(dev, func_name=None):
                    obj = dev._object
                    f = getattr(obj, func_name)
                    result = server.worker.execute(f)
                    return server.dumps(result)
            else:
                def _command(dev, param, func_name=None):
                    obj = dev._object
                    args, kwargs = loads(*param)
                    f = getattr(obj, func_name)
                    result = server.worker.execute(f, *args, **kwargs)
                    return server.dumps(result)
            cmd = functools.partial(_command, func_name=name)
            cmd.__name__ = name
            doc = member.__doc__
            if doc is None:
                doc = ""
            cmd.__doc__ = doc
            cmd = six.create_unbound_method(cmd, DeviceDispatcher)
            setattr(DeviceDispatcher, name, cmd)
            DeviceDispatcherClass.cmd_list[name] = \
                [[in_type, doc], [out_type, ""]]
        else:
            read_only = False
            if hasattr(obj_klass, name):
                kmember = getattr(obj_klass, name)
                if inspect.isdatadescriptor(kmember):
                    if kmember.fset is None:
                        read_only = True
                else:
                    continue
            value = member
            dtype, fmt, x, y = __to_tango_type_fmt(value)
            if dtype is None or dtype == CmdArgType.DevEncoded:
                dtype = CmdArgType.DevEncoded
                fmt = AttrDataFormat.SCALAR

                def read(dev, attr):
                    name = attr.get_name()
                    value = server.worker.execute(getattr, dev._object, name)
                    attr.set_value(*server.dumps(value))

                def write(dev, attr):
                    name = attr.get_name()
                    value = attr.get_write_value()
                    value = loads(*value)
                    server.worker.execute(setattr, dev._object, name, value)
            else:

                def read(dev, attr):
                    name = attr.get_name()
                    value = server.worker.execute(getattr, dev._object, name)
                    attr.set_value(value)

                def write(dev, attr):
                    name = attr.get_name()
                    value = attr.get_write_value()
                    server.worker.execute(setattr, dev._object, name, value)
            read.__name__ = "_read_" + name
            setattr(DeviceDispatcher, read.__name__, read)

            pars = dict(name=name, dtype=dtype, dformat=fmt,
                        max_dim_x=x, max_dim_y=y, fget=read)
            if not read_only:
                write.__name__ = "_write_" + name
                pars['fset'] = write
                setattr(DeviceDispatcher, write.__name__, write)
            attr_data = AttrData.from_dict(pars)
            DeviceDispatcherClass.attr_list[name] = attr_data
    return DeviceDispatcher


class Server:
    """
    Server helper
    """

    Phase0, Phase1, Phase2 = range(3)
    PreInitPhase = Phase1
    PostInitPhase = Phase2

    class TangoObjectAdapter:

        def __init__(self, server, obj, full_name, alias=None,
                     tango_class_name=None):
            self.__server = weakref.ref(server)
            self.full_name = full_name
            self.alias = alias
            self.class_name = obj.__class__.__name__
            if tango_class_name is None:
                tango_class_name = self.class_name
            self.tango_class_name = tango_class_name
            self.__object = weakref.ref(obj, self.__onObjectDeleted)

        def __onObjectDeleted(self, object_weak):
            self.__object = None
            server = self._server
            server.log.info("object deleted %s(%s)", self.class_name,
                            self.full_name)
            server.unregister_object(self.full_name)

        @property
        def _server(self):
            return self.__server()

        @property
        def _object(self):
            obj = self.__object
            if obj is None:
                return None
            return obj()

    def __init__(self, server_name, server_type=None, port=None,
                 event_loop_callback=None, init_callbacks=None,
                 auto_clean=False, green_mode=None, tango_classes=None,
                 protocol="pickle"):
        if server_name is None:
            raise ValueError("Must give a valid server name")
        self.__server_name = server_name
        self.__server_type = server_type
        self.__port = port
        self.__event_loop_callback = event_loop_callback
        if init_callbacks is None:
            init_callbacks = {}
        self.__init_callbacks = init_callbacks
        self.__util = None
        self.__objects = {}
        self.__running = False
        self.__auto_clean = auto_clean
        self.__green_mode = green_mode
        self.__protocol = protocol
        self.__tango_classes = _to_classes(tango_classes or [])
        self.__tango_devices = []
        if self.async_mode:
            self.__worker = get_executor(self.green_mode)
        else:
            self.__worker = get_worker()
        set_worker(self.__worker)
        self.log = logging.getLogger("tango.Server")
        self.__phase = Server.Phase0

    def __build_args(self):
        args = [self.server_type, self.__server_name]
        if self.__port is not None:
            args.extend(["-ORBendPoint",
                         "giop:tcp::{0}".format(self.__port)])
        return args

    def __exec_cb(self, cb):
        if not cb:
            return
        self.worker.execute(cb)

    def __find_tango_class(self, key):
        pass

    def __prepare(self):
        """Update database with existing devices"""
        self.log.debug("prepare")

        if self.__phase > 0:
            raise RuntimeError("Internal error: Can only prepare in phase 0")

        server_instance = self.server_instance
        db = Database()

        # get list of server devices if server was already registered
        server_registered = server_instance in db.get_server_list()

        if server_registered:
            dserver_name = "dserver/{0}".format(server_instance)
            if db.import_device(dserver_name).exported:
                import tango
                dserver = tango.DeviceProxy(dserver_name)
                try:
                    dserver.ping()
                    raise Exception("Server already running")
                except:
                    self.log.info("Last time server was not properly "
                                  "shutdown!")
            _, db_device_map = self.get_devices()
        else:
            db_device_map = {}

        db_devices_add = {}

        # all devices that are registered in database that are not registered
        # as tango objects or for which the tango class changed will be removed
        db_devices_remove = set(db_device_map) - set(self.__objects)

        for local_name, local_object in self.__objects.items():
            local_class_name = local_object.tango_class_name
            db_class_name = db_device_map.get(local_name)
            if db_class_name:
                if local_class_name != db_class_name:
                    db_devices_remove.add(local_name)
                    db_devices_add[local_name] = local_object
            else:
                db_devices_add[local_name] = local_object

        for device in db_devices_remove:
            db.delete_device(device)
            try:
                db.delete_device_alias(db.get_alias(device))
            except:
                pass

        # register devices in database

        # add DServer
        db_dev_info = DbDevInfo()
        db_dev_info.server = server_instance
        db_dev_info._class = "DServer"
        db_dev_info.name = "dserver/" + server_instance

        db_dev_infos = [db_dev_info]
        aliases = []
        for obj_name, obj in db_devices_add.items():
            db_dev_info = DbDevInfo()
            db_dev_info.server = server_instance
            db_dev_info._class = obj.tango_class_name
            db_dev_info.name = obj.full_name
            db_dev_infos.append(db_dev_info)
            if obj.alias:
                aliases.append((obj.full_name, obj.alias))

        db.add_server(server_instance, db_dev_infos)

        # add aliases
        for alias_info in aliases:
            db.put_device_alias(*alias_info)

    def __clean_up_process(self):
        if not self.__auto_clean:
            return
        clean_up = _CLEAN_UP_TEMPLATE.format(
            server_instance=self.server_instance)
        import subprocess
        res = subprocess.call([sys.executable, "-c", clean_up])
        if res:
            self.log.error("Failed to cleanup")

    def __initialize(self):
        self.log.debug("initialize")
        async_mode = self.async_mode
        event_loop = self.__event_loop_callback

        util = self.tango_util
        u_instance = util.instance()

        if async_mode:
            if event_loop:
                event_loop = functools.partial(self.worker.execute,
                                               event_loop)
        if event_loop:
            u_instance.server_set_event_loop(event_loop)

        _add_classes(util, self.__tango_classes)

    def __run(self, timeout=None):
        return self.worker.run(self.__tango_loop, wait=True, timeout=timeout)

    def __tango_loop(self):
        self.log.debug("server loop started")
        self.__running = True
        u_instance = self.tango_util.instance()
        u_instance.server_init()
        self._phase = Server.Phase2
        self.log.info("Ready to accept request")
        u_instance.server_run()
        if self.__auto_clean:
            self.__clean_up_process()
        self.log.debug("server loop exit")

    @property
    def _phase(self):
        return self.__phase

    @_phase.setter
    def _phase(self, phase):
        self.__phase = phase
        cb = self.__init_callbacks.get(phase)
        self.__exec_cb(cb)

    @property
    def server_type(self):
        server_type = self.__server_type
        if server_type is None:
            server_file = os.path.basename(sys.argv[0])
            server_type = os.path.splitext(server_file)[0]
        return server_type

    @property
    def server_instance(self):
        return "{0}/{1}".format(self.server_type, self.__server_name)

    @property
    def tango_util(self):
        if self.__util is None:
            import tango
            self.__util = tango.Util(self.__build_args())
            self._phase = Server.Phase1
        return self.__util

    @property
    def green_mode(self):
        gm = self.__green_mode
        if gm is None:
            from tango import get_green_mode
            gm = get_green_mode()
        return gm

    @green_mode.setter
    def green_mode(self, gm):
        if gm == self.__green_mode:
            return
        if self.__running:
            raise RuntimeError("Cannot change green mode while "
                               "server is running")
        self.__green_mode = gm

    @property
    def async_mode(self):
        return self.green_mode in (GreenMode.Gevent, GreenMode.Asyncio)

    @property
    def worker(self):
        return self.__worker

    def dumps(self, obj):
        return dumps(self.__protocol, obj)

    def get_devices(self):
        """
        Helper that retuns a dict of devices for this server.

        :return:
            Returns a tuple of two elements:
              - dict<tango class name : list of device names>
              - dict<device names : tango class name>
        :rtype: tuple<dict, dict>
        """
        if self.__util is None:
            import tango
            db = tango.Database()
        else:
            db = self.__util.get_database()
        server = self.server_instance
        dev_list = db.get_device_class_list(server)
        class_map, dev_map = {}, {}
        for class_name, dev_name in zip(dev_list[1::2], dev_list[::2]):
            dev_names = class_map.get(class_name)
            if dev_names is None:
                class_map[class_name] = dev_names = []
            dev_name = dev_name.lower()
            dev_names.append(dev_name)
            dev_map[dev_name] = class_name
        return class_map, dev_map

    def get_tango_object(self, name):
        return self.__objects.get(name.lower())

    def get_tango_class(self, tango_class_name):
        for klass in self.__tango_classes:
            if klass.TangoClassName == tango_class_name:
                return klass

    def register_tango_device(self, klass, name):
        if inspect.isclass(klass):
            if isinstance(klass, Device):
                # TODO
                raise NotImplementedError
            else:
                raise ValueError
        else:
            raise NotImplementedError

    def register_tango_class(self, klass):
        if self._phase > Server.Phase1:
            raise RuntimeError("Cannot add new class after phase 1 "
                               "(i.e. after server_init)")
        self.__tango_classes.append(klass)

    def unregister_object(self, name):
        tango_object = self.__objects.pop(name.lower())
        if self._phase > Server.Phase1:
            import tango
            util = tango.Util.instance()
            if not util.is_svr_shutting_down():
                util.delete_device(tango_object.tango_class_name, name)

    def register_object(self, obj, name, tango_class_name=None,
                        member_filter=None):
        """
        :param member_filter:
            callable(obj, tango_class_name, member_name, member) -> bool
        """
        slash_count = name.count("/")
        if slash_count == 0:
            alias = name
            full_name = "{0}/{1}".format(self.server_instance, name)
        elif slash_count == 2:
            alias = None
            full_name = name
        else:
            raise ValueError("Invalid name")

        class_name = tango_class_name or obj.__class__.__name__
        tango_class = self.get_tango_class(class_name)

        if tango_class is None:
            tango_class = create_tango_class(self, obj, class_name,
                                             member_filter=member_filter)
            self.register_tango_class(tango_class)

        tango_object = self.TangoObjectAdapter(self, obj, full_name, alias,
                                               tango_class_name=class_name)
        self.__objects[full_name.lower()] = tango_object
        if self._phase > Server.Phase1:
            import tango
            util = tango.Util.instance()
            util.create_device(class_name, name)
        return tango_object

    def run(self, timeout=None):
        self.log.debug("run")
        async_mode = self.async_mode
        running = self.__running
        if not running:
            self.__prepare()
            self.__initialize()
        else:
            if not async_mode:
                raise RuntimeError("Server is already running")
        self.__run(timeout=timeout)
