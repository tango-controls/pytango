"""Provide a context to run a device without a database."""

from __future__ import absolute_import

__all__ = ["DeviceTestContext", "run_device_test_context"]

# Imports
import os
import time
import socket
import platform
import tempfile
import functools

# Concurrency imports
from threading import Thread
from multiprocessing import Process

# CLI imports
from ast import literal_eval
from importlib import import_module
from argparse import ArgumentParser

# Local imports
from .server import run
from . import DeviceProxy, Database, ConnectionFailed, DevFailed


# Helpers

def retry(period, errors, pause=0.001):
    """Retry decorator."""
    errors = tuple(errors)

    def dec(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            stop = time.time() + period
            first = True
            while first or time.time() < stop:
                time.sleep(pause)
                try:
                    return func(*args, **kwargs)
                except errors as exc:
                    e = exc
                    first = False
            raise e
        return wrapper
    return dec


def get_port():
    sock = socket.socket()
    sock.bind(('', 0))
    res = sock.getsockname()[1]
    del sock
    return res


def literal_dict(arg):
    return dict(literal_eval(arg))


def device(path):
    """Get the device class from a given module."""
    module_name, device_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, device_name)


# Device test context

class DeviceTestContext(object):
    """ Context to run a device without a database."""

    nodb = "#dbase=no"
    command = "{0} {1} -ORBendPoint giop:tcp::{2} -file={3}"
    connect_timeout = 6.0
    disconnect_timeout = connect_timeout

    def __init__(self, device, device_cls=None, server_name=None,
                 instance_name=None, device_name=None, properties={},
                 db=None, port=0, debug=3, daemon=False, process=False):
        """Inititalize the context to run a given device."""
        # Argument
        tangoclass = device.__name__
        if not server_name:
            server_name = tangoclass
        if not instance_name:
            instance_name = server_name.lower()
        if not device_name:
            device_name = 'test/nodb/' + server_name.lower()
        if not port:
            port = get_port()
        if db is None:
            _, db = tempfile.mkstemp()
        # Attributes
        self.db = db
        self.port = port
        self.device_name = device_name
        self.server_name = "/".join(("dserver", server_name, instance_name))
        self.host = "{0}:{1}/".format(platform.node(), self.port)
        self.device = self.server = None
        # File
        self.generate_db_file(server_name, instance_name, device_name,
                              tangoclass, properties)
        # Command args
        string = self.command.format(server_name, instance_name, port, db)
        string += " -v{0}".format(debug) if debug else ""
        cmd_args = string.split()
        # Target and arguments
        if device_cls:
            target = run
            args = ({tangoclass: (device_cls, device)}, cmd_args)
        elif not hasattr(device, 'run_server'):
            target = run
            args = ((device,), cmd_args)
        else:
            target = device.run_server
            args = (cmd_args,)
        # Thread
        cls = Process if process else Thread
        self.thread = cls(target=target, args=args)
        self.thread.daemon = daemon

    def generate_db_file(self, server, instance, device,
                         tangoclass=None, properties={}):
        """Generate a database file corresponding to the given arguments."""
        if not tangoclass:
            tangoclass = server
        # Open the file
        with open(self.db, 'w') as f:
            f.write("/".join((server, instance, "DEVICE", tangoclass)))
            f.write(': "' + device + '"\n')
        # Create database
        db = Database(self.db)
        # Patched the property dict to avoid a PyTango bug
        patched = dict((key, value if value != '' else ' ')
                       for key, value in properties.items())
        # Write properties
        db.put_device_property(device, patched)
        return db

    def get_device_access(self):
        """Return the full device name."""
        return self.host+self.device_name+self.nodb

    def get_server_access(self):
        """Return the full server name."""
        return self.host+self.server_name+self.nodb

    def start(self):
        """Run the server."""
        self.thread.start()
        self.connect()
        return self

    @retry(connect_timeout, [ConnectionFailed, DevFailed])
    def connect(self):
        if not self.thread.is_alive():
            raise RuntimeError(
                'The server did not start. Check stdout for more information.')
        self.device = DeviceProxy(self.get_device_access())
        self.device.ping()
        self.server = DeviceProxy(self.get_server_access())
        self.server.ping()

    def stop(self, timeout=None):
        """Kill the server."""
        if self.server:
            self.server.command_inout('Kill')
        self.join(timeout)
        os.unlink(self.db)

    def join(self, timeout=None):
        if timeout is None:
            timeout = self.disconnect_timeout
        self.thread.join(timeout)

    def __enter__(self):
        """Enter method for context support."""
        if not self.thread.is_alive():
            self.start()
        return self.device

    def __exit__(self, exc_type, exception, trace):
        """Exit method for context support."""
        self.stop()


# Command line interface

def parse_command_line_args(args=None):
    """Parse arguments given in command line."""
    desc = "Run a given device on a given port."
    parser = ArgumentParser(description=desc)
    # Add arguments
    msg = 'The device to run as a python path.'
    parser.add_argument('device', metavar='DEVICE',
                        type=device, help=msg)
    msg = "The port to use."
    parser.add_argument('--port', metavar='PORT',
                        type=int, help=msg, default=0)
    msg = "The debug level."
    parser.add_argument('--debug', metavar='DEBUG',
                        type=int, help=msg, default=0)
    msg = "The properties to set as python dict."
    parser.add_argument('--prop', metavar='PROP',
                        type=literal_dict, help=msg, default='{}')
    # Parse arguments
    namespace = parser.parse_args(args)
    return namespace.device, namespace.port, namespace.prop, namespace.debug


def run_device_test_context(args=None):
    device, port, properties, debug = parse_command_line_args(args)
    context = DeviceTestContext(
        device, properties=properties, port=port, debug=debug)
    context.start()
    msg = '{0} started on port {1} with properties {2}'
    print(msg.format(device.__name__, context.port, properties))
    print('Device access: {}'.format(context.get_device_access()))
    print('Server access: {}'.format(context.get_server_access()))
    context.join()
    print("Done")


# Main execution

if __name__ == "__main__":
    run_device_test_context()
