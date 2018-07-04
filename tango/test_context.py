"""Provide a context to run a device without a database."""

from __future__ import absolute_import

# Imports
import os
import sys
import six
import time
import struct
import socket
import tempfile
import traceback
import collections
from functools import partial

# Concurrency imports
import threading
import multiprocessing
from six.moves import queue

# CLI imports
from ast import literal_eval
from importlib import import_module
from argparse import ArgumentParser, ArgumentTypeError

# Local imports
from .server import run
from . import DeviceProxy, Database, Util

__all__ = ("DeviceTestContext", "run_device_test_context")

# Helpers

IOR = collections.namedtuple(
    'IOR',
    'first dtype_length dtype nb_profile tag '
    'length major minor wtf host_length host port body')


def ascii_to_bytes(s):
    convert = lambda x: six.int2byte(int(x, 16))
    return b''.join(convert(s[i:i + 2]) for i in range(0, len(s), 2))


def parse_ior(encoded_ior):
    assert encoded_ior[:4] == 'IOR:'
    ior = ascii_to_bytes(encoded_ior[4:])
    dtype_length = struct.unpack_from('II', ior)[-1]
    form = 'II{:d}sIIIBBHI'.format(dtype_length)
    host_length = struct.unpack_from(form, ior)[-1]
    form = 'II{:d}sIIIBBHI{:d}sH0I'.format(dtype_length, host_length)
    values = struct.unpack_from(form, ior)
    values += (ior[struct.calcsize(form):],)
    strip = lambda x: x[:-1] if isinstance(x, bytes) else x
    return IOR(*map(strip, values))


def get_server_host_port():
    util = Util.instance()
    ds = util.get_dserver_device()
    encoded_ior = util.get_dserver_ior(ds)
    ior = parse_ior(encoded_ior)
    return ior.host.decode(), ior.port


def literal_dict(arg):
    return dict(literal_eval(arg))


def device(path):
    """Get the device class from a given module."""
    module_name, device_name = path.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except Exception:
        raise ArgumentTypeError("Error importing {0}.{1}:\n{2}"
                                .format(module_name, device_name,
                                        traceback.format_exc()))
    return getattr(module, device_name)


def get_hostname():
    """Get the hostname corresponding to the primary, external IP.

    This is useful because an explicit hostname is required to get
    tango events to work properly. Note that localhost does not work
    either.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Connecting to a UDP address doesn't send packets
    s.connect(('8.8.8.8', 0))
    # Get ip address
    ip = s.getsockname()[0]
    # Return host name
    return socket.gethostbyaddr(ip)[0].split('.')[0]


# Device test context

class DeviceTestContext(object):
    """ Context to run a device without a database."""

    nodb = "dbase=no"
    command = "{0} {1} -ORBendPoint giop:tcp:{2}:{3} -file={4}"

    thread_timeout = 3.
    process_timeout = 5.

    def __init__(self, device, device_cls=None, server_name=None,
                 instance_name=None, device_name=None, properties=None,
                 db=None, host=None, port=0, debug=3,
                 process=False, daemon=False, timeout=None):
        """Inititalize the context to run a given device."""
        # Argument
        tangoclass = device.__name__
        if not server_name:
            server_name = tangoclass
        if not instance_name:
            instance_name = server_name.lower()
        if not device_name:
            device_name = 'test/nodb/' + server_name.lower()
        if db is None:
            _, db = tempfile.mkstemp()
        if host is None:
            host = get_hostname()
        if properties is None:
            properties = {}
        if timeout is None:
            timeout = self.process_timeout if process else self.thread_timeout
        # Patch bug #819
        if process:
            os.environ['ORBscanGranularity'] = '0'
        # Attributes
        self.db = db
        self.host = host
        self.port = port
        self.timeout = timeout
        self.device_name = device_name
        self.server_name = "/".join(("dserver", server_name, instance_name))
        self.device = self.server = None
        self.queue = multiprocessing.Queue() if process else queue.Queue()
        # File
        self.generate_db_file(server_name, instance_name, device_name,
                              tangoclass, properties)
        # Command args
        string = self.command.format(
            server_name, instance_name, host, port, db)
        string += " -v{0}".format(debug) if debug else ""
        cmd_args = string.split()
        # Target and arguments
        if device_cls:
            class_dct = {tangoclass: (device_cls, device)}
            runserver = partial(run, class_dct, cmd_args)
        elif not hasattr(device, 'run_server'):
            runserver = partial(run, (device,), cmd_args)
        else:
            runserver = partial(device.run_server, cmd_args)
        # Thread
        cls = multiprocessing.Process if process else threading.Thread
        self.thread = cls(target=self.target, args=(runserver, process))
        self.thread.daemon = daemon

    def target(self, runserver, process=False):
        try:
            runserver(post_init_callback=self.post_init, raises=True)
        except Exception:
            # Put exception in the queue
            etype, value, tb = sys.exc_info()
            if process:
                tb = None  # Traceback objects can't be pickled
            self.queue.put((etype, value, tb))
        finally:
            # Put something in the queue just in case
            exc = RuntimeError("The server failed to report anything")
            self.queue.put((None, exc, None))
            # Make sure the process has enough time to send the items
            # because the it might segfault while cleaning up the
            # the tango resources
            if process:
                time.sleep(0.01)

    def post_init(self):
        try:
            host, port = get_server_host_port()
            self.queue.put((host, port))
        except Exception as exc:
            self.queue.put((None, exc, None))
        finally:
            # Put something in the queue just in case
            exc = RuntimeError(
                "The post_init routine failed to report anything")
            self.queue.put((None, exc, None))

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
        # Patch the property dict to avoid a PyTango bug
        patched = dict((key, value if value != '' else ' ')
                       for key, value in properties.items())
        # Write properties
        db.put_device_property(device, patched)
        return db

    def get_device_access(self):
        """Return the full device name."""
        form = 'tango://{0}:{1}/{2}#{3}'
        return form.format(self.host, self.port, self.device_name, self.nodb)

    def get_server_access(self):
        """Return the full server name."""
        form = 'tango://{0}:{1}/{2}#{3}'
        return form.format(self.host, self.port, self.server_name, self.nodb)

    def start(self):
        """Run the server."""
        self.thread.start()
        self.connect()
        return self

    def connect(self):
        try:
            args = self.queue.get(
                timeout=self.timeout)
        except queue.Empty:
            if self.thread.is_alive():
                raise RuntimeError(
                    'The server appears to be stuck at initialization. '
                    'Check stdout/stderr for more information.')
            elif hasattr(self.thread, 'exitcode'):
                raise RuntimeError(
                    'The server process stopped with exitcode {}. '
                    'Check stdout/stderr for more information.'
                    ''.format(self.thread.exitcode))
            else:
                raise RuntimeError(
                    'The server stopped without reporting. '
                    'Check stdout/stderr for more information.')
        try:
            self.host, self.port = args
        except ValueError:
            six.reraise(*args)
        # Get server proxy
        self.server = DeviceProxy(self.get_server_access())
        self.server.ping()
        # Get device proxy
        self.device = DeviceProxy(self.get_device_access())
        self.device.ping()

    def stop(self):
        """Kill the server."""
        try:
            if self.server:
                self.server.command_inout('Kill')
            self.join(self.timeout)
        finally:
            os.unlink(self.db)

    def join(self, timeout=None):
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
    msg = "The hostname to use."
    parser.add_argument('--host', metavar='HOST',
                        type=str, help=msg, default=None)
    msg = "The port to use."
    parser.add_argument('--port', metavar='PORT',
                        type=int, help=msg, default=8888)
    msg = "The debug level."
    parser.add_argument('--debug', metavar='DEBUG',
                        type=int, help=msg, default=3)
    msg = "The properties to set as python dict."
    parser.add_argument('--prop', metavar='PROP',
                        type=literal_dict, help=msg, default='{}')
    # Parse arguments
    namespace = parser.parse_args(args)
    return (namespace.device, namespace.host, namespace.port,
            namespace.prop, namespace.debug)


def run_device_test_context(args=None):
    device, host, port, properties, debug = parse_command_line_args(args)
    context = DeviceTestContext(
        device, properties=properties, host=host, port=port, debug=debug)
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
