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

__all__ = ["load_config"]

import sys
import PyTango

from IPython.utils.ipstruct import Struct
from IPython.utils.coloransi import TermColors

def __get_python_version():
    return '.'.join(map(str,sys.version_info[:3]))

def __get_ipython_version():
    """Returns the current IPython version"""
    import IPython
    v = "<Unknown>"
    try:
        v = IPython.release.version
    except Exception:
        pass
    return v

def __get_pytango_version():
    vi = PyTango.Release.version_info
    return ".".join(map(str,vi[:3]))+vi[3]


def load_config(config):
    d = { "version" : PyTango.Release.version,
          "pyver" : __get_python_version(),
          "ipyver" : __get_ipython_version(),
          "pytangover" : __get_pytango_version() }
    d.update(TermColors.__dict__)

    so = Struct(
        spock_banner="""%(Blue)shint: Try typing: mydev = Device("%(LightBlue)s<tab>%(Normal)s""")

    so = config.get("spock_options", so)

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
    #i_shell.prompt_in1 = 'Spock <$DB_NAME> [\\#]: '
    #i_shell.prompt_out = 'Result [\\#]: '
    
    # ------------------------------------
    # PromptManager
    # ------------------------------------
    prompt = config.PromptManager
    prompt.in_template = 'Spock {DB_NAME} [\\#]: '
    #prompt.in2_template = 
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
    ipython_widget.in_prompt  = 'Spock {DB_NAME} [<span class="in-prompt-number">%i</span>]: '
    ipython_widget.out_prompt = '  Out [<span class="out-prompt-number">%i</span>]: '
    
    #zmq_i_shell = config.ZMQInteractiveShell
    
    # ------------------------------------
    # TerminalInteractiveShell
    # ------------------------------------
    term_i_shell = config.TerminalInteractiveShell
    banner = """\
%(Purple)sSpock %(version)s%(Normal)s -- An interactive %(Purple)sTango%(Normal)s client.

Running on top of Python %(pyver)s, IPython %(ipyver)s and PyTango %(pytangover)s

help      -> Spock's help system.
object?   -> Details about 'object'. ?object also works, ?? prints more.
"""
    
    banner = banner % d
    banner = banner.format(**d)
    spock_banner = so.spock_banner % d
    spock_banner = spock_banner.format(**d)
    term_i_shell.banner1 = banner
    term_i_shell.banner2 = spock_banner

