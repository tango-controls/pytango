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

from IPython.utils.ipstruct import Struct
from IPython.utils.coloransi import TermColors

def load_config(config):
    import PyTango.ipython
    
    d = { "version" : PyTango.ipython.get_pytango_version(),
          "pyver" : PyTango.ipython.get_python_version(),
          "ipyver" : PyTango.ipython.get_ipython_version(),
          "pytangover" : PyTango.ipython.get_pytango_version(), }
    d.update(TermColors.__dict__)

    so = Struct(
        tango_banner="""%(Blue)shint: Try typing: mydev = Device("%(LightBlue)s<tab>%(Normal)s""")

    so = config.get("tango_options", so)

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
    # PromptManager
    # ------------------------------------
    prompt = config.PromptManager
    prompt.in_template = 'ITango [\\#]: '
    #prompt.in2_template = 
    prompt.out_template = 'Result [\\#]: '
    
    # ------------------------------------
    # InteractiveShellApp
    # ------------------------------------
    i_shell_app = config.InteractiveShellApp
    extensions = getattr(i_shell_app, 'extensions', [])
    extensions.append('PyTango.ipython')
    i_shell_app.extensions = extensions
    i_shell_app.ignore_old_config=True
    
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
    ipython_widget.out_prompt = '   Out [<span class="out-prompt-number">%i</span>]: '
    
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
    term_i_shell.banner1 = banner
    term_i_shell.banner2 = tango_banner
    
    # ------------------------------------
    # FrontendWidget
    # ------------------------------------
    frontend_widget = config.FrontendWidget
    frontend_widget.banner = banner
