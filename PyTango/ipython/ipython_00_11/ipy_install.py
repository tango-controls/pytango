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

from __future__ import with_statement

import sys
import os
import io

import IPython
from IPython.core.profiledir import ProfileDirError, ProfileDir
from IPython.core.application import BaseIPythonApplication
from IPython.utils.path import get_ipython_dir
from IPython.utils.io import ask_yes_no

import PyTango

__PROFILE = """\
#!/usr/bin/env ipython
\"\"\"An automaticaly generated IPython profile designed to provide a user 
friendly interface to Tango.
Created with PyTango {pytangover} for IPython {ipyver}\"\"\"

import PyTango.ipython

config = get_config()
PyTango.ipython.load_config(config)

# Put any additional environment here
"""

def is_installed(ipydir=None, profile='tango'):
    ipython_dir = ipydir or get_ipython_dir()
    try:
        p_dir = ProfileDir.find_profile_dir_by_name(ipython_dir, profile)
    except ProfileDirError:
        return False
    config_file_name = BaseIPythonApplication.config_file_name.default_value
    abs_config_file_name = os.path.join(p_dir.location, config_file_name)
    return os.path.isfile(abs_config_file_name)

def install(ipydir=None, verbose=True, profile='tango'):
    if verbose:
        out = sys.stdout
    else:
        out = io.StringIO()
    
    ipython_dir = ipydir or get_ipython_dir()
    try:
        p_dir = ProfileDir.find_profile_dir_by_name(ipython_dir, profile)
    except ProfileDirError:
        p_dir = ProfileDir.create_profile_dir_by_name(ipython_dir, profile)
    
    config_file_name = BaseIPythonApplication.config_file_name.default_value
    abs_config_file_name = os.path.join(p_dir.location, config_file_name)
    create_config = True
    if os.path.isfile(abs_config_file_name):
        create_config = ask_yes_no("Tango configuration file already exists. "\
                                   "Do you wish to replace it?", default='y')
    
    if not create_config:
        return

    out.write(u"Installing tango extension to ipython... ")
    out.flush()

    profile = __PROFILE.format(pytangover=PyTango.Release.version,
                               ipyver=IPython.release.version)
    with open(abs_config_file_name, "w") as f:
        f.write(profile)
        f.close()
    out.write(u"[DONE]\n\n")
    out.write(u"""\
To start ipython with tango interface simply type on the command line:
%% ipython --profile=tango

For more information goto:
http://www.tango-controls.org/static/PyTango/latest/doc/html/

Have fun with ITango!
The PyTango team
""")
    
def main():
    d = None
    if len(sys.argv) > 1:
        d = sys.argv[1]
    install(d)
    
if __name__ == "__main__":
    main()
