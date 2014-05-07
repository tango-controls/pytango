#!/usr/bin/env python

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

_CONFIG_FILE_NAME = 'ipython_config.py'

def is_installed(ipydir=None, profile='tango'):
    ipython_dir = ipydir or get_ipython_dir()
    try:
        p_dir = ProfileDir.find_profile_dir_by_name(ipython_dir, profile)
    except ProfileDirError:
        return False
    abs_config_file_name = os.path.join(p_dir.location, _CONFIG_FILE_NAME)
    return os.path.isfile(abs_config_file_name)

def install(ipydir=None, verbose=True, profile='tango'):
    if verbose:
        def out(msg):
            sys.stdout.write(msg)
            sys.stdout.flush()
    else:
        out = lambda x : None
            
    ipython_dir = ipydir or get_ipython_dir()
    try:
        p_dir = ProfileDir.find_profile_dir_by_name(ipython_dir, profile)
    except ProfileDirError:
        p_dir = ProfileDir.create_profile_dir_by_name(ipython_dir, profile)
    abs_config_file_name = os.path.join(p_dir.location, _CONFIG_FILE_NAME)
    create_config = True
    if os.path.isfile(abs_config_file_name):
        create_config = ask_yes_no("Tango configuration file already exists. "\
                                   "Do you wish to replace it?", default='y')
    
    if not create_config:
        return

    out("Installing tango extension to ipython... ")

    profile = __PROFILE.format(pytangover=PyTango.Release.version,
                               ipyver=IPython.release.version)
    with open(abs_config_file_name, "w") as f:
        f.write(profile)
        f.close()
    out("[DONE]\n\n")
    out("""\
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
