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

import sys
import os
import io

import IPython.genutils
import PyTango

__PROFILE = """\
#!/usr/bin/env ipython
\"\"\"An automaticaly generated IPython profile designed to provide a user 
friendly interface to Tango.
Created with PyTango {pytangover} for IPython {ipyver}\"\"\"

import IPython
import PyTango.ipython

ip = IPython.ipapi.get()
PyTango.ipython.init_ipython(ip)

"""

def is_installed(ipydir=None):
    install_dir = ipydir or IPython.genutils.get_ipython_dir()
    f_name = os.path.join(install_dir, 'ipy_profile_tango.py')
    return os.path.isfile(f_name)
    

def install(ipydir=None,verbose=True):
    install_dir = ipydir or IPython.genutils.get_ipython_dir()
    f_name = os.path.join(install_dir, 'ipy_profile_tango.py')
    if verbose:
        out = sys.stdout
    else:
        out = io.StringIO()
    if ipydir is None and os.path.isfile(f_name):
        print("Warning: The file '%s' already exists." % f_name)
        r = ''
        while r.lower() not in ('y', 'n'):
            r = input("Do you wish to override it [Y/n]?")
            r = r or 'y'
        if r.lower() == 'n':
            return
    profile = __PROFILE.format(pytangover=PyTango.Release.version, ipyver=IPython.Release.version)
    
    out.write(u"Installing tango extension to ipython... ")
    out.flush()
    try:
        f = open(f_name, "w")
        f.write(profile)
        f.close()
        out.write(u"[DONE]\n\n")
    except:
        out.write(u"[FAILED]\n\n")
        raise
    
    ipy_user_config = os.path.join(IPython.genutils.get_ipython_dir(), 'ipy_user_conf.py')
    out.write("""\
To start ipython with tango interface simply type on the command line:
%% ipython -p tango

If you want tango extension to be automaticaly active when you start ipython,
edit your %s and add the line:
import ipy_profile_tango

Next time, just start ipython on the command line:
%% ipython

and your tango extension should be loaded automaticaly. Note that if you are also
loading other extensions that, for example, overwrite the prompt, the prompt
that will appear is the one from the last extension to be imported.

For more information goto: http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

Have fun with ITango!
The PyTango team
    """ % (ipy_user_config,))

def main():
    d = None
    if len(sys.argv) > 1:
        d = sys.argv[1]
    install(d)
    
if __name__ == "__main__":
    main()
