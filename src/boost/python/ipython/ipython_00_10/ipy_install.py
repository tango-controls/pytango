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

import sys
import os

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
    

def install(ipydir=None, verbose=True, profile='tango'):
    install_dir = ipydir or IPython.genutils.get_ipython_dir()
    f_name = os.path.join(install_dir, 'ipy_profile_tango.py')
    if verbose:
        def out(msg):
            sys.stdout.write(msg)
            sys.stdout.flush()
    else:
        out = lambda x : None
            
    if ipydir is None and os.path.isfile(f_name):
        print("Warning: The file '%s' already exists." % f_name)
        r = ''
        while r.lower() not in ('y', 'n'):
            r = input("Do you wish to override it [Y/n]?")
            r = r or 'y'
        if r.lower() == 'n':
            return
    profile = __PROFILE.format(pytangover=PyTango.Release.version, ipyver=IPython.Release.version)
    
    out("Installing tango extension to ipython... ")
    try:
        f = open(f_name, "w")
        f.write(profile)
        f.close()
        out("[DONE]\n\n")
    except:
        out("[FAILED]\n\n")
        raise
    
    ipy_user_config = os.path.join(IPython.genutils.get_ipython_dir(), 'ipy_user_conf.py')
    out("""\
To start ipython with tango interface simply type on the command line:
%% ipython -p tango

If you want tango extension to be automaticaly active when you start ipython,
edit your {0} and add the line:
import ipy_profile_tango

Next time, just start ipython on the command line:
%% ipython

and your tango extension should be loaded automaticaly. Note that if you are also
loading other extensions that, for example, overwrite the prompt, the prompt
that will appear is the one from the last extension to be imported.

For more information goto: http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

Have fun with ITango!
The PyTango team
    """.format(ipy_user_config))

def main():
    d = None
    if len(sys.argv) > 1:
        d = sys.argv[1]
    install(d)
    
if __name__ == "__main__":
    main()
