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

################################################################################
# WARNING: This script should only be executed as a Post-Build Event from inside
#          Microsoft Visual Studio and not from the command line
################################################################################


import sys
import os
import os.path as osp

executable = sys.executable

curr_dir = os.getcwd()

winsetup_dir = osp.dirname(osp.abspath(__file__))
os.chdir(winsetup_dir)
setup_name = "setup.py"
bitmap = osp.join(winsetup_dir, 'doc', 'logo-medium.bmp')
ver = ".".join(map(str, sys.version_info[:2]))

try:
    cmd_line = '%s %s build_py --force --no-compile' % (executable, setup_name)
    os.system(cmd_line)

    cmd_line = '%s %s bdist_msi --skip-build --target-version=%s' \
               % (executable, setup_name, ver)
    os.system(cmd_line)

    cmd_line = '%s %s bdist_wininst --skip-build --target-version=%s ' \
               '--title="PyTango 7" ' \
               '--bitmap="%s"' % (executable, setup_name, ver, bitmap)
    os.system(cmd_line)
except:
    print "Failed:"
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    os.chdir(curr_dir)
