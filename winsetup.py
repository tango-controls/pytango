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

from __future__ import print_function

import sys
import os
import os.path as osp

def main():
    executable = sys.executable

    curr_dir = os.getcwd()

    winsetup_dir = osp.dirname(osp.abspath(__file__))
    os.chdir(winsetup_dir)
    setup_name = "setup.py"
    bitmap = osp.join(winsetup_dir, 'doc', 'logo-medium.bmp')
    ver = ".".join(map(str, sys.version_info[:2]))

    if len(sys.argv) < 6:
        print("Need to supply build directory, distribution directory, temporary binary install directory, configuration name and platform name")
        return 1

    build_dir, dist_dir, bdist_dir = map(osp.abspath, sys.argv[1:4])
    config_name, plat_name = sys.argv[4:6]
#    temp_base_dir = osp.abspath(os.environ["TEMP"])
#    temp_dir = osp.join(temp_base_dir, "PyTango", config_name)

    try:
        cmd_line =  '%s %s ' % (executable, setup_name)
        cmd_line += 'build_py --force --no-compile ' \
                    '--build-lib=%s ' \
                    % (build_dir,)
        cmd_line += 'build_scripts --force '
        cmd_line += 'install_lib --skip-build --no-compile ' \
                    '--build-dir=%s ' \
                    % (build_dir, )
        cmd_line += 'bdist_msi --skip-build --target-version=%s ' \
                    '--bdist-dir=%s ' \
                    '--dist-dir=%s ' \
                    '--plat-name=%s ' \
                    '--install-script=winpostinstall.py ' % (ver, bdist_dir, dist_dir, plat_name)
        cmd_line += 'bdist_wininst --skip-build --target-version=%s ' \
                    '--bdist-dir=%s ' \
                    '--dist-dir=%s ' \
                    '--title="PyTango 8" ' \
                    '--bitmap="%s" ' \
                    '--plat-name=%s ' \
                    '--install-script=winpostinstall.py ' % (ver, bdist_dir, dist_dir, bitmap, plat_name)
        os.system(cmd_line)
    except:
        print("Failed:")
        import traceback
        traceback.print_exc()
        return 2
    finally:
        os.chdir(curr_dir)

    return 0

if __name__ == "__main__":
    ret = main()
    sys.exit(ret)