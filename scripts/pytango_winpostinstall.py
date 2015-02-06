# ------------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

import sys
import os.path

def install():
    import struct
    bits = 8 * struct.calcsize("P")
    is_64 = bits == 64
    pyver = "(Py {1}.{2} {0}bits)".format(bits, *sys.version_info[:2])
    
    itango_lnk = 'ITango {0}.lnk'.format(pyver)
    itango_qt_lnk = 'ITango Qt {0}.lnk'.format(pyver)
    
    python = os.path.join(sys.prefix, 'python.exe')
    pythonw = os.path.join(sys.prefix, 'pythonw.exe')
    
    desktop = get_special_folder_path("CSIDL_COMMON_DESKTOPDIRECTORY")
    tango_menu = os.path.join(get_special_folder_path("CSIDL_COMMON_PROGRAMS"), 'Tango')
    itango_args = os.path.join(sys.prefix, 'scripts', 'itango-script.py')
    itango_qt_args = os.path.join(sys.prefix, 'scripts', 'itango-qt-script.pyw')
    itango_icon = os.path.join(sys.prefix, 'scripts', 'itango.ico')
    itango_workdir = ''

    # create desktop shortcuts
    itango_desktop_shortcut = os.path.join(desktop, itango_lnk)
    create_shortcut(python, "ITango console", itango_desktop_shortcut,
                    itango_args, itango_workdir, itango_icon, 0)
    file_created(itango_desktop_shortcut)

    itango_qt_desktop_shortcut = os.path.join(desktop, itango_qt_lnk)
    create_shortcut(pythonw, "ITango Qt console", itango_qt_desktop_shortcut,
                    itango_qt_args, itango_workdir, itango_icon, 0)
    file_created(itango_qt_desktop_shortcut)

    # create tango menu shortcuts
    itango_menu_shortcut = os.path.join(tango_menu, itango_lnk)
    create_shortcut(python, "ITango console", itango_menu_shortcut,
                    itango_args, itango_workdir, itango_icon, 0)
    file_created(itango_menu_shortcut)
    
    itango_qt_menu_shortcut = os.path.join(tango_menu, itango_qt_lnk)
    create_shortcut(pythonw, "ITango Qt console", itango_qt_menu_shortcut,
                    itango_qt_args, itango_workdir, itango_icon, 0)
    file_created(itango_qt_menu_shortcut)
    
    
def remove():
    print ("Removing...")
    print ("DONE!")
    
    
def main():
    if '-install' in sys.argv:
        return install()
    elif '-remove' in sys.argv:
        return remove()


main()