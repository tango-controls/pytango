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

""" IPython 'spock' profile, to preload PyTango and offer a friendly interface to Tango."""

import IPython.ipapi
import ipy_defaults

def main():
    ip = IPython.ipapi.get()
    try:
        ip.ex("import IPython.ipapi")
        ip.ex("import PyTango.ipython")
        ip.ex("PyTango.ipython.init_ipython(IPython.ipapi.get())")
    except ImportError:
        print "Unable to start spock profile, is PyTango installed?"

main()
