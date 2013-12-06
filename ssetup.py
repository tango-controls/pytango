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

from ez_setup import use_setuptools
use_setuptools()

import setuptools
import setup


def main():
    sargs = setup.setup_args()
    sargs['entry_points'] = {
        "console_scripts": [
            "itango = PyTango.ipython:run",
        ],
    }
    dist = setuptools.setup(**sargs)
    return dist


if __name__ == "__main__":
    main()
