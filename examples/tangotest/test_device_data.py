# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2019 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

import tango
import numpy as np
from tango import DevState
from tango.device_data import DeviceData
from pytest import approx

dd = DeviceData()
dd.is_empty() is True
dd.insert(tango.CmdArgType.DevDouble, 3.142)
assert dd.get_type() == tango.CmdArgType.DevDouble
assert dd.is_empty() is False
assert dd.extract() == 3.142
dd.insert(tango.CmdArgType.DevFloat, 6.284)
assert dd.get_type() == tango.CmdArgType.DevFloat
assert dd.extract() == approx(6.284)
dd.insert(tango.CmdArgType.DevLong, 32767)
assert dd.extract() == 32767
dd.insert(tango.CmdArgType.DevShort, 255)
assert dd.extract() == 255
dd.insert(tango.CmdArgType.DevString, "This is a test string *!")
assert dd.extract() == "This is a test string *!"
dd.insert(tango.CmdArgType.DevBoolean, True)
assert dd.extract() is True
dd.insert(tango.CmdArgType.DevState, DevState.MOVING)
assert dd.extract() == DevState.MOVING
dd.insert(tango.CmdArgType.DevEncoded, ("format", [0, 1, 2, 3, 0xfd, 0xfe, 0xff]))
assert dd.extract() == ("format", [0, 1, 2, 3, 0xfd, 0xfe, 0xff])

dd.insert(tango.CmdArgType.DevVarDoubleArray, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
dvda = dd.extract()
assert type(dvda) == np.ndarray
assert((dvda == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).all())
dd.insert(tango.CmdArgType.DevVarDoubleArray, np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype='double'))
dvda = dd.extract()
assert type(dvda) == np.ndarray
assert((dvda == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]).all())
dd.insert(tango.CmdArgType.DevVarLong64Array, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
dvl64a = dd.extract()
assert type(dvl64a) == np.ndarray
assert((dvl64a == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]).all())
dd.insert(tango.CmdArgType.DevVarStringArray, ["abc", "def", "ghi", "jkl", "mno"])
assert dd.extract() == ["abc", "def", "ghi", "jkl", "mno"]
dd.insert(tango.CmdArgType.DevVarLongStringArray, ([1, 2, 3, 4, 5], ["abc", "def", "ghi", "jkl", "mno"]))
assert dd.extract() == ([1, 2, 3, 4, 5], ["abc", "def", "ghi", "jkl", "mno"])
dd.insert(tango.CmdArgType.DevVarDoubleStringArray, ([0.3, 0.4, 0.5, 0.6, 0.7], ["abc", "def", "ghi", "jkl", "mno"]))
assert dd.extract() == ([0.3, 0.4, 0.5, 0.6, 0.7], ["abc", "def", "ghi", "jkl", "mno"])

print("passed device_data tests")
