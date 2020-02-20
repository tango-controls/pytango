import tango
import socket
import pytest
import numpy as np
# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2019 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

from time import sleep
from tango import DeviceProxy
from tango import DevState
from tango import Database
from tango import DbDatum
from tango import Release

dp = DeviceProxy('sys/tg_test/1')

msg  = ("DevFailed[\n[DevError(desc = 'MARSHAL CORBA system exception: MARSHAL_InvalidEnumValue', origin = 'DeviceProxy::write_attribute()', reason = 'API_CorbaException', severity = ErrSeverity.ERR), DevError(desc = 'Failed to execute write_attributes on device sys/tg_test/1', origin = 'DeviceProxy::write_attribute()', reason = 'API_CommunicationFailed', severity = ErrSeverity.ERR)]]")
try:
    dp.write_attribute('long_scaler', 12345)
except tango.DevFailed as e:
    result = e.__str__()
    for c1, c2 in zip(result, msg):
        if c1 != c2:
            assert False
    assert True

msg = ("DevFailed[\n[DevError(desc = 'MARSHAL CORBA system exception: MARSHAL_InvalidEnumValue', origin = 'DeviceProxy::write_attribute()', reason = 'API_CorbaException', severity = ErrSeverity.ERR), DevError(desc = 'Failed to execute write_attributes on device sys/tg_test/1', origin = 'DeviceProxy::write_attribute()', reason = 'API_CommunicationFailed', severity = ErrSeverity.ERR)]]")
try:
    dp.write_attribute('long_scaler', 'abcd')
except tango.DevFailed as e:
    result = e.__str__()
    for c1, c2 in zip(result, msg):
        if c1 != c2:
            assert False
    assert True

print("passed")
