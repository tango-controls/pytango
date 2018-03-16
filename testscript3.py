import tango
import socket
import pytest
import numpy as np
from time import sleep
from tango import DeviceProxy
from tango import DevState
from tango import Database
from tango import DbDatum
from tango import Release

dp = DeviceProxy('sys/tg_test/1')

try:
    dp.write_attribute('long_scaler', 12345)
except tango.DevFailed as e:
    result = e.__str__() 
    strng  = ("DevFailed[\nDevError[\n    desc = MARSHAL CORBA system exception: MARSHAL_InvalidEnumValue\n"
              "  origin = DeviceProxy::write_attribute()\n  reason = API_CorbaException\nseverity = ErrSeverity.ERR]\n\n"
              "DevError[\n    desc = Failed to execute write_attributes on device sys/tg_test/1\n"
              "  origin = DeviceProxy::write_attribute()\n  reason = API_CommunicationFailed\nseverity = ErrSeverity.ERR]\n]\n")
    for c1, c2 in zip(result,strng):
        if c1 != c2:
            assert False
    assert True
try:
    dp.write_attribute('long_scaler', 'abcd')
except tango.DevFailed as e:
    result = e.__str__()
    strng = ("DevFailed[\nDevError[\n    desc = long_scaler attribute not found\n"
             "  origin = MultiAttribute::get_attr_ind_by_name\n  reason = API_AttrNotFound\nseverity = ErrSeverity.ERR]\n\n"
             "DevError[\n    desc = Failed to write_attribute on device sys/tg_test/1, attribute long_scaler\n"
             "  origin = DeviceProxy::write_attribute()\n  reason = API_AttributeFailed\nseverity = ErrSeverity.ERR]\n]\n")
    for c1, c2 in zip(result,strng):
        if c1 != c2:
            assert False
    assert True

try:
    dp.read_attribute('throw_exception')
except tango.DevFailed as e:
    print e

print("passed")
