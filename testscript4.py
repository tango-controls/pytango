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

# dp['encoded_scalar_w'] = ("String array", [1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,16,17,18,19])
# attr = dp['encoded_scalar_w']
# print "result",attr.value

dp['long_scalar_w','double_scalar_w', 'string_scalar', 'boolean_scalar'] = 1234, 6.284, 'ESRF', True
attr = dp['long_scalar_w','double_scalar_w', 'string_scalar', 'boolean_scalar']
assert attr[0].value == 1234
assert attr[1].value == 6.284
assert attr[2].value == 'ESRF'
assert attr[3].value == True
  
dp['ulong_scalar_w','float_scalar_w', 'long64_scalar_w', 'ulong64_scalar_w'] = 65535, 6.284, -987654321, 987654321
attr = dp['ulong_scalar_w','float_scalar_w', 'long64_scalar_w', 'ulong64_scalar_w']
assert attr[0].value == 65535
assert attr[1].value == pytest.approx(6.284, rel=1e-3)
assert attr[2].value == -987654321
assert attr[3].value == 987654321
  
dp['ushort_scalar_w','short_scalar_w'] = 1234, -6284
attr = dp['ushort_scalar_w','short_scalar_w']
assert attr[0].value == 1234
assert attr[1].value == -6284
  
dp['long_scalar_w','double_spectrum', 'string_scalar', 'long_spectrum'] = 1234, [3.142, 6.284, 9.426], 'ESRF', np.array([1,2,3,4,5])
attr = dp['long_scalar_w','double_spectrum', 'string_scalar', 'long_spectrum']
assert attr[0].value == 1234
spectre = np.array([3.142, 6.284, 9.426])
assert set(attr[1].value) == set(spectre)
assert attr[2].value == 'ESRF'
spectre = np.array((1,2,3,4,5))
assert set(attr[3].value) == set(spectre)
 
dp['long64_spectrum', 'long_spectrum'] = np.array([1,2,3,4,5]), np.array([11,12,13,14,15])
attr = dp['long64_spectrum', 'long_spectrum']
spectre1 = np.array((1,2,3,4,5))
spectre2 = np.array((11,12,13,14,15))
assert set(attr[0].value) == set(spectre1)
assert set(attr[1].value) == set(spectre2)
 
dp['long64_spectrum', 'float_spectrum', 'short_spectrum'] = [31,32,33,34,35], np.array([21.,22.,23.,24.,25.]), [9,8,7,6,5,4,3,2,1]
attr = dp['long64_spectrum', 'float_spectrum', 'short_spectrum']
spectre3 = np.array((31,32,33,34,35))
assert set(attr[0].value) == set(spectre3)
spectre4 = np.array((21.,22.,23.,24.,25.))
assert set(attr[1].value) == set(spectre4)
spectre5 = np.array((9,8,7,6,5,4,3,2,1))
assert set(attr[2].value) == set(spectre5)

print("passed")