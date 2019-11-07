import tango
import pytest
import numpy as np
from time import sleep
from tango import DeviceProxy
from tango import DevState
from tango import Database
from tango import DbDatum
from tango import Release

dp = DeviceProxy('sys/tg_test/1')

bytes = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], dtype='byte')
dp['encoded_byte_scalar'] = ("String format", bytes)
attr = dp['encoded_byte_scalar']
assert attr.value[0] == "String format"
assert((attr.value[1] == bytes).all())

dp['long_scalar','double_scalar', 'string_scalar', 'boolean_scalar'] = 1234, 6.284, 'abcd', True
attr = dp['long_scalar','double_scalar', 'string_scalar', 'boolean_scalar']
assert attr[0].value == 1234
assert attr[1].value == 6.284
assert attr[2].value == 'abcd'
assert attr[3].value == True
  
dp['ulong_scalar','float_scalar', 'long64_scalar', 'ulong64_scalar'] = 65535, 6.284, -987654321, 987654321
attr = dp['ulong_scalar','float_scalar', 'long64_scalar', 'ulong64_scalar']
assert attr[0].value == 65535
assert attr[1].value == pytest.approx(6.284, rel=1e-3)
assert attr[2].value == -987654321
assert attr[3].value == 987654321
  
dp['ushort_scalar','short_scalar'] = 1234, -6284
attr = dp['ushort_scalar','short_scalar']
assert attr[0].value == 1234
assert attr[1].value == -6284
  
dp['long_scalar','double_spectrum', 'string_scalar', 'long_spectrum'] = 1234, [3.142, 6.284, 9.426], 'abcd', np.array([1,2,3,4,5])
attr = dp['long_scalar','double_spectrum', 'string_scalar', 'long_spectrum']
assert attr[0].value == 1234
spectre = np.array([3.142, 6.284, 9.426])
assert type(attr[1].value) == np.ndarray
assert((attr[1].value == spectre).all())
assert attr[2].value == 'abcd'
spectre = np.array((1,2,3,4,5))
assert type(attr[3].value) == np.ndarray
assert((attr[3].value == spectre).all())
 
dp['long64_spectrum', 'long_spectrum'] = np.array([1,2,3,4,5]), np.array([11,12,13,14,15])
attr = dp['long64_spectrum', 'long_spectrum']
spectre1 = np.array((1,2,3,4,5))
spectre2 = np.array((11,12,13,14,15))
assert type(attr[0].value) == np.ndarray
assert type(attr[1].value) == np.ndarray
assert((attr[0].value == spectre1).all())
assert((attr[1].value == spectre2).all())
 
dp['long64_spectrum', 'float_spectrum', 'short_spectrum'] = [31,32,33,34,35], np.array([21.,22.,23.,24.,25.]), [9,8,7,6,5,4,3,2,1]
attr = dp['long64_spectrum', 'float_spectrum', 'short_spectrum']
spectre1 = np.array((31,32,33,34,35))
assert type(attr[0].value) == np.ndarray
assert((attr[0].value == spectre1).all())
spectre2 = np.array((21.,22.,23.,24.,25.))
assert type(attr[1].value) == np.ndarray
assert((attr[1].value == spectre2).all())
spectre3 = np.array((9,8,7,6,5,4,3,2,1))
assert type(attr[2].value) == np.ndarray
assert((attr[2].value == spectre3).all())

print("passed")