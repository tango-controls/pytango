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

nx = 32
spectre = np.array([float(value*value) for value in range(nx)])
dp.write_attribute('double_spectrum', spectre)
attr = dp.read_attribute('double_spectrum')
assert attr.data_format == tango._tango.AttrDataFormat.SPECTRUM
assert attr.dim_x == nx
assert attr.dim_y == 0
assert attr.has_failed == False
assert attr.is_empty == False
assert attr.name == 'double_spectrum'
assert attr.nb_read == nx
assert attr.nb_written == nx
assert attr.quality == tango._tango.AttrQuality.ATTR_VALID
assert type(attr.r_dimension) == tango._tango.AttributeDimension
assert attr.r_dimension.dim_x == nx
assert attr.r_dimension.dim_y == 0
assert type(attr.time) == tango._tango.TimeVal
assert attr.type == tango._tango.CmdArgType.DevDouble
assert type(attr.value) == np.ndarray
assert((attr.value == spectre).all())
assert attr.w_dim_x == nx
assert attr. w_dim_y == 0
assert type(attr.w_dimension) == tango._tango.AttributeDimension
assert attr.w_dimension.dim_x == nx
assert attr.w_dimension.dim_y == 0
assert((attr.w_value == spectre).all())

spectre = np.array([True, True, False, True, False])
dp.write_attribute('boolean_spectrum', spectre)
attr = dp.read_attribute('boolean_spectrum')
assert type(attr.value) == np.ndarray
assert((attr.value == spectre).all())
assert((attr.w_value == spectre).all())

spectre = [i*i for i in range(32)]
dp.write_attribute('short_spectrum', spectre)
attr = dp.read_attribute('short_spectrum')
assert type(attr.value) == np.ndarray
assert((attr.value == spectre).all())
assert((attr.w_value == spectre).all())

spectre = [i*i*i for i in range(32)]
dp.write_attribute('long64_spectrum', spectre)
attr = dp.read_attribute('long64_spectrum')
assert type(attr.value) == np.ndarray
assert((attr.value == spectre).all())
assert((attr.w_value == spectre).all())

spectre = np.array(['ABCD', 'EFGH', 'IJKL', 'MNOP', 'QRST', 'UVWX'])
dp.write_attribute('string_spectrum', spectre)
attr = dp.read_attribute('string_spectrum')
assert type(attr.value) == np.ndarray
assert((attr.value == spectre).all())
assert((attr.w_value == spectre).all())

nx = 7
ny = 4
img = np.array([[(x*y) for x in range(nx)] for y in range(ny)],dtype=np.int16)
dp.write_attribute('short_image', img)
attr = dp.read_attribute('short_image')
assert attr.data_format == tango._tango.AttrDataFormat.IMAGE
assert attr.dim_x == nx
assert attr.dim_y == ny
assert attr.has_failed == False
assert attr.is_empty == False
assert attr.name == 'short_image'
assert attr.nb_read == nx*ny
assert attr.nb_written == nx*ny
assert attr.quality == tango._tango.AttrQuality.ATTR_VALID
assert type(attr.r_dimension) == tango._tango.AttributeDimension
assert attr.r_dimension.dim_x == nx
assert attr.r_dimension.dim_y == ny
assert type(attr.time) == tango._tango.TimeVal
assert attr.type == tango._tango.CmdArgType.DevShort
assert attr.w_dim_x == nx
assert attr. w_dim_y == ny
assert type(attr.w_dimension) == tango._tango.AttributeDimension
assert attr.w_dimension.dim_x == nx
assert attr.w_dimension.dim_y == ny
assert type(attr.value) == np.ndarray
for a, b in zip(attr.value.ravel(), img.ravel()):
    assert a == b
for a, b in zip(attr.w_value.ravel(), img.ravel()):
    assert a == b

img = np.array([['one', 'two', 'three', 'four'],['five', 'six', 'seven', 'eight'], ['nine', 'ten', 'eleven', 'twelve']])
dp.write_attribute('string_image', img)
attr = dp.read_attribute('string_image')
assert type(attr.value) == np.ndarray
for a, b in zip(attr.value.ravel(), img.ravel()):
    assert a == b
for a, b in zip(attr.w_value.ravel(), img.ravel()):
    assert a == b

print("passed")