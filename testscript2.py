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
assert attr.value.all() == spectre.all()
assert attr.w_dim_x == nx
assert attr. w_dim_y == 0
assert type(attr.w_dimension) == tango._tango.AttributeDimension
assert attr.w_dimension.dim_x == nx
assert attr.w_dimension.dim_y == 0
assert attr.w_value.all() == spectre.all()

spectre = np.array([True, True, False, True, False])
dp.write_attribute('boolean_spectrum', spectre)
attr = dp.read_attribute('boolean_spectrum')
assert attr.value.all() == spectre.all()
assert attr.w_value.all() == spectre.all()

spectre = [i*i*i for i in range(32)]
dp.write_attribute('long_spectrum', spectre)
attr = dp.read_attribute('long_spectrum')
assert attr.value.all() == np.array(spectre).all()
assert attr.w_value.all() == np.array(spectre).all()

spectre = np.array(['ESRF', 'ALBA', 'MAXIV', 'PETRA3', 'SOLEIL', 'SOLARIS'])
dp.write_attribute('string_spectrum', spectre)
attr = dp.read_attribute('string_spectrum')

for a, b in zip(attr.value, spectre):
    assert a == b
for a, b in zip(attr.w_value, spectre):
    assert a == b

spectre = ['ESRF', 'ALBA', 'MAXIV', 'PETRA3', 'SOLEIL', 'SOLARIS']
dp.write_attribute('string_spectrum', spectre)
attr = dp.read_attribute('string_spectrum')
for a, b in zip(attr.value, np.array(spectre)):
    assert a == b
for a, b in zip(attr.w_value, np.array(spectre)):
    assert a == b

nx = 7
ny = 4
img = np.array([[(x*y) for x in range(nx)] for y in range(ny)])
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
assert attr.value.all() == img.all()
assert attr.w_dim_x == nx
assert attr. w_dim_y == ny
assert type(attr.w_dimension) == tango._tango.AttributeDimension
assert attr.w_dimension.dim_x == nx
assert attr.w_dimension.dim_y == ny
assert attr.w_value.all() == img.all()

img = np.array([['one', 'two', 'three', 'four'],['five', 'six', 'seven', 'eight'], ['nine', 'ten', 'eleven', 'twelve']])
dp.write_attribute('string_image', img)
attr = dp.read_attribute('string_image')
for a, b in zip(attr.value.ravel(), img.ravel()):
    assert a == b
for a, b in zip(attr.w_value.ravel(), img.ravel()):
    assert a == b

img = [['one', 'two', 'three', 'four'],['five', 'six', 'seven', 'eight'], ['nine', 'ten', 'eleven', 'twelve']]
dp.write_attribute('string_image', img)
attr = dp.read_attribute('string_image')
for a, b in zip(attr.value.ravel(), np.array(img).ravel()):
    assert a == b
for a, b in zip(attr.w_value.ravel(), np.array(img).ravel()):
    assert a == b

print("passed")