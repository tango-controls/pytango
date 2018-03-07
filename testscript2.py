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
spectre = [float(value*value) for value in range(nx)]
dp.write_attribute('double_spectrum', spectre)
attr = dp.read_attribute('double_spectrum', extract_as=tango._tango.ExtractAs.ByteArray)
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
assert attr.value == spectre+spectre
assert attr.w_dim_x == nx
assert attr. w_dim_y == 0
assert type(attr.w_dimension) == tango._tango.AttributeDimension
assert attr.w_dimension.dim_x == nx
assert attr.w_dimension.dim_y == 0
assert attr.w_value == None


nx = 7
ny = 4
img = [[(x*y) for x in range(nx)] for y in range(ny)]
dp.write_attribute('short_image', img)
attr = dp.read_attribute('short_image', extract_as=tango._tango.ExtractAs.ByteArray)
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
result = [x*y for y in range(4) for x in range(7)]
assert attr.value == result + result
assert attr.w_dim_x == nx
assert attr. w_dim_y == ny
assert type(attr.w_dimension) == tango._tango.AttributeDimension
assert attr.w_dimension.dim_x == nx
assert attr.w_dimension.dim_y == ny
assert attr.w_value == None
print("passed")