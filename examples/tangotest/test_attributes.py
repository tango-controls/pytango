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
import pytest
import numpy as np
from time import sleep
from tango import DevState
from tango import DeviceProxy

dp = DeviceProxy('sys/tg_test/1')

assert dp.get_attribute_list() == [u'devstate_scalar', u'encoded_string_scalar',
                                   u'long64_scalar', u'double_scalar',
                                   u'boolean_spectrum', u'long_scalar_ro',
                                   u'boolean_scalar', u'encoded_string_scalar_ro',
                                   u'ulong_spectrum', u'long_scalar',
                                   u'devstate_spectrum', u'uchar_scalar',
                                   u'uchar_spectrum', u'long64_scalar_ro',
                                   u'ulong64_spectrum', u'ulong_scalar',
                                   u'string_scalar', u'long64_spectrum',
                                   u'string_image', u'ulong64_scalar',
                                   u'ulong_image', u'encoded_byte_scalar',
                                   u'short_scalar_ro', u'ushort_image',
                                   u'ulong_scalar_ro', u'ushort_scalar',
                                   u'long_image', u'ulong64_scalar_ro',
                                   u'encoded_byte_scalar_ro', u'float_scalar',
                                   u'float_scalar_ro', u'boolean_image',
                                   u'ulong64_image', u'short_spectrum',
                                   u'ushort_spectrum', u'short_scalar',
                                   u'ushort_scalar_ro', u'float_image',
                                   u'double_scalar_ro', u'long64_image',
                                   u'string_spectrum', u'float_spectrum',
                                   u'double_spectrum', u'double_image',
                                   u'long_spectrum', u'uchar_image', u'short_image',
                                   u'State', u'Status']

reply = dp.get_attribute_config('float_scalar')
assert type(reply) == tango._tango.AttributeInfoEx
assert type(reply.alarms) == tango._tango.AttributeAlarmInfo
assert reply.alarms.delta_t == 'Not specified'
assert reply.alarms.delta_val == 'Not specified'
assert reply.alarms.extensions == []
assert reply.alarms.max_alarm == 'Not specified'
assert reply.alarms.max_warning == 'Not specified'
assert reply.alarms.min_alarm == 'Not specified'
assert reply.alarms.min_warning == 'Not specified'
assert reply.data_format == tango._tango.AttrDataFormat.SCALAR
assert reply.data_type == tango._tango.CmdArgType.DevFloat
assert reply.description == 'A Tango::DevFloat scalar attribute'
assert reply.disp_level == tango._tango.DispLevel.OPERATOR
assert reply.display_unit == 'No display unit'
assert reply.enum_labels == []
assert type(reply.events) == tango._tango.AttributeEventInfo
assert type(reply.events.arch_event) == tango._tango.ArchiveEventInfo
assert reply.events.arch_event.archive_abs_change == 'Not specified'
assert reply.events.arch_event.archive_period == 'Not specified'
assert reply.events.arch_event.archive_rel_change == 'Not specified'
assert reply.events.arch_event.extensions == []
assert type(reply.events.ch_event) == tango._tango.ChangeEventInfo
assert reply.events.ch_event.abs_change == 'Not specified'
assert reply.events.ch_event.extensions == []
assert reply.events.ch_event.rel_change == 'Not specified'
assert type(reply.events.per_event) == tango._tango.PeriodicEventInfo
assert reply.events.per_event.extensions == []
assert reply.events.per_event.period == '1000'

assert reply.format == '%6.2f'
assert reply.label == 'float_scalar'
assert reply.max_alarm == 'Not specified'
assert reply.max_dim_x == 1
assert reply.max_dim_y == 0
assert float(reply.max_value) == 4096.0
assert float(reply.min_value) == 0.0
assert reply.memorized == tango._tango.AttrMemorizedType.NONE
assert reply.min_alarm == 'Not specified'
assert reply.name == 'float_scalar'
assert reply.root_attr_name == u'Not specified'
assert reply.standard_unit == 'No standard unit'
assert reply.sys_extensions == []
assert reply.unit == ''
assert reply.writable == tango._tango.AttrWriteType.WT_UNKNOWN
assert reply.writable_attr_name == 'float_scalar'

reply = dp.get_attribute_config(['float_scalar', 'long_scalar'])
assert len(reply) == 2
assert type(reply[0]) == tango._tango.AttributeInfo
assert reply[1].data_format == tango._tango.AttrDataFormat.SCALAR
assert reply[1].data_type == tango._tango.CmdArgType.DevLong
assert reply[1].description == 'A Tango::DevLong scalar attribute (int32)'
assert reply[1].disp_level == tango._tango.DispLevel.OPERATOR
assert reply[1].display_unit == 'No display unit'
assert reply[1].extensions == []
assert reply[1].format == '%d'
assert reply[1].label == 'long_scalar'
assert reply[1].max_alarm == 'Not specified'
assert reply[1].max_dim_x == 1
assert reply[1].max_dim_y == 0
assert reply[1].max_value == 'Not specified'
assert reply[1].min_alarm == 'Not specified'
assert reply[1].min_value == 'Not specified'
assert reply[1].name == 'long_scalar'
assert reply[1].standard_unit == 'No standard unit'
assert reply[1].unit == ''
assert reply[1].writable == tango._tango.AttrWriteType.WT_UNKNOWN
assert reply[1].writable_attr_name == 'long_scalar'

reply1 = dp.get_attribute_config_ex('float_scalar')
assert len(reply1) == 1
assert type(reply1[0]) == tango._tango.AttributeInfoEx
reply2 = dp.get_attribute_config_ex(['float_scalar', 'long_scalar'])
assert len(reply2) == 2
assert type(reply2[0]) == tango._tango.AttributeInfoEx
attr_info = dp.attribute_query('string_scalar')
assert type(attr_info) == tango._tango.AttributeInfoEx

dp.write_attribute('long_scalar', 23456)
attr = dp.read_attribute('long_scalar')
assert attr.data_format == tango._tango.AttrDataFormat.SCALAR
assert attr.dim_x == 1
assert attr.dim_y == 0
assert attr.has_failed is False
assert attr.is_empty is False
assert attr.name == 'long_scalar'
assert attr.nb_read == 1
assert attr.nb_written == 1
assert attr.quality == tango._tango.AttrQuality.ATTR_VALID
assert type(attr.r_dimension) == tango._tango.AttributeDimension
assert attr.r_dimension.dim_x == 1
assert attr.r_dimension.dim_y == 0
assert type(attr.time) == tango._tango.TimeVal
assert attr.type == tango._tango.CmdArgType.DevLong
assert attr.value == 23456
assert attr.w_dim_x == 1
assert attr. w_dim_y == 0
assert type(attr.w_dimension) == tango._tango.AttributeDimension
assert attr.w_dimension.dim_x == 1
assert attr.w_dimension.dim_y == 0
assert attr.w_value == 23456

dp.write_attribute('long64_scalar', -987654321)
assert dp.read_attribute('long64_scalar').value == -987654321
dp.write_attribute('float_scalar', 45.67)
assert dp.read_attribute('float_scalar').value == pytest.approx(45.67, rel=1e-3)
dp.write_attribute('ulong_scalar', 987654)
assert dp.read_attribute('ulong_scalar').value == 987654
dp.write_attribute('ulong64_scalar', 987654321)
assert dp.read_attribute('ulong64_scalar').value == 987654321
dp.write_attribute('ushort_scalar', 65535)
assert dp.read_attribute('ushort_scalar').value == 65535
dp.write_attribute('uchar_scalar', 255)
assert dp.read_attribute('uchar_scalar').value == 255

assert dp.read_attribute('status').value == 'The device is in RUNNING state.'
assert dp.read_attribute('state').value == DevState.RUNNING
dp.write_attribute('string_scalar', "AbcdefghijklmnopqrstuvwxyZ")
assert dp.read_attribute('string_scalar').value == "AbcdefghijklmnopqrstuvwxyZ"
dp.write_attribute('encoded_byte_scalar', ("Array of Bytes", ([1, 2, 3, 4, 5, 6, 7, 8, 9])))
attr = dp.read_attribute('encoded_byte_scalar')
assert attr.value[0] == 'Array of Bytes'
assert attr.value[1] == [1, 2, 3, 4, 5, 6, 7, 8, 9]

dp.write_attribute('double_scalar', 3.142)
dp.write_attribute('long_scalar', 1357)
dp.write_attribute('boolean_scalar', True)
attrs = dp.read_attributes(['long_scalar', 'double_scalar', 'boolean_scalar'])
vals = [attrib.value for attrib in attrs]
assert vals == [1357, 3.142, True]
args = [('long_scalar', 8642), ('double_scalar', 6.284), ('boolean_scalar', False)]
dp.write_attributes(args)
attrs = dp.read_attributes(['long_scalar', 'double_scalar', 'boolean_scalar'])
vals = [attrib.value for attrib in attrs]
assert vals == [8642, 6.284, False]
assert dp.write_read_attribute('long_scalar', 98765).value == 98765
args = [('long_scalar', 3459), ('double_scalar', 15.71), ('boolean_scalar', True)]
attrs = dp.write_read_attributes(args)
vals = [attrib.value for attrib in attrs]
assert vals == [3459, 15.71, True]

args = [('long_scalar', 4567), ('double_scalar', 12.568), ('boolean_scalar', True)]
dp.write_attributes(args)
id = dp.read_attributes_asynch(['long_scalar', 'double_scalar', 'boolean_scalar'])
sleep(0.05)
attrs = dp.read_attributes_reply(id)
vals = [attrib.value for attrib in attrs]
assert vals == [4567, 12.568, True]
args = [('long_scalar', 1289), ('double_scalar', 9.426), ('boolean_scalar', False)]
dp.write_attributes(args)
id = dp.read_attributes_asynch(['long_scalar', 'double_scalar', 'boolean_scalar'])
attrs = dp.read_attributes_reply(id, 50)
vals = [attrib.value for attrib in attrs]
assert vals == [1289, 9.426, False]
with pytest.raises(Exception):
    id = dp.read_attributes_asynch(['long_scalar', 'double_scalar', 'boolean_scalar'])
    attrs = dp.read_attributes_reply(id, 1)  # should timeout and raise an exception
assert dp.pending_asynch_call(tango.asyn_req_type.ALL_ASYNCH) == 1

args = [('long_scalar', 1289), ('double_scalar', 9.426), ('boolean_scalar', False)]
id = dp.write_attributes_asynch(args)
sleep(0.05)
dp.write_attributes_reply(id)
args = [('long_scalar', 1289)]
id = dp.write_attributes_asynch(args)
with pytest.raises(Exception):
    dp.write_attributes_reply(id, 1)  # should timeout and raise an exception

nx = 32
spectre = np.array([float(value * value) for value in range(nx)])
dp.write_attribute('double_spectrum', spectre)
attr = dp.read_attribute('double_spectrum')
assert attr.data_format == tango._tango.AttrDataFormat.SPECTRUM
assert attr.dim_x == nx
assert attr.dim_y == 0
assert attr.has_failed is False
assert attr.is_empty is False
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

spectre = [i * i for i in range(32)]
dp.write_attribute('short_spectrum', spectre)
attr = dp.read_attribute('short_spectrum')
assert type(attr.value) == np.ndarray
assert((attr.value == spectre).all())
assert((attr.w_value == spectre).all())

spectre = [i * i * i for i in range(32)]
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
img = np.array([[(x * y) for x in range(nx)] for y in range(ny)], dtype=np.int16)
dp.write_attribute('short_image', img)
attr = dp.read_attribute('short_image')
assert attr.data_format == tango._tango.AttrDataFormat.IMAGE
assert attr.dim_x == nx
assert attr.dim_y == ny
assert attr.has_failed is False
assert attr.is_empty is False
assert attr.name == 'short_image'
assert attr.nb_read == nx * ny
assert attr.nb_written == nx * ny
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

img = np.array([['one', 'two', 'three', 'four'], ['five', 'six', 'seven', 'eight'], ['nine', 'ten', 'eleven', 'twelve']])
dp.write_attribute('string_image', img)
attr = dp.read_attribute('string_image')
assert type(attr.value) == np.ndarray
for a, b in zip(attr.value.ravel(), img.ravel()):
    assert a == b
for a, b in zip(attr.w_value.ravel(), img.ravel()):
    assert a == b

bytes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='byte')
dp['encoded_byte_scalar'] = ("String format", bytes)
attr = dp['encoded_byte_scalar']
assert attr.value[0] == "String format"
assert((attr.value[1] == bytes).all())

dp['long_scalar', 'double_scalar', 'string_scalar', 'boolean_scalar'] = 1234, 6.284, 'abcd', True
attr = dp['long_scalar', 'double_scalar', 'string_scalar', 'boolean_scalar']
assert attr[0].value == 1234
assert attr[1].value == 6.284
assert attr[2].value == 'abcd'
assert attr[3].value is True

dp['ulong_scalar', 'float_scalar', 'long64_scalar', 'ulong64_scalar'] = 65535, 6.284, -987654321, 987654321
attr = dp['ulong_scalar', 'float_scalar', 'long64_scalar', 'ulong64_scalar']
assert attr[0].value == 65535
assert attr[1].value == pytest.approx(6.284, rel=1e-3)
assert attr[2].value == -987654321
assert attr[3].value == 987654321

dp['ushort_scalar', 'short_scalar'] = 1234, -6284
attr = dp['ushort_scalar', 'short_scalar']
assert attr[0].value == 1234
assert attr[1].value == -6284

dp['long_scalar', 'double_spectrum', 'string_scalar', 'long_spectrum'] = 1234, [3.142, 6.284, 9.426], 'abcd', np.array([1, 2, 3, 4, 5])
attr = dp['long_scalar', 'double_spectrum', 'string_scalar', 'long_spectrum']
assert attr[0].value == 1234
spectre = np.array([3.142, 6.284, 9.426])
assert type(attr[1].value) == np.ndarray
assert((attr[1].value == spectre).all())
assert attr[2].value == 'abcd'
spectre = np.array((1, 2, 3, 4, 5))
assert type(attr[3].value) == np.ndarray
assert((attr[3].value == spectre).all())

dp['long64_spectrum', 'long_spectrum'] = np.array([1, 2, 3, 4, 5]), np.array([11, 12, 13, 14, 15])
attr = dp['long64_spectrum', 'long_spectrum']
spectre1 = np.array((1, 2, 3, 4, 5))
spectre2 = np.array((11, 12, 13, 14, 15))
assert type(attr[0].value) == np.ndarray
assert type(attr[1].value) == np.ndarray
assert((attr[0].value == spectre1).all())
assert((attr[1].value == spectre2).all())

dp['long64_spectrum', 'float_spectrum', 'short_spectrum'] = [31, 32, 33, 34, 35], np.array([21., 22., 23., 24., 25.]), [9, 8, 7, 6, 5, 4, 3, 2, 1]
attr = dp['long64_spectrum', 'float_spectrum', 'short_spectrum']
spectre1 = np.array((31, 32, 33, 34, 35))
assert type(attr[0].value) == np.ndarray
assert((attr[0].value == spectre1).all())
spectre2 = np.array((21., 22., 23., 24., 25.))
assert type(attr[1].value) == np.ndarray
assert((attr[1].value == spectre2).all())
spectre3 = np.array((9, 8, 7, 6, 5, 4, 3, 2, 1))
assert type(attr[2].value) == np.ndarray
assert((attr[2].value == spectre3).all())

print("passed attribute tests")
