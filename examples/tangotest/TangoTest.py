# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2019 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

""" TANGO Device Server for testing generic clients

A device to test generic clients. It offers a \"echo\" like command for
each TANGO data type (i.e. each command returns an exact copy of <value>).
"""

__all__ = ["TangoTest"]

# tango imports
import tango
import gevent
import numpy as np
from gevent import lock
from tango import GreenMode
from tango import DevState
from tango.server import Device
from tango.server import attribute, command, pipe
from tango.server import device_property

# Additional import


class TangoTest(Device):
    """
    A device to test generic clients. It offers a \"echo\" like command for
    each TANGO data type (i.e. each command returns an exact copy of <value>).
    """
    # arbitary minimum and maximum values that each
    # attribute is allowed. Also used to generate random
    # values to continually update each attribute
    double_minmax = (0.0, 9999999.0)
    float_minmax = (0.0, 4096.0)
    long64_minmax = (-4294967296, 4294967296)
    long_minmax = (-65536, 65536)
    short_minmax = (-32767, 32767)
    ulong64_minmax = (0, 4294967296)
    ulong_minmax = (0, 65536)
    ushort_minmax = (0, 32767)

    # ----------------
    # Class Properties
    # ----------------

    # -----------------
    # Device Properties
    # -----------------

    multithreaded_impl = device_property(dtype="int16")
    sleep_period = device_property(dtype="int")
    image_size = device_property(dtype="int", default_value=251)

    # ---------------
    # General methods
    # ---------------

    def init_device(self):
        Device.init_device(self)
        self.__boolean_scalar = True
        self.__double_scalar = 152.34
        self.__double_scalar_ro = 6.284
        self.__float_scalar = 3.142
        self.__float_scalar_ro = 3.142
        self.__long64_scalar = 4294967296
        self.__long64_scalar_ro = 4294967296
        self.__long_scalar = 65536
        self.__long_scalar_ro = 65536
        self.__short_scalar = 32767
        self.__short_scalar_ro = 32767
        self.__string_scalar = "test string"
        self.__uchar_scalar = 255
        self.__ulong64_scalar = 4294967296
        self.__ulong64_scalar_ro = 4294967296
        self.__ulong_scalar = 65536
        self.__ulong_scalar_ro = 65536
        self.__ushort_scalar = 32767
        self.__ushort_scalar_ro = 32767
        self.__encoded_string_scalar = ("format", "real data")
        self.__encoded_string_scalar_ro = ("format", "real data")
        self.__encoded_byte_scalar = ("format",
                                      [0x00, 0x01, 0x02, 0xfd, 0xfe, 0xff])
        self.__encoded_byte_scalar_ro = ("format",
                                         [0x00, 0x01, 0x02, 0xfd, 0xfe, 0xff])
        self.__devstate_scalar = DevState.ON
        self.__boolean_spectrum = [True, True, False, False]
        self.__double_spectrum = np.array([10.1, 11.1, 12.1, 13.1, 14.1], np.float64)
        self.__float_spectrum = np.array([0.1, 1.1, 2.1, 3.1, 4.1], np.float)
        self.__long64_spectrum = np.array([110, 111, 112, 113, 114, 115, 116, 117, 118, 119], np.int64)
        self.__long_spectrum = np.array([10, 11, 12, 13, 14, 15, 16], np.int32)
        self.__short_spectrum = np.array([0, -1, -2, -3, -4], np.short)
        self.__string_spectrum = ["abc12", "def34", "ghi56", "jkl78"]
        self.__uchar_spectrum = np.array([0, 1, 2, 3, 4, 5, 6, 7], np.byte)
        self.__ulong64_spectrum = np.array([200, 201, 202, 203, 204, 205], np.uint64)
        self.__ulong_spectrum = np.array([20, 21, 22, 23, 24, 25, 26], np.uint32)
        self.__ushort_spectrum = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], np.ushort)
        self.__devstate_spectrum = np.array([DevState.ON, DevState.OFF, DevState.MOVING])
        self.__boolean_image = [[True]]
        self.__double_image = np.array([[10.1, 11.1, 12.1, 13.1, 14.1],
                                        [10.2, 11.2, 12.2, 13.2, 14.2],
                                        [10.3, 11.3, 12.3, 13.3, 14.3]], np.float64)
        self.__float_image = np.array([[0.1, 1.1, 2.1, 3.1, 4.1],
                                       [0.2, 1.2, 2.2, 3.2, 4.2],
                                       [0.3, 1.3, 2.3, 3.3, 4.3]], np.float)
        self.__long64_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int64)
        self.__long_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int32)
        self.__short_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int8)
        self.__string_image = [["abc12", "def34", "ghi56", "jkl78"],
                               ["mno12", "pqr34", "stu56", "vwx78"]]
        self.__uchar_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.byte)
        self.__ulong64_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.uint64)
        self.__ulong_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.uint32)
        self.__ushort_image = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.ushort)
        self.__rootBlobName = 'theBlob'
        self.__inner_blob = ("Inner", [("double_list", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                                       ("np_array", np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], np.int64)),
                                       ("int_list", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
                                       ("string_list", ["abc", "def", "ghi", "jkl", "mno"]),
                                       ]
                             )
        self.__blob = self.__rootBlobName, [("double", 3.142),
                                            ("integer64", 32767),
                                            ("string", "abcdefghjklmno"),
                                            ("bool", True),
                                            ("innerblob", self.__inner_blob),
                                            {"name": "encoded", "value": ("format",
                                                                          [0x00, 0x01, 0x02, 0xfd, 0xfe, 0xff]), "dtype": tango.CmdArgType.DevEncoded},
                                            ]
        self.__lock = lock.Semaphore()
        self.__generate_task = None
        self.__push_scalar_change_events = False
        self.__push_scalar_archive_events = False
        self.__push_pipe_events = False
        self.set_state(tango.DevState.RUNNING)

    def always_executed_hook(self):
        pass

    def delete_device(self):
        pass

    # ------------------
    # Attributes methods
    # ------------------

    @attribute(
        dtype="bool",
        label="boolean_scalar",
        doc="A Tango::DevBoolean scalar attribute",
    )
    def boolean_scalar(self):
        return self.__boolean_scalar

    @boolean_scalar.write
    def boolean_scalar(self, value):
        self.__boolean_scalar = value

    @attribute(
        dtype="double",
        label="double_scalar",
        min_warning="-1.0",
        max_warning="20000.0",
        abs_change=0.1,
        doc="A Tango::DevDouble scalar attribute",
    )
    def double_scalar(self):
        return self.__double_scalar

    @double_scalar.write
    def double_scalar(self, value):
        self.__double_scalar = value

    @attribute(
        dtype="double",
        label="double_scalar_ro",
        min_value=double_minmax[0],
        max_value=double_minmax[1],
        doc="A Tango::DevDouble readonly scalar attribute",
    )
    def double_scalar_ro(self):
        return self.__double_scalar_ro

    @attribute(
        dtype='DevFloat',
        label="float_scalar",
        min_value=float_minmax[0],
        max_value=float_minmax[1],
        doc="A Tango::DevFloat scalar attribute",
    )
    def float_scalar(self):
        return self.__float_scalar

    @float_scalar.write
    def float_scalar(self, value):
        self.__float_scalar = value

    @attribute(
        dtype="float",
        label="float_scalar_ro",
        min_value=float_minmax[0],
        max_value=float_minmax[1],
        doc="A Tango::DevFloat readonly scalar attribute",
    )
    def float_scalar_ro(self):
        return self.__float_scalar_ro

    @attribute(
        dtype="int64",
        label="long64_scalar",
        doc="A Tango::DevLong64 scalar attribute (int64)",
    )
    def long64_scalar(self):
        return self.__long64_scalar

    @long64_scalar.write
    def long64_scalar(self, value):
        self.__long64_scalar = value

    @attribute(
        dtype="int64",
        label="long64_scalar_ro",
        min_value=long64_minmax[0],
        max_value=long64_minmax[1],
        doc="A Tango::DevLong64 readonly scalar attribute (int64)",
    )
    def long64_scalar_ro(self):
        return self.__long64_scalar_ro

    @attribute(
        dtype="int32",
        label="long_scalar",
        doc="A Tango::DevLong scalar attribute (int32)",
    )
    def long_scalar(self):
        return self.__long_scalar

    @long_scalar.write
    def long_scalar(self, value):
        self.__long_scalar = value

    @attribute(
        dtype="int32",
        label="long_scalar_ro",
        min_value=long_minmax[0],
        max_value=long_minmax[1],
        doc="A Tango::DevLong readonly scalar attribute (int32)",
    )
    def long_scalar_ro(self):
        return self.__long_scalar_ro

    @attribute(
        dtype="int16",
        label="short_scalar",
        doc="A Tango::DevShort scalar attribute (int16)",
    )
    def short_scalar(self):
        return self.__short_scalar

    @short_scalar.write
    def short_scalar(self, value):
        self.__short_scalar = value

    @attribute(
        dtype="int16",
        label="short_scalar_ro",
        doc="A Tango::DevShort readonly scalar attribute (int16)",
    )
    def short_scalar_ro(self):
        return self.__short_scalar_ro

    @attribute(
        dtype="str",
        label="string_scalar",
        doc="A string scalar attribute",
    )
    def string_scalar(self):
        return self.__string_scalar

    @string_scalar.write
    def string_scalar(self, value):
        self.__string_scalar = value

    @attribute(
        dtype="byte",
        label="uchar_scalar",
        doc="A uchar scalar attribute",
    )
    def uchar_scalar(self):
        return self.__uchar_scalar

    @uchar_scalar.write
    def uchar_scalar(self, value):
        self.__uchar_scalar = value

    @attribute(
        dtype="uint64",
        label="ulong64_scalar",
        doc="A Tango::DevULong64 scalar attribute (uint64)",
    )
    def ulong64_scalar(self):
        return self.__ulong64_scalar

    @ulong64_scalar.write
    def ulong64_scalar(self, value):
        self.__ulong64_scalar = value

    @attribute(
        dtype="uint64",
        label="ulong64_scalar_ro",
        min_valu=ulong64_minmax[0],
        max_value=ulong64_minmax[1],
        doc="A Tango::DevULong64 readonly scalar attribute (int64)",
    )
    def ulong64_scalar_ro(self):
        return self.__ulong64_scalar_ro

    @attribute(
        dtype="uint32",
        label="ulong_scalar",
        doc="A Tango::DevULong scalar attribute (uint32)",
    )
    def ulong_scalar(self):
        return self.__ulong_scalar

    @ulong_scalar.write
    def ulong_scalar(self, value):
        self.__ulong_scalar = value

    @attribute(
        dtype="uint32",
        label="ulong_scalar_ro",
        min_value=ulong_minmax[0],
        max_value=ulong_minmax[1],
        doc="A Tango::DevULong readonly scalar attribute (int32)",
    )
    def ulong_scalar_ro(self):
        return self.__ulong_scalar_ro

    @attribute(
        dtype="uint16",
        label="ushort_scalar",
        doc="A Tango::UShort scalar attribute (uint16)",
    )
    def ushort_scalar(self):
        return self.__ushort_scalar

    @ushort_scalar.write
    def ushort_scalar(self, value):
        self.__ushort_scalar = value

    @attribute(
        dtype="uint16",
        label="ushort_scalar_ro",
        min_value=ushort_minmax[0],
        max_value=ushort_minmax[1],
        doc="A Tango::UShort readonly scalar attribute (uint16)",
    )
    def ushort_scalar_ro(self):
        return self.__ushort_scalar_ro

    @attribute(
        dtype="DevEncoded",
        label="encoded_string_scalar",
        doc="A encoded string scalar attribute",
    )
    def encoded_string_scalar(self):
        return self.__encoded_string_scalar

    @encoded_string_scalar.write
    def encoded_string_scalar(self, value):
        self.__encoded_string_scalar = value

    @attribute(
        dtype="DevEncoded",
        label="encoded_string_scalar_ro",
        doc="A encoded string readonly scalar attribute",
    )
    def encoded_string_scalar_ro(self):
        return self.__encoded_string_scalar_ro

    @attribute(
        dtype="DevEncoded",
        label="encoded_byte_scalar",
        doc="A encoded byte scalar attribute",
    )
    def encoded_byte_scalar(self):
        return self.__encoded_byte_scalar

    @encoded_byte_scalar.write
    def encoded_byte_scalar(self, value):
        self.__encoded_byte_scalar = value

    @attribute(
        dtype="DevEncoded",
        label="encoded_byte_scalar_ro",
        doc="A encoded byte readonly scalar attribute",
    )
    def encoded_byte_scalar_ro(self):
        return self.__encoded_byte_scalar_ro

    @attribute(
        dtype="DevState",
        label="device state",
        doc="A device state attribute",
    )
    def devstate_scalar(self):
        return self.__devstate_scalar

    @devstate_scalar.write
    def devstate_scalar(self, value):
        self.__devstate_scalar = value

    @attribute(
        dtype=("bool",),
        max_dim_x=4096,
        label="boolean_spectrum",
        doc="A boolean spectrum attribute",
    )
    def boolean_spectrum(self):
        return self.__boolean_spectrum

    @boolean_spectrum.write
    def boolean_spectrum(self, values):
        self.__boolean_spectrum = values

    @attribute(
        dtype=("double",),
        max_dim_x=4096,
        label="double_spectrum",
        doc="A double spectrum attribute",
    )
    def double_spectrum(self):
        return self.__double_spectrum

    @double_spectrum.write
    def double_spectrum(self, values):
        self.__double_spectrum = values

    @attribute(
        dtype=("float",),
        max_dim_x=4096,
        label="float_spectrum",
        doc="A float spectrum attribute",
    )
    def float_spectrum(self):
        return self.__float_spectrum

    @float_spectrum.write
    def float_spectrum(self, values):
        self.__float_spectrum = values

    @attribute(
        dtype=("int64",),
        max_dim_x=4096,
        label="long64_spectrum",
        doc="A long64 spectrum attribute (int64)",
    )
    def long64_spectrum(self):
        return self.__long64_spectrum

    @long64_spectrum.write
    def long64_spectrum(self, values):
        self.__long64_spectrum = values
        pass

    @attribute(
        dtype=("int32",),
        max_dim_x=4096,
        label="long_spectrum",
        doc="A long spectrum attribute (int32)",
    )
    def long_spectrum(self):
        return self.__long_spectrum

    @long_spectrum.write
    def long_spectrum(self, values):
        self.__long_spectrum = values

    @attribute(
        dtype=("int16",),
        max_dim_x=4096,
        label="short_spectrum",
        doc="A short spectrum attribute (int16)",
    )
    def short_spectrum(self):
        return self.__short_spectrum

    @short_spectrum.write
    def short_spectrum(self, values):
        self.__short_spectrum = values

    @attribute(
        dtype=("string",),
        max_dim_x=4096,
        label="string_spectrum",
        doc="A string spectrum attribute",
    )
    def string_spectrum(self):
        return self.__string_spectrum

    @string_spectrum.write
    def string_spectrum(self, values):
        self.__string_spectrum = values

    @attribute(
        dtype=("byte",),
        max_dim_x=4096,
        label="uchar_spectrum",
        doc="A unsigned char spectrum attribute (uint8/byte)",
    )
    def uchar_spectrum(self):
        return self.__uchar_spectrum

    @uchar_spectrum.write
    def uchar_spectrum(self, values):
        self.__uchar_spectrum = values

    @attribute(
        dtype=("uint64",),
        max_dim_x=4096,
        min_value="0",
        label="ulong64_spectrum",
        doc="A ulong64 spectrum attribute (uint64)",
    )
    def ulong64_spectrum(self):
        return self.__ulong64_spectrum

    @ulong64_spectrum.write
    def ulong64_spectrum(self, values):
        self.__ulong64_spectrum = values

    @attribute(
        dtype=("uint32",),
        max_dim_x=4096,
        min_value="0",
        label="ulong_spectrum",
        doc="A ulong spectrum attribute (uint32)",
    )
    def ulong_spectrum(self):
        return self.__ulong_spectrum

    @ulong_spectrum.write
    def ulong_spectrum(self, values):
        self.__ulong_spectrum = values

    @attribute(
        dtype=("uint16",),
        max_dim_x=4096,
        min_value="0",
        label="ushort_spectrum",
        doc="A ushort spectrum attribute (uint16)",
    )
    def ushort_spectrum(self):
        return self.__ushort_spectrum

    @ushort_spectrum.write
    def ushort_spectrum(self, values):
        self.__ushort_spectrum = values

    @attribute(
        dtype=("DevState",),
        label="device state",
        doc="A device state attribute",
    )
    def devstate_spectrum(self):
        return self.__devstate_spectrum

    @devstate_spectrum.write
    def devstate_spectrum(self, values):
        self.__devstate_spectrum = values

    @attribute(
        dtype=(("bool",),),
        max_dim_x=251,
        max_dim_y=251,
        label="boolean_image",
        doc="A boolean image attribute",
    )
    def boolean_image(self):
        return self.__boolean_image

    @boolean_image.write
    def boolean_image(self, values):
        self.__boolean_image = values

    @attribute(
        dtype=(("double",),),
        max_dim_x=251,
        max_dim_y=251,
        label="double_image",
        doc="A double image attribute",
    )
    def double_image(self):
        return self.__double_image

    @double_image.write
    def double_image(self, values):
        self.__double_image = values

    @attribute(
        dtype=(("float",),),
        max_dim_x=251,
        max_dim_y=251,
        label="float_image",
        doc="A float image attribute",
    )
    def float_image(self):
        return self.__float_image

    @float_image.write
    def float_image(self, values):
        self.__float_image = values

    @attribute(
        dtype=(("int64",),),
        max_dim_x=251,
        max_dim_y=251,
        label="long64_image",
        doc="A long64 image attribute (int64)",
    )
    def long64_image(self):
        return self.__long64_image

    @long64_image.write
    def long64_image(self, values):
        self.__long64_image = values

    @attribute(
        dtype=(("int32",),),
        max_dim_x=251,
        max_dim_y=251,
        label="long_image",
        doc="A long image attribute (int32)",
    )
    def long_image(self):
        return self.__long_image

    @long_image.write
    def long_image(self, values):
        self.__long_image = values

    @attribute(
        dtype=(("int16",),),
        max_dim_x=251,
        max_dim_y=251,
        label="short_image",
        doc="A short image attribute (int16)",
    )
    def short_image(self):
        return self.__short_image

    @short_image.write
    def short_image(self, values):
        self.__short_image = values

    @attribute(
        dtype=(("str",),),
        max_dim_x=251,
        max_dim_y=251,
        label="string_image",
        doc="A string image attribute",
    )
    def string_image(self):
        return self.__string_image

    @string_image.write
    def string_image(self, values):
        self.__string_image = values

    @attribute(
        dtype=(("byte",),),
        max_dim_x=251,
        max_dim_y=251,
        label="uchar_image",
        doc="An unsigned char image attribute (uint8/byte)",
    )
    def uchar_image(self):
        return self.__uchar_image

    @uchar_image.write
    def uchar_image(self, values):
        self.__uchar_image = values

    @attribute(
        dtype=(("uint64",),),
        max_dim_x=251,
        max_dim_y=251,
        min_value="0",
        label="ulong64_image",
        doc="An unsigned long64 image attribute (uint64)",
    )
    def ulong64_image(self):
        return self.__ulong64_image

    @ulong64_image.write
    def ulong64_image(self, values):
        self.__ulong64_image = values

    @attribute(
        dtype=(("uint32",),),
        max_dim_x=251,
        max_dim_y=251,
        min_value="0",
        label="ulong_image",
        doc="An unsigned long image attribute (unit32)",
    )
    def ulong_image(self):
        return self.__ulong_image

    @ulong_image.write
    def ulong_image(self, values):
        self.__ulong_image = values

    @attribute(
        dtype=(("uint16",),),
        max_dim_x=251,
        max_dim_y=251,
        min_value="0",
        label="ushort_image",
        doc="An unsigned short image attribute (uint16)",
    )
    def ushort_image(self):
        return self.__ushort_image

    @ushort_image.write
    def ushort_image(self, values):
        self.__ushort_image = values

#     def throw_exception(self):
#         return 0

    # --------
    # Commands
    # --------

    @command(
        dtype_in="bool",
        doc_in="A boolean value",
        dtype_out="bool",
        doc_out="Echo of the input value",
    )
    def DevBoolean(self, value):
        return value

    @command(
        dtype_in="float",
        doc_in="A DevFloat value",
        dtype_out="float",
        doc_out="Echo of the input value",
    )
    def DevFloat(self, value):
        return value

    @command(
        dtype_in="double",
        doc_in="A DevDouble value",
        dtype_out="double",
        doc_out="Echo of the input value",
    )
    def DevDouble(self, value):
        return value

    @command(
        dtype_in="int16",
        doc_in="A DevShort value",
        dtype_out="int16",
        doc_out="Echo of the input value",
    )
    def DevShort(self, value):
        return value

    @command(
        dtype_in="int32",
        doc_in="A DevLong value",
        dtype_out="int32",
        doc_out="Echo of the input value",
    )
    def DevLong(self, value):
        return value

    @command(
        dtype_in="int64",
        doc_in="A DevLong64 value",
        dtype_out="int64",
        doc_out="Echo of the input value",
    )
    def DevLong64(self, value):
        return value

    @command(
        dtype_in="uint16",
        doc_in="A DevUShort value",
        dtype_out="uint16",
        doc_out="Echo of the input value",
    )
    def DevUShort(self, value):
        return value

    @command(
        dtype_in="uint32",
        doc_in="A DevULong",
        dtype_out="uint32",
        doc_out="Echo of the input value",
    )
    def DevULong(self, value):
        return value

    @command(
        dtype_in="uint64",
        doc_in="A DevULong64 value",
        dtype_out="uint64",
        doc_out="Echo of the input value",
    )
    def DevULong64(self, value):
        return value

    @command(
        dtype_in="str",
        doc_in="A string value",
        dtype_out="str",
        doc_out="Echo of the input value",
    )
    def DevString(self, value):
        return value

    @command(
        dtype_in="DevState",
        doc_in="A DevState value",
        dtype_out="DevState",
        doc_out="Echo of the input value"
    )
    def DevState(self, value):
        return value

    @command(
        dtype_in="DevEncoded",
        doc_in="A DevEncoded value",
        dtype_out="DevEncoded",
        doc_out="Echo of the input value"
    )
    def DevEncoded(self, value):
        return value

    @command(
        dtype_in=("char",),
        doc_in="An array of characters",
        dtype_out=("char",),
        doc_out="Echo of the input values"
    )
    def DevVarCharArray(self, values):
        return values

    @command(
        dtype_in=("double",),
        doc_in="An array of double values",
        dtype_out=("double",),
        doc_out="Echo of the input values"
    )
    def DevVarDoubleArray(self, values):
        return values

    @command(
        dtype_in=("float",),
        doc_in="An array of float values",
        dtype_out=("float",),
        doc_out="Echo of the input values"
    )
    def DevVarFloatArray(self, values):
        return values

    @command(
        dtype_in=("int16",),
        doc_in="An array of short values",
        dtype_out=("int16",),
        doc_out="Echo of the input values"
    )
    def DevVarShortArray(self, values):
        return values

    @command(
        dtype_in=("int32",),
        doc_in="An array of long values",
        dtype_out=("int32",),
        doc_out="Echo of the input values"
    )
    def DevVarLongArray(self, values):
        return [0]

    @command(
        dtype_in=("int64",),
        doc_in="An array of long64 values",
        dtype_out=("int64",),
        doc_out="Echo of the input values",
    )
    def DevVarLong64Array(self, values):
        return values

    @command(
        dtype_in=("uint16",),
        doc_in="An array of unsigned short values",
        dtype_out=("uint16",),
        doc_out="Echo of the input values"
    )
    def DevVarUShortArray(self, values):
        return values

    @command(
        dtype_in=("uint32",),
        doc_in="An array of unsigned long values",
        dtype_out=("uint32",),
        doc_out="Echo of the input values"
    )
    def DevVarULongArray(self, values):
        return values

    @command(
        dtype_in=("uint64",),
        doc_in="An array of unsigned long64 values",
        dtype_out=("uint64",),
        doc_out="Echo of the input values"
    )
    def DevVarULong64Array(self, values):
        return values

    @command(
        dtype_in=("str",),
        doc_in="An array of strings",
        dtype_out=("str",),
        doc_out="Echo of the input values"
    )
    def DevVarStringArray(self, values):
        return values

    @command(
        dtype_in="DevVarLongStringArray",
        doc_in="A tuple array of longs & strings",
        dtype_out="DevVarLongStringArray",
        doc_out="Echo of the input values"
    )
    def DevVarLongStringArray(self, values):
        return values

    @command(
        dtype_in="DevVarDoubleStringArray",
        doc_in="A tuple array of doubles & strings",
        dtype_out="DevVarDoubleStringArray",
        doc_out="Echo of the input values"
    )
    def DevVarDoubleStringArray(self, values):
        return values

    @command
    def DevVoid(self):
        pass

    @command(
        dtype_in="DevState",
        doc_in="Set the state of the server",
    )
    def ChangeState(self, new_state):
        self.set_state(new_state)

    @command(
        dtype_in="bool",
        doc_in="Push change events == true else false",
    )
    def PushScalarChangeEvents(self, enabled):
        self.__push_scalar_change_events = enabled

    @command(
        dtype_in="bool",
        doc_in="Push archive events == true else false",
    )
    def PushScalarArchiveEvents(self, enabled):
        self.__push_scalar_archive_events = enabled

    @command(
        dtype_in="bool",
        doc_in="Push pipe events == true else false",
    )
    def PushPipeEvents(self, enabled):
        self.__push_pipe_events = enabled

    @command
    def Randomise(self):
        if self.__generate_task is None:
            with self.__lock:
                self.__generate_task = gevent.spawn(self.__generate_random_data)
        else:
            self.__generate_task = None

    def __generate_random_data(self):
        while True:
            self.__double_scalar_ro = np.random.uniform(self.double_minmax[0],
                                                        self.double_minmax[1])
            self.__float_scalar_ro = np.random.uniform(self.float_minmax[0],
                                                       self.float_minmax[1])
            self.__long64_scalar_ro = np.random.randint(self.long64_minmax[0],
                                                        self.long64_minmax[1])
            self.__long_scalar_ro = np.random.randint(self.long_minmax[0],
                                                      self.long_minmax[1])
            self.__short_scalar_ro = np.random.randint(self.short_minmax[0],
                                                       self.short_minmax[1])
            self.__ulong64_scalar_ro = np.random.randint(self.ulong64_minmax[0],
                                                         self.ulong64_minmax[1])
            self.__ulong_scalar_ro = np.random.randint(self.ulong_minmax[0],
                                                       self.ulong_minmax[1])
            self.__ushort_scalar_ro = np.random.randint(self.ushort_minmax[0],
                                                        self.ushort_minmax[1])
            if self.__push_scalar_change_events:
                self.push_change_event("double_scalar_ro", self.__double_scalar_ro)
            if self.__push_scalar_archive_events:
                self.push_archive_event("double_scalar_ro", self.__double_scalar_ro)
            if self.__push_pipe_events:
                self.push_pipe_event("TestPipe", self.__blob)
            gevent.sleep(1.0)

    # --------
    # pipes
    # --------

    @pipe(label="Test pipe", description="This is a test pipe")
    def TestPipe(self):
        print("Reading TestPipe blob")
        return self.__blob

    @TestPipe.write
    def TestPipe(self, blob):
        print("Writing blob")
        self.__blob = blob
        print(blob)


# ----------
# Run server
# ----------


if __name__ == "__main__":
    TangoTest.run_server(green_mode=GreenMode.Gevent)
