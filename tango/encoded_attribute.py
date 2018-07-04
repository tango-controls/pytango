# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

"""
This is an internal PyTango module.
"""

__all__ = ("encoded_attribute_init",)

__docformat__ = "restructuredtext"

from ._tango import EncodedAttribute, ExtractAs, _ImageFormat
from ._tango import constants

from .utils import is_pure_str, is_seq

if constants.NUMPY_SUPPORT:
    try:
        import numpy

        np = numpy
    except:
        np = None
else:
    np = None

_allowed_extract = (
    ExtractAs.Numpy, ExtractAs.String, ExtractAs.Tuple, ExtractAs.List)


def __EncodedAttribute_encode_jpeg_gray8(self, gray8, width=0, height=0, quality=100.0):
    """Encode a 8 bit grayscale image as JPEG format

           :param gray8: an object containning image information
           :type gray8: :py:obj:`str` or :class:`numpy.ndarray` or seq< seq<element> >
           :param width: image width. **MUST** be given if gray8 is a string or
                         if it is a :class:`numpy.ndarray` with ndims != 2.
                         Otherwise it is calculated internally.
           :type width: :py:obj:`int`
           :param height: image height. **MUST** be given if gray8 is a string
                          or if it is a :class:`numpy.ndarray` with ndims != 2.
                          Otherwise it is calculated internally.
           :type height: :py:obj:`int`
           :param quality: Quality of JPEG (0=poor quality 100=max quality) (default is 100.0)
           :type quality: :py:obj:`float`

       .. note::
           When :class:`numpy.ndarray` is given:

               - gray8 **MUST** be CONTIGUOUS, ALIGNED
               - if gray8.ndims != 2, width and height **MUST** be given and
                 gray8.nbytes **MUST** match width*height
               - if gray8.ndims == 2, gray8.itemsize **MUST** be 1 (typically,
                 gray8.dtype is one of `numpy.dtype.byte`, `numpy.dtype.ubyte`,
                 `numpy.dtype.int8` or `numpy.dtype.uint8`)

       Example::

           def read_myattr(self, attr):
               enc = tango.EncodedAttribute()
               data = numpy.arange(100, dtype=numpy.byte)
               data = numpy.array((data,data,data))
               enc.encode_jpeg_gray8(data)
               attr.set_value(enc)
    """
    self._generic_encode_gray8(gray8, width=width, height=height, quality=quality, format=_ImageFormat.JpegImage)


def __EncodedAttribute_encode_gray8(self, gray8, width=0, height=0):
    """Encode a 8 bit grayscale image (no compression)

           :param gray8: an object containning image information
           :type gray8: :py:obj:`str` or :class:`numpy.ndarray` or seq< seq<element> >
           :param width: image width. **MUST** be given if gray8 is a string or
                         if it is a :class:`numpy.ndarray` with ndims != 2.
                         Otherwise it is calculated internally.
           :type width: :py:obj:`int`
           :param height: image height. **MUST** be given if gray8 is a string
                          or if it is a :class:`numpy.ndarray` with ndims != 2.
                          Otherwise it is calculated internally.
           :type height: :py:obj:`int`

       .. note::
           When :class:`numpy.ndarray` is given:

               - gray8 **MUST** be CONTIGUOUS, ALIGNED
               - if gray8.ndims != 2, width and height **MUST** be given and
                 gray8.nbytes **MUST** match width*height
               - if gray8.ndims == 2, gray8.itemsize **MUST** be 1 (typically,
                 gray8.dtype is one of `numpy.dtype.byte`, `numpy.dtype.ubyte`,
                 `numpy.dtype.int8` or `numpy.dtype.uint8`)

       Example::

           def read_myattr(self, attr):
               enc = tango.EncodedAttribute()
               data = numpy.arange(100, dtype=numpy.byte)
               data = numpy.array((data,data,data))
               enc.encode_gray8(data)
               attr.set_value(enc)
    """
    self._generic_encode_gray8(gray8, width=width, height=height, format=_ImageFormat.RawImage)


def __EncodedAttribute_generic_encode_gray8(self, gray8, width=0, height=0, quality=0, format=_ImageFormat.RawImage):
    """Internal usage only"""
    if not is_seq(gray8):
        raise TypeError("Expected sequence (str, numpy.ndarray, list, tuple "
                        "or bytearray) as first argument")

    is_str = is_pure_str(gray8)
    if is_str:
        if not width or not height:
            raise ValueError("When giving a string as data, you must also "
                             "supply width and height")

    if np and isinstance(gray8, np.ndarray):
        if gray8.ndim != 2:
            if not width or not height:
                raise ValueError("When giving a non 2D numpy array, width and "
                                 "height must be supplied")
            if gray8.nbytes != width * height:
                raise ValueError("numpy array size mismatch")
        else:
            if gray8.itemsize != 1:
                raise TypeError("Expected numpy array with itemsize == 1")
        if not gray8.flags.c_contiguous:
            raise TypeError("Currently, only contiguous, aligned numpy arrays "
                            "are supported")
        if not gray8.flags.aligned:
            raise TypeError("Currently, only contiguous, aligned numpy arrays "
                            "are supported")

    if not is_str and (not width or not height):
        height = len(gray8)
        if height < 1:
            raise IndexError("Expected sequence with at least one row")

        row0 = gray8[0]
        if not is_seq(row0):
            raise IndexError("Expected sequence (str, numpy.ndarray, list, tuple or "
                             "bytearray) inside a sequence")
        width = len(row0)

    if format == _ImageFormat.RawImage:
        self._encode_gray8(gray8, width, height)
    elif format == _ImageFormat.JpegImage:
        self._encode_jpeg_gray8(gray8, width, height, quality)


def __EncodedAttribute_encode_gray16(self, gray16, width=0, height=0):
    """Encode a 16 bit grayscale image (no compression)

           :param gray16: an object containning image information
           :type gray16: :py:obj:`str` or :py:obj:`buffer` or :class:`numpy.ndarray` or seq< seq<element> >
           :param width: image width. **MUST** be given if gray16 is a string or
                         if it is a :class:`numpy.ndarray` with ndims != 2.
                         Otherwise it is calculated internally.
           :type width: :py:obj:`int`
           :param height: image height. **MUST** be given if gray16 is a string
                          or if it is a :class:`numpy.ndarray` with ndims != 2.
                          Otherwise it is calculated internally.
           :type height: :py:obj:`int`

       .. note::
           When :class:`numpy.ndarray` is given:

               - gray16 **MUST** be CONTIGUOUS, ALIGNED
               - if gray16.ndims != 2, width and height **MUST** be given and
                 gray16.nbytes/2 **MUST** match width*height
               - if gray16.ndims == 2, gray16.itemsize **MUST** be 2 (typically,
                 gray16.dtype is one of `numpy.dtype.int16`, `numpy.dtype.uint16`,
                 `numpy.dtype.short` or `numpy.dtype.ushort`)

       Example::

           def read_myattr(self, attr):
               enc = tango.EncodedAttribute()
               data = numpy.arange(100, dtype=numpy.int16)
               data = numpy.array((data,data,data))
               enc.encode_gray16(data)
               attr.set_value(enc)
    """
    if not is_seq(gray16):
        raise TypeError("Expected sequence (str, numpy.ndarray, list, tuple "
                        "or bytearray) as first argument")

    is_str = is_pure_str(gray16)
    if is_str:
        if not width or not height:
            raise ValueError("When giving a string as data, you must also "
                             "supply width and height")

    if np and isinstance(gray16, np.ndarray):
        if gray16.ndim != 2:
            if not width or not height:
                raise ValueError("When giving a non 2D numpy array, width and "
                                 "height must be supplied")
            if gray16.nbytes / 2 != width * height:
                raise ValueError("numpy array size mismatch")
        else:
            if gray16.itemsize != 2:
                raise TypeError("Expected numpy array with itemsize == 2")
        if not gray16.flags.c_contiguous:
            raise TypeError("Currently, only contiguous, aligned numpy arrays "
                            "are supported")
        if not gray16.flags.aligned:
            raise TypeError("Currently, only contiguous, aligned numpy arrays "
                            "are supported")

    if not is_str and (not width or not height):
        height = len(gray16)
        if height < 1:
            raise IndexError("Expected sequence with at least one row")

        row0 = gray16[0]
        if not is_seq(row0):
            raise IndexError("Expected sequence (str, numpy.ndarray, list, tuple or "
                             "bytearray) inside a sequence")
        width = len(row0)
        if is_pure_str(row0) or type(row0) == bytearray:
            width /= 2

    self._encode_gray16(gray16, width, height)


def __EncodedAttribute_encode_jpeg_rgb24(self, rgb24, width=0, height=0, quality=100.0):
    """Encode a 24 bit rgb color image as JPEG format.

           :param rgb24: an object containning image information
           :type rgb24: :py:obj:`str` or :class:`numpy.ndarray` or seq< seq<element> >
           :param width: image width. **MUST** be given if rgb24 is a string or
                         if it is a :class:`numpy.ndarray` with ndims != 3.
                         Otherwise it is calculated internally.
           :type width: :py:obj:`int`
           :param height: image height. **MUST** be given if rgb24 is a string
                          or if it is a :class:`numpy.ndarray` with ndims != 3.
                          Otherwise it is calculated internally.
           :type height: :py:obj:`int`
           :param quality: Quality of JPEG (0=poor quality 100=max quality) (default is 100.0)
           :type quality: :py:obj:`float`

       .. note::
           When :class:`numpy.ndarray` is given:

               - rgb24 **MUST** be CONTIGUOUS, ALIGNED
               - if rgb24.ndims != 3, width and height **MUST** be given and
                 rgb24.nbytes/3 **MUST** match width*height
               - if rgb24.ndims == 3, rgb24.itemsize **MUST** be 1 (typically,
                 rgb24.dtype is one of `numpy.dtype.byte`, `numpy.dtype.ubyte`,
                 `numpy.dtype.int8` or `numpy.dtype.uint8`) and shape **MUST** be
                 (height, width, 3)

       Example::

           def read_myattr(self, attr):
               enc = tango.EncodedAttribute()
               # create an 'image' where each pixel is R=0x01, G=0x01, B=0x01
               arr = numpy.ones((10,10,3), dtype=numpy.uint8)
               enc.encode_jpeg_rgb24(data)
               attr.set_value(enc)
    """
    self._generic_encode_rgb24(rgb24, width=width, height=height, quality=quality, format=_ImageFormat.JpegImage)


def __EncodedAttribute_encode_rgb24(self, rgb24, width=0, height=0):
    """Encode a 24 bit color image (no compression)

           :param rgb24: an object containning image information
           :type rgb24: :py:obj:`str` or :class:`numpy.ndarray` or seq< seq<element> >
           :param width: image width. **MUST** be given if rgb24 is a string or
                         if it is a :class:`numpy.ndarray` with ndims != 3.
                         Otherwise it is calculated internally.
           :type width: :py:obj:`int`
           :param height: image height. **MUST** be given if rgb24 is a string
                          or if it is a :class:`numpy.ndarray` with ndims != 3.
                          Otherwise it is calculated internally.
           :type height: :py:obj:`int`

       .. note::
           When :class:`numpy.ndarray` is given:

               - rgb24 **MUST** be CONTIGUOUS, ALIGNED
               - if rgb24.ndims != 3, width and height **MUST** be given and
                 rgb24.nbytes/3 **MUST** match width*height
               - if rgb24.ndims == 3, rgb24.itemsize **MUST** be 1 (typically,
                 rgb24.dtype is one of `numpy.dtype.byte`, `numpy.dtype.ubyte`,
                 `numpy.dtype.int8` or `numpy.dtype.uint8`) and shape **MUST** be
                 (height, width, 3)

       Example::

           def read_myattr(self, attr):
               enc = tango.EncodedAttribute()
               # create an 'image' where each pixel is R=0x01, G=0x01, B=0x01
               arr = numpy.ones((10,10,3), dtype=numpy.uint8)
               enc.encode_rgb24(data)
               attr.set_value(enc)
    """
    self._generic_encode_rgb24(rgb24, width=width, height=height, format=_ImageFormat.RawImage)


def __EncodedAttribute_generic_encode_rgb24(self, rgb24, width=0, height=0, quality=0, format=_ImageFormat.RawImage):
    """Internal usage only"""
    if not is_seq(rgb24):
        raise TypeError("Expected sequence (str, numpy.ndarray, list, tuple "
                        "or bytearray) as first argument")

    is_str = is_pure_str(rgb24)
    if is_str:
        if not width or not height:
            raise ValueError("When giving a string as data, you must also "
                             "supply width and height")

    if np and isinstance(rgb24, np.ndarray):
        if rgb24.ndim != 3:
            if not width or not height:
                raise ValueError("When giving a non 2D numpy array, width and "
                                 "height must be supplied")
            if rgb24.nbytes / 3 != width * height:
                raise ValueError("numpy array size mismatch")
        else:
            if rgb24.itemsize != 1:
                raise TypeError("Expected numpy array with itemsize == 1")
        if not rgb24.flags.c_contiguous:
            raise TypeError("Currently, only contiguous, aligned numpy arrays "
                            "are supported")
        if not rgb24.flags.aligned:
            raise TypeError("Currently, only contiguous, aligned numpy arrays "
                            "are supported")

    if not is_str and (not width or not height):
        height = len(rgb24)
        if height < 1:
            raise IndexError("Expected sequence with at least one row")

        row0 = rgb24[0]
        if not is_seq(row0):
            raise IndexError("Expected sequence (str, numpy.ndarray, list, tuple or "
                             "bytearray) inside a sequence")
        width = len(row0)
        if is_pure_str(row0) or type(row0) == bytearray:
            width /= 3
    if format == _ImageFormat.RawImage:
        self._encode_rgb24(rgb24, width, height)
    elif format == _ImageFormat.JpegImage:
        self._encode_jpeg_rgb24(rgb24, width, height, quality)


def __EncodedAttribute_encode_jpeg_rgb32(self, rgb32, width=0, height=0, quality=100.0):
    """Encode a 32 bit rgb color image as JPEG format.

           :param rgb32: an object containning image information
           :type rgb32: :py:obj:`str` or :class:`numpy.ndarray` or seq< seq<element> >
           :param width: image width. **MUST** be given if rgb32 is a string or
                         if it is a :class:`numpy.ndarray` with ndims != 2.
                         Otherwise it is calculated internally.
           :type width: :py:obj:`int`
           :param height: image height. **MUST** be given if rgb32 is a string
                          or if it is a :class:`numpy.ndarray` with ndims != 2.
                          Otherwise it is calculated internally.
           :type height: :py:obj:`int`

       .. note::
           When :class:`numpy.ndarray` is given:

               - rgb32 **MUST** be CONTIGUOUS, ALIGNED
               - if rgb32.ndims != 2, width and height **MUST** be given and
                 rgb32.nbytes/4 **MUST** match width*height
               - if rgb32.ndims == 2, rgb32.itemsize **MUST** be 4 (typically,
                 rgb32.dtype is one of `numpy.dtype.int32`, `numpy.dtype.uint32`)

       Example::

           def read_myattr(self, attr):
               enc = tango.EncodedAttribute()
               data = numpy.arange(100, dtype=numpy.int32)
               data = numpy.array((data,data,data))
               enc.encode_jpeg_rgb32(data)
               attr.set_value(enc)
    """
    if not is_seq(rgb32):
        raise TypeError("Expected sequence (str, numpy.ndarray, list, tuple "
                        "or bytearray) as first argument")

    is_str = is_pure_str(rgb32)
    if is_str:
        if not width or not height:
            raise ValueError("When giving a string as data, you must also "
                             "supply width and height")

    if np and isinstance(rgb32, np.ndarray):
        if rgb32.ndim != 2:
            if not width or not height:
                raise ValueError("When giving a non 2D numpy array, width and "
                                 "height must be supplied")
            if rgb32.nbytes / 4 != width * height:
                raise ValueError("numpy array size mismatch")
        else:
            if rgb32.itemsize != 4:
                raise TypeError("Expected numpy array with itemsize == 4")
        if not rgb32.flags.c_contiguous:
            raise TypeError("Currently, only contiguous, aligned numpy arrays "
                            "are supported")
        if not rgb32.flags.aligned:
            raise TypeError("Currently, only contiguous, aligned numpy arrays "
                            "are supported")

    if not is_str and (not width or not height):
        height = len(rgb32)
        if height < 1:
            raise IndexError("Expected sequence with at least one row")

        row0 = rgb32[0]
        if not is_seq(row0):
            raise IndexError("Expected sequence (str, numpy.ndarray, list, tuple or "
                             "bytearray) inside a sequence")
        width = len(row0)
        if is_pure_str(row0) or type(row0) == bytearray:
            width /= 4

    self._encode_jpeg_rgb32(rgb32, width, height, quality)


def __EncodedAttribute_decode_gray8(self, da, extract_as=ExtractAs.Numpy):
    """Decode a 8 bits grayscale image (JPEG_GRAY8 or GRAY8) and returns a 8 bits gray scale image.

        :param da: :class:`DeviceAttribute` that contains the image
        :type da: :class:`DeviceAttribute`
        :param extract_as: defaults to ExtractAs.Numpy
        :type extract_as: ExtractAs
        :return: the decoded data

        - In case String string is choosen as extract method, a tuple is returned:
            width<int>, height<int>, buffer<str>
        - In case Numpy is choosen as extract method, a :class:`numpy.ndarray` is
          returned with ndim=2, shape=(height, width) and dtype=numpy.uint8.
        - In case Tuple or List are choosen, a tuple<tuple<int>> or list<list<int>>
          is returned.

       .. warning::
           The PyTango calls that return a :class:`DeviceAttribute`
           (like :meth:`DeviceProxy.read_attribute` or :meth:`DeviceProxy.command_inout`)
           automatically extract the contents by default. This method requires
           that the given :class:`DeviceAttribute` is obtained from a
           call which **DOESN'T** extract the contents. Example::

               dev = tango.DeviceProxy("a/b/c")
               da = dev.read_attribute("my_attr", extract_as=tango.ExtractAs.Nothing)
               enc = tango.EncodedAttribute()
               data = enc.decode_gray8(da)
    """
    if hasattr(da, 'value'):
        raise TypeError("DeviceAttribute argument must have been obtained from "
                        "a call which doesn't extract the contents")
    if extract_as not in _allowed_extract:
        raise TypeError("extract_as must be one of Numpy, String, Tuple, List")
    return self._decode_gray8(da, extract_as)


def __EncodedAttribute_decode_gray16(self, da, extract_as=ExtractAs.Numpy):
    """Decode a 16 bits grayscale image (GRAY16) and returns a 16 bits gray scale image.

        :param da: :class:`DeviceAttribute` that contains the image
        :type da: :class:`DeviceAttribute`
        :param extract_as: defaults to ExtractAs.Numpy
        :type extract_as: ExtractAs
        :return: the decoded data

        - In case String string is choosen as extract method, a tuple is returned:
            width<int>, height<int>, buffer<str>
        - In case Numpy is choosen as extract method, a :class:`numpy.ndarray` is
          returned with ndim=2, shape=(height, width) and dtype=numpy.uint16.
        - In case Tuple or List are choosen, a tuple<tuple<int>> or list<list<int>>
          is returned.

       .. warning::
           The PyTango calls that return a :class:`DeviceAttribute`
           (like :meth:`DeviceProxy.read_attribute` or :meth:`DeviceProxy.command_inout`)
           automatically extract the contents by default. This method requires
           that the given :class:`DeviceAttribute` is obtained from a
           call which **DOESN'T** extract the contents. Example::

               dev = tango.DeviceProxy("a/b/c")
               da = dev.read_attribute("my_attr", extract_as=tango.ExtractAs.Nothing)
               enc = tango.EncodedAttribute()
               data = enc.decode_gray16(da)
    """
    if hasattr(da, 'value'):
        raise TypeError("DeviceAttribute argument must have been obtained from "
                        "a call which doesn't extract the contents")
    if extract_as not in _allowed_extract:
        raise TypeError("extract_as must be one of Numpy, String, Tuple, List")
    return self._decode_gray16(da, extract_as)


def __EncodedAttribute_decode_rgb32(self, da, extract_as=ExtractAs.Numpy):
    """Decode a color image (JPEG_RGB or RGB24) and returns a 32 bits RGB image.

        :param da: :class:`DeviceAttribute` that contains the image
        :type da: :class:`DeviceAttribute`
        :param extract_as: defaults to ExtractAs.Numpy
        :type extract_as: ExtractAs
        :return: the decoded data

        - In case String string is choosen as extract method, a tuple is returned:
            width<int>, height<int>, buffer<str>
        - In case Numpy is choosen as extract method, a :class:`numpy.ndarray` is
          returned with ndim=2, shape=(height, width) and dtype=numpy.uint32.
        - In case Tuple or List are choosen, a tuple<tuple<int>> or list<list<int>>
          is returned.

       .. warning::
           The PyTango calls that return a :class:`DeviceAttribute`
           (like :meth:`DeviceProxy.read_attribute` or :meth:`DeviceProxy.command_inout`)
           automatically extract the contents by default. This method requires
           that the given :class:`DeviceAttribute` is obtained from a
           call which **DOESN'T** extract the contents. Example::

               dev = tango.DeviceProxy("a/b/c")
               da = dev.read_attribute("my_attr", extract_as=tango.ExtractAs.Nothing)
               enc = tango.EncodedAttribute()
               data = enc.decode_rgb32(da)
    """
    if hasattr(da, 'value'):
        raise TypeError("DeviceAttribute argument must have been obtained from "
                        "a call which doesn't extract the contents")
    if extract_as not in _allowed_extract:
        raise TypeError("extract_as must be one of Numpy, String, Tuple, List")
    return self._decode_rgb32(da, extract_as)


def __init_EncodedAttribute():
    EncodedAttribute._generic_encode_gray8 = __EncodedAttribute_generic_encode_gray8
    EncodedAttribute.encode_gray8 = __EncodedAttribute_encode_gray8
    EncodedAttribute.encode_jpeg_gray8 = __EncodedAttribute_encode_jpeg_gray8
    EncodedAttribute.encode_gray16 = __EncodedAttribute_encode_gray16
    EncodedAttribute._generic_encode_rgb24 = __EncodedAttribute_generic_encode_rgb24
    EncodedAttribute.encode_rgb24 = __EncodedAttribute_encode_rgb24
    EncodedAttribute.encode_jpeg_rgb24 = __EncodedAttribute_encode_jpeg_rgb24
    EncodedAttribute.encode_jpeg_rgb32 = __EncodedAttribute_encode_jpeg_rgb32
    EncodedAttribute.decode_gray8 = __EncodedAttribute_decode_gray8
    EncodedAttribute.decode_gray16 = __EncodedAttribute_decode_gray16
    EncodedAttribute.decode_rgb32 = __EncodedAttribute_decode_rgb32


def __doc_EncodedAttribute():
    pass


def encoded_attribute_init(doc=True):
    __init_EncodedAttribute()
    if doc:
        __doc_EncodedAttribute()
