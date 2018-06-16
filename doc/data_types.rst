.. currentmodule:: tango

.. _pytango-data-types:

Data types
==========

This chapter describes the mapping of data types between Python and Tango.

Tango has more data types than Python which is more dynamic. The input and
output values of the commands are translated according to the array below.
Note that if PyTango is compiled with :py:mod:`numpy` support the numpy type
will be the used for the input arguments. Also, it is recomended to use numpy
arrays of the appropiate type for output arguments as well, as they tend to be
much more efficient.

**For scalar types (SCALAR)**

+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|   Tango data type       |              Python 2.x type                                              |              Python 3.x type (*New in PyTango 8.0*)                       |
+=========================+===========================================================================+===========================================================================+
|          DEV_VOID       |                    No data                                                |                    No data                                                |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|       DEV_BOOLEAN       | :py:obj:`bool`                                                            | :py:obj:`bool`                                                            |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|         DEV_SHORT       | :py:obj:`int`                                                             | :py:obj:`int`                                                             |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|         DEV_LONG        | :py:obj:`int`                                                             | :py:obj:`int`                                                             |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|        DEV_LONG64       | - :py:obj:`long` (on a 32 bits computer)                                  | :py:obj:`int`                                                             |
|                         | - :py:obj:`int` (on a 64 bits computer)                                   |                                                                           |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|         DEV_FLOAT       | :py:obj:`float`                                                           | :py:obj:`float`                                                           |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|       DEV_DOUBLE        | :py:obj:`float`                                                           | :py:obj:`float`                                                           |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|        DEV_USHORT       | :py:obj:`int`                                                             | :py:obj:`int`                                                             |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|        DEV_ULONG        | :py:obj:`int`                                                             | :py:obj:`int`                                                             |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|        DEV_ULONG64      | * :py:obj:`long` (on a 32 bits computer)                                  | :py:obj:`int`                                                             |
|                         | * :py:obj:`int` (on a 64 bits computer)                                   |                                                                           |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|        DEV_STRING       | :py:obj:`str`                                                             | :py:obj:`str` (decoded with *latin-1*, aka *ISO-8859-1*)                  |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | sequence of two elements:                                                 | sequence of two elements:                                                 |
| DEV_ENCODED             |                                                                           |                                                                           |
| (*New in PyTango 8.0*)  | 0. :py:obj:`str`                                                          | 0. :py:obj:`str` (decoded with *latin-1*, aka *ISO-8859-1*)               |
|                         | 1. :py:obj:`bytes` (for any value of *extract_as*)                        | 1. :py:obj:`bytes` (for any value of *extract_as*, except String.         |
|                         |                                                                           |    In this case it is :py:obj:`str` (decoded with default python          |
|                         |                                                                           |    encoding *utf-8*))                                                     |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | * :py:obj:`int` (for value)                                               | * :py:obj:`int` (for value)                                               |
|                         | * :py:class:`list` <:py:obj:`str`> (for enum_labels)                      | * :py:class:`list` <:py:obj:`str`>  (for enum_labels)                     |
|        DEV_ENUM         |                                                                           |                                                                           |
| (*New in PyTango 9.0*)  | Note:  direct attribute access via DeviceProxy will return enumerated     | Note:  direct attribute access via DeviceProxy will return enumerated     |
|                         |        type :py:obj:`enum.IntEnum`.                                       |        type :py:obj:`enum.IntEnum`.                                       |
|                         |        This type uses the package enum34.                                 |        Python < 3.4, uses the package enum34.                             |
|                         |                                                                           |        Python >= 3.4, uses standard package enum.                         |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+

**For array types (SPECTRUM/IMAGE)**

+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|    Tango data type      |   ExtractAs     |                          Data type (Python 2.x)                           |             Data type (Python 3.x) (*New in PyTango 8.0*)                 |
+=========================+=================+===========================================================================+===========================================================================+
| DEVVAR_CHARARRAY        | Numpy           | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint8`)                  | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint8`)                  |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Bytes           | :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)                 | :py:obj:`bytes`                                                           |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | ByteArray       | :py:obj:`bytearray`                                                       | :py:obj:`bytearray`                                                       |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | String          | :py:obj:`str`                                                             | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | List            | :py:class:`list` <:py:obj:`int`>                                          | :py:class:`list` <:py:obj:`int`>                                          |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Tuple           | :py:class:`tuple` <:py:obj:`int`>                                         | :py:class:`tuple` <:py:obj:`int`>                                         |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| DEVVAR_SHORTARRAY       | Numpy           | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint16`)                 | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint16`)                 |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_SHORT + SPECTRUM)  | Bytes           | :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)                 | :py:obj:`bytes`                                                           |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_SHORT + IMAGE)     | ByteArray       | :py:obj:`bytearray`                                                       | :py:obj:`bytearray`                                                       |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | String          | :py:obj:`str`                                                             | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | List            | :py:class:`list` <:py:obj:`int`>                                          | :py:class:`list` <:py:obj:`int`>                                          |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Tuple           | :py:class:`tuple` <:py:obj:`int`>                                         | :py:class:`tuple` <:py:obj:`int`>                                         |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| DEVVAR_LONGARRAY        | Numpy           | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint32`)                 | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint32`)                 |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_LONG + SPECTRUM)   | Bytes           | :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)                 | :py:obj:`bytes`                                                           |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_LONG + IMAGE)      | ByteArray       | :py:obj:`bytearray`                                                       | :py:obj:`bytearray`                                                       |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | String          | :py:obj:`str`                                                             | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | List            | :py:class:`list` <:py:obj:`int`>                                          | :py:class:`list` <:py:obj:`int`>                                          |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Tuple           | :py:class:`tuple` <:py:obj:`int`>                                         | :py:class:`tuple` <:py:obj:`int`>                                         |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| DEVVAR_LONG64ARRAY      | Numpy           | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint64`)                 | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint64`)                 |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_LONG64 + SPECTRUM) | Bytes           | :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)                 | :py:obj:`bytes`                                                           |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_LONG64 + IMAGE)    | ByteArray       | :py:obj:`bytearray`                                                       | :py:obj:`bytearray`                                                       |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | String          | :py:obj:`str`                                                             | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | List            | :py:class:`list` <int (64 bits) / long (32 bits)>                         | :py:class:`list` <:py:obj:`int`>                                          |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Tuple           | :py:class:`tuple` <int (64 bits) / long (32 bits)>                        | :py:class:`tuple` <:py:obj:`int`>                                         |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| DEVVAR_FLOATARRAY       | Numpy           | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float32`)                | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float32`)                |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_FLOAT + SPECTRUM)  | Bytes           | :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)                 | :py:obj:`bytes`                                                           |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_FLOAT + IMAGE)     | ByteArray       | :py:obj:`bytearray`                                                       | :py:obj:`bytearray`                                                       |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | String          | :py:obj:`str`                                                             | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | List            | :py:class:`list` <:py:obj:`float`>                                        | :py:class:`list` <:py:obj:`float`>                                        |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Tuple           | :py:class:`tuple` <:py:obj:`float`>                                       | :py:class:`tuple` <:py:obj:`float`>                                       |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| DEVVAR_DOUBLEARRAY      | Numpy           | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float64`)                | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float64`)                |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_DOUBLE + SPECTRUM) | Bytes           | :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)                 | :py:obj:`bytes`                                                           |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_DOUBLE + IMAGE)    | ByteArray       | :py:obj:`bytearray`                                                       | :py:obj:`bytearray`                                                       |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | String          | :py:obj:`str`                                                             | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | List            | :py:class:`list` <:py:obj:`float`>                                        | :py:class:`list` <:py:obj:`float`>                                        |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Tuple           | :py:class:`tuple` <:py:obj:`float`>                                       | :py:class:`tuple` <:py:obj:`float`>                                       |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| DEVVAR_USHORTARRAY      | Numpy           | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint16`)                 | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint16`)                 |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_USHORT + SPECTRUM) | Bytes           | :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)                 | :py:obj:`bytes`                                                           |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_USHORT + IMAGE)    | ByteArray       | :py:obj:`bytearray`                                                       | :py:obj:`bytearray`                                                       |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | String          | :py:obj:`str`                                                             | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | List            | :py:class:`list` <:py:obj:`int`>                                          | :py:class:`list` <:py:obj:`int`>                                          |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Tuple           | :py:class:`tuple` <:py:obj:`int`>                                         | :py:class:`tuple` <:py:obj:`int`>                                         |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| DEVVAR_ULONGARRAY       | Numpy           | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint32`)                 | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint32`)                 |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_ULONG + SPECTRUM)  | Bytes           | :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)                 | :py:obj:`bytes`                                                           |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_ULONG + IMAGE)     | ByteArray       | :py:obj:`bytearray`                                                       | :py:obj:`bytearray`                                                       |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | String          | :py:obj:`str`                                                             | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | List            | :py:class:`list` <:py:obj:`int`>                                          | :py:class:`list` <:py:obj:`int`>                                          |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Tuple           | :py:class:`tuple` <:py:obj:`int`>                                         | :py:class:`tuple` <:py:obj:`int`>                                         |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| DEVVAR_ULONG64ARRAY     | Numpy           | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint64`)                 | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint64`)                 |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_ULONG64 + SPECTRUM)| Bytes           | :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)                 | :py:obj:`bytes`                                                           |
| or                      +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| (DEV_ULONG64 + IMAGE)   | ByteArray       | :py:obj:`bytearray`                                                       | :py:obj:`bytearray`                                                       |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | String          | :py:obj:`str`                                                             | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | List            | :py:class:`list` <int (64 bits) / long (32 bits)>                         | :py:class:`list` <:py:obj:`int`>                                          |
|                         +-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | Tuple           | :py:class:`tuple` <int (64 bits) / long (32 bits)>                        | :py:class:`tuple` <:py:obj:`int`>                                         |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
| DEVVAR_STRINGARRAY      |                 | sequence<:py:obj:`str`>                                                   | sequence<:py:obj:`str`>                                                   |
| or                      |                 |                                                                           | (decoded with *latin-1*, aka *ISO-8859-1*)                                |
| (DEV_STRING + SPECTRUM) |                 |                                                                           |                                                                           |
| or                      |                 |                                                                           |                                                                           |
| (DEV_STRING + IMAGE)    |                 |                                                                           |                                                                           |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         |                 | sequence of two elements:                                                 | sequence of two elements:                                                 |
|  DEV_LONGSTRINGARRAY    |                 |                                                                           |                                                                           |
|                         |                 | 0. :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.int32`) or            | 0. :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.int32`) or            |
|                         |                 |    sequence<:py:obj:`int`>                                                |    sequence<:py:obj:`int`>                                                |
|                         |                 | 1. sequence<:py:obj:`str`>                                                | 1.  sequence<:py:obj:`str`> (decoded with *latin-1*, aka *ISO-8859-1*)    |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         |                 | sequence of two elements:                                                 | sequence of two elements:                                                 |
|  DEV_DOUBLESTRINGARRAY  |                 |                                                                           |                                                                           |
|                         |                 | 0. :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float64`) or          | 0. :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float64`) or          |
|                         |                 |    sequence<:py:obj:`int`>                                                |    sequence<:py:obj:`int`>                                                |
|                         |                 | 1. sequence<:py:obj:`str`>                                                | 1. sequence<:py:obj:`str`> (decoded with *latin-1*, aka *ISO-8859-1*)     |
+-------------------------+-----------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+

For SPECTRUM and IMAGES the actual sequence object used depends on the context
where the tango data is used, and the availability of :py:mod:`numpy`.

1. for properties the sequence is always a :py:class:`list`. Example::

    >>> import tango
    >>> db = tango.Database()
    >>> s = db.get_property(["TangoSynchrotrons"])
    >>> print type(s)
    <type 'list'>

2. for attribute/command values
    - :py:class:`numpy.ndarray` if PyTango was compiled with :py:mod:`numpy`
      support (default) and :py:mod:`numpy` is installed.
    - :py:class:`list` otherwise


.. _pytango-pipe-data-types:

Pipe data types
---------------

Pipes require different data types. You can think of them as a structured type.

A pipe transports data which is called a *blob*. A *blob* consists of name and
a list of fields. Each field is called *data element*. Each *data element*
consists of a name and a value. *Data element* names must be unique in the same
blob.

The value can be of any of the SCALAR or SPECTRUM tango data types (except
DevEnum).

Additionally, the value can be a *blob* itself.

In PyTango, a *blob* is represented by a sequence of two elements:

* blob name (str)
* data is either:

  * sequence (:py:class:`list`, :py:class:`tuple`, or other) of data elements
    where each element is a :py:class:`dict` with the following keys:

    * *name* (mandatory): (str) data element name
    * *value* (mandatory): data (compatible with any of the SCALAR or SPECTRUM
      data types except DevEnum). If value is to be a sub-*blob* then it
      should be sequence of [*blob name*, sequence of data elements]
      (see above)
    * *dtype* (optional, mandatory if a DevEncoded is required):
      see :ref:`Data type equivalence <pytango-hlapi-datatypes>`. If dtype
      key is not given, PyTango will try to find the proper tango type by
      inspecting the value.

  * a :py:class:`dict` where key is the data element name and value is the data
    element value (compact version)

When using the compact dictionary version note that the order of the data elements
is lost. If the order is important for you, consider using
:py:class:`collections.OrderedDict` instead (if you have python >=2.7. If not you can
use ``ordereddict`` backport module available on pypi).
Also, in compact mode it is not possible to enforce a specific type. As a
consequence, DevEncoded is not supported in compact mode.

The description sounds more complicated that it actually is. Here are some practical
examples of what you can return in a server as a read request from a pipe::

    import numpy as np

    # plain (one level) blob showing different tango data types
    # (explicity and implicit):

    PIPE0 = \
    ('BlobCase0',
     ({'name': 'DE1', 'value': 123,},                                # converts to DevLong64
      {'name': 'DE2', 'value': np.int32(456),},                      # converts to DevLong
      {'name': 'DE3', 'value': 789, 'dtype': 'int32'},               # converts to DevLong
      {'name': 'DE4', 'value': np.uint32(123)},                      # converts to DevULong
      {'name': 'DE5', 'value': range(5), 'dtype': ('uint16',)},      # converts to DevVarUShortArray
      {'name': 'DE6', 'value': [1.11, 2.22], 'dtype': ('float64',)}, # converts to DevVarDoubleArray
      {'name': 'DE7', 'value': numpy.zeros((100,))},                 # converts to DevVarDoubleArray
      {'name': 'DE8', 'value': True},                                # converts to DevBoolean
     )
    )


    # similar as above but in compact version (implicit data type conversion):

    PIPE1 = \
    ('BlobCase1', dict(DE1=123, DE2=np.int32(456), DE3=np.int32(789),
                       DE4=np.uint32(123), DE5=np.arange(5, dtype='uint16'),
		       DE6=[1.11, 2.22], DE7=numpy.zeros((100,)),
		       DE8=True)
    )

    # similar as above but order matters so we use an ordered dict:

    import collections

    data = collections.OrderedDict()
    data['DE1'] = 123
    data['DE2'] = np.int32(456)
    data['DE3'] = np.int32(789)
    data['DE4'] = np.uint32(123)
    data['DE5'] = np.arange(5, dtype='uint16')
    data['DE6'] = [1.11, 2.22]
    data['DE7'] = numpy.zeros((100,))
    data['DE8'] = True

    PIPE2 = 'BlobCase2', data

    # another plain blob showing string, string array and encoded data types:

    PIPE3 = \
    ('BlobCase3',
     ({'name': 'stringDE',  'value': 'Hello'},
      {'name': 'VectorStringDE', 'value': ('bonjour', 'le', 'monde')},
      {'name': 'DevEncodedDE', 'value': ('json', '"isn\'t it?"'), 'dtype': 'bytes'},
     )
    )

    # blob with sub-blob which in turn has a sub-blob

    PIPE4 = \
    ('BlobCase4',
     ({'name': '1DE', 'value': ('Inner', ({'name': '1_1DE', 'value': 'Grenoble'},
                                          {'name': '1_2DE', 'value': ('InnerInner',
                                                                      ({'name': '1_1_1DE', 'value': np.int32(111)},
                                                                       {'name': '1_1_2DE', 'value': [3.33]}))
                                         })
      )},
      {'name': '2DE', 'value': (3,4,5,6), 'dtype': ('int32',) },
     )
    )
