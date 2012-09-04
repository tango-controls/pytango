
.. currentmodule:: PyTango

.. highlight:: python
   :linenothreshold: 3
   
.. _server:
   
The Tango Device Server Python API
==================================

This chapter does not explain what a Tango device or a device server is.
This is explained in details in "The Tango control system manual" available at
http://www.tango-controls.org/TangoKernel.
The device server described in the following example is a Tango device server
with one Tango class called *PyDsExp*. This class has two commands called
*IOLong* and *IOStringArray* and two attributes called *Long_attr* and
*Short_attr_rw*.

Importing python modules
------------------------

To write a Python script which is a Tango device server, you need to import 
two modules which are:

1. The :mod:`PyTango` module which is the Python to C++ interface
2. The Python classical :mod:`sys` module

This could be done with code like (supposing the PYTHONPATH environment variable
is correctly set)::

    import PyTango
    import sys

The main part of a Python device server
---------------------------------------

The rule of this part of a Tango device server is to:

    - Create the :class:`Util` object passing it the Python interpreter command
      line arguments
    - Add to this object the list of Tango class(es) which have to be hosted by
      this interpreter
    - Initialize the device server
    - Run the device server loop

The following is a typical code for this main function::

    if __name__ == '__main__':
        util = PyTango.Util(sys.argv)
        util.add_class(PyDsExpClass, PyDsExp)
        
        U = PyTango.Util.instance()
        U.server_init()
        U.server_run()

**Line 2**
    Create the Util object passing it the interpreter command line arguments
**Line 3**
    Add the Tango class *PyDsExp* to the device server. The :meth:`Util.add_class`
    method of the Util class has two arguments which are the Tango class 
    PyDsExpClass instance and the Tango PyDsExp instance.
    This :meth:`Util.add_class` method is only available since version 
    7.1.2. If you are using an older version please use 
    :meth:`Util.add_TgClass` instead.
**Line 7**
    Initialize the Tango device server
**Line 8**
    Run the device server loop

The PyDsExpClass class in Python
--------------------------------

The rule of this class is to :

    - Host and manage data you have only once for the Tango class whatever
      devices of this class will be created
    - Define Tango class command(s)
    - Define Tango class attribute(s)

In our example, the code of this Python class looks like::

    class PyDsExpClass(PyTango.DeviceClass):

        cmd_list = { 'IOLong' : [ [ PyTango.ArgType.DevLong, "Number" ],
                                  [ PyTango.ArgType.DevLong, "Number * 2" ] ],
                     'IOStringArray' : [ [ PyTango.ArgType.DevVarStringArray, "Array of string" ],
                                         [ PyTango.ArgType.DevVarStringArray, "This reversed array"] ],
        }

        attr_list = { 'Long_attr' : [ [ PyTango.ArgType.DevLong ,
                                        PyTango.AttrDataFormat.SCALAR ,
                                        PyTango.AttrWriteType.READ],
                                      { 'min alarm' : 1000, 'max alarm' : 1500 } ],

                     'Short_attr_rw' : [ [ PyTango.ArgType.DevShort,
                                           PyTango.AttrDataFormat.SCALAR,
                                           PyTango.AttrWriteType.READ_WRITE ] ]
        }

        

**Line 1** 
    The PyDsExpClass class has to inherit from the :class:`DeviceClass` class
    
**Line 3 to 7**
    Definition of the cmd_list :class:`dict` defining commands. The *IOLong* command 
    is defined at lines 3 and 4. The *IOStringArray* command is defined in 
    lines 5 and 6
**Line 9 to 17**
    Definition of the attr_list :class:`dict` defining attributes. The *Long_attr*
    attribute is defined at lines 9 to 12 and the *Short_attr_rw* attribute is 
    defined at lines 14 to 16
    
If you have something specific to do in the class constructor like
initializing some specific data member, you will have to code a class 
constructor. An example of such a contructor is ::

    def __init__(self, name):
        PyTango.DeviceClass.__init__(self, name)
        self.set_type("TestDevice")

The device type is set at line 3.

Defining commands
-----------------

As shown in the previous example, commands have to be defined in a :class:`dict`
called *cmd_list* as a data member of the xxxClass class of the Tango class.
This :class:`dict` has one element per command. The element key is the command
name. The element value is a python list which defines the command. The generic
form of a command definition is:
    
    ``'cmd_name' : [ [in_type, <"In desc">], [out_type, <"Out desc">], <{opt parameters}>]``

The first element of the value list is itself a list with the command input
data type (one of the :class:`PyTango.ArgType` pseudo enumeration value) and
optionally a string describing this input argument. The second element of the
value list is also a list with the command output data type (one of the
:class:`PyTango.ArgType` pseudo enumeration value) and optionaly a string
describing it. These two elements are mandatory. The third list element is
optional and allows additional command definition. The authorized element for
this :class:`dict` are summarized in the following array:

    +-------------------+----------------------+------------------------------------------+
    |      key          |        Value         |             Definition                   |
    +===================+======================+==========================================+
    | "display level"   | DispLevel enum value |       The command display level          |
    +-------------------+----------------------+------------------------------------------+
    | "polling period"  | Any number           |     The command polling period (mS)      |
    +-------------------+----------------------+------------------------------------------+
    | "default command" | True or False        | To define that it is the default command |
    +-------------------+----------------------+------------------------------------------+

Defining attributes
-------------------

As shown in the previous example, attributes have to be defined in a :class:`dict`
called **attr_list** as a data
member of the xxxClass class of the Tango class. This :class:`dict` has one element
per attribute. The element key is the attribute name. The element value is a
python :class:`list` which defines the attribute. The generic form of an 
attribute definition is:

    ``'attr_name' : [ [mandatory parameters], <{opt parameters}>]``

For any kind of attributes, the mandatory parameters are:

    ``[attr data type, attr data format, attr data R/W type]``
    
The attribute data type is one of the possible value for attributes of the
:class:`PyTango.ArgType` pseudo enunmeration. The attribute data format is one
of the possible value of the :class:`PyTango.AttrDataFormat` pseudo enumeration
and the attribute R/W type is one of the possible value of the
:class:`PyTango.AttrWriteType` pseudo enumeration. For spectrum attribute,
you have to add the maximum X size (a number). For image attribute, you have
to add the maximun X and Y dimension (two numbers). The authorized elements for
the :class:`dict` defining optional parameters are summarized in the following
array:

    +-------------------+-----------------------------------+------------------------------------------+
    |       key         |              value                |            definition                    |
    +===================+===================================+==========================================+
    | "display level"   | PyTango.DispLevel enum value      |   The attribute display level            |
    +-------------------+-----------------------------------+------------------------------------------+
    |"polling period"   |          Any number               | The attribute polling period (mS)        |
    +-------------------+-----------------------------------+------------------------------------------+
    |  "memorized"      | True or True_without_hard_applied | Define if and how the att. is memorized  |
    +-------------------+-----------------------------------+------------------------------------------+
    |     "label"       |            A string               |       The attribute label                |
    +-------------------+-----------------------------------+------------------------------------------+
    |  "description"    |            A string               |   The attribute description              |
    +-------------------+-----------------------------------+------------------------------------------+
    |     "unit"        |            A string               |       The attribute unit                 |
    +-------------------+-----------------------------------+------------------------------------------+
    |"standard unit"    |           A number                |  The attribute standard unit             |
    +-------------------+-----------------------------------+------------------------------------------+
    | "display unit"    |            A string               |   The attribute display unit             |
    +-------------------+-----------------------------------+------------------------------------------+
    |    "format"       |            A string               | The attribute display format             |
    +-------------------+-----------------------------------+------------------------------------------+
    |  "max value"      |          A number                 |   The attribute max value                |
    +-------------------+-----------------------------------+------------------------------------------+
    |   "min value"     |           A number                |    The attribute min value               |
    +-------------------+-----------------------------------+------------------------------------------+
    |  "max alarm"      |           A number                |    The attribute max alarm               |
    +-------------------+-----------------------------------+------------------------------------------+
    |  "min alarm"      |           A number                |    The attribute min alarm               |
    +-------------------+-----------------------------------+------------------------------------------+
    | "min warning"     |           A number                |  The attribute min warning               |
    +-------------------+-----------------------------------+------------------------------------------+
    |"max warning"      |           A number                |  The attribute max warning               |
    +-------------------+-----------------------------------+------------------------------------------+
    |  "delta time"     |           A number                | The attribute RDS alarm delta time       |
    +-------------------+-----------------------------------+------------------------------------------+
    |   "delta val"     |           A number                | The attribute RDS alarm delta val        |
    +-------------------+-----------------------------------+------------------------------------------+

The PyDsExp class in Python
---------------------------

The rule of this class is to implement methods executed by commands and attributes.
In our example, the code of this class looks like::

    class PyDsExp(PyTango.Device_4Impl):

        def __init__(self,cl,name):
            PyTango.Device_4Impl.__init__(self, cl, name)
            self.info_stream('In PyDsExp.__init__')
            PyDsExp.init_device(self)

        def init_device(self):
            self.info_stream('In Python init_device method')
            self.set_state(PyTango.DevState.ON)
            self.attr_short_rw = 66
            self.attr_long = 1246

        #------------------------------------------------------------------

        def delete_device(self):
            self.info_stream('PyDsExp.delete_device')

        #------------------------------------------------------------------
        # COMMANDS
        #------------------------------------------------------------------

        def is_IOLong_allowed(self):
            return self.get_state() == PyTango.DevState.ON

        def IOLong(self, in_data):
            self.info_stream('IOLong', in_data)
            in_data = in_data * 2
            self.info_stream('IOLong returns', in_data)
            return in_data

        #------------------------------------------------------------------

        def is_IOStringArray_allowed(self):
            return self.get_state() == PyTango.DevState.ON

        def IOStringArray(self, in_data):
            l = range(len(in_data)-1, -1, -1)
            out_index=0
            out_data=[]
            for i in l:
                self.info_stream('IOStringArray <-', in_data[out_index])
                out_data.append(in_data[i])
                self.info_stream('IOStringArray ->',out_data[out_index])
                out_index += 1
            self.y = out_data
            return out_data

        #------------------------------------------------------------------
        # ATTRIBUTES
        #------------------------------------------------------------------

        def read_attr_hardware(self, data):
            self.info_stream('In read_attr_hardware')

        def read_Long_attr(self, the_att):
            self.info_stream("read_Long_attr")

            the_att.set_value(self.attr_long)

        def is_Long_attr_allowed(self, req_type):
            return self.get_state() in (PyTango.DevState.ON,)

        def read_Short_attr_rw(self, the_att):
            self.info_stream("read_Short_attr_rw")

            the_att.set_value(self.attr_short_rw)

        def write_Short_attr_rw(self, the_att):
            self.info_stream("write_Short_attr_rw")

            self.attr_short_rw = the_att.get_write_value()

        def is_Short_attr_rw_allowed(self, req_type):
            return self.get_state() in (PyTango.DevState.ON,)

**Line 1**
    The PyDsExp class has to inherit from the PyTango.Device_4Impl
**Line 3 to 6**
    PyDsExp class constructor. Note that at line 6, it calls the *init_device()*
    method
**Line 8 to 12**
    The init_device() method. It sets the device state (line 9) and initialises
    some data members
**Line 16 to 17**
    The delete_device() method. This method is not mandatory. You define it
    only if you have to do something specific before the device is destroyed
**Line 23 to 30**
    The two methods for the *IOLong* command. The first method is called
    *is_IOLong_allowed()* and it is the command is_allowed method (line 23 to 24).
    The second method has the same name than the command name. It is the method
    which executes the command. The command input data type is a Tango long
    and therefore, this method receives a python integer.
**Line 34 to 47**
    The two methods for the *IOStringArray* command. The first method is its
    is_allowed method (Line 34 to 35). The second one is the command
    execution method (Line 37 to 47). The command input data type is a string
    array. Therefore, the method receives the array in a python list of python
    strings.
**Line 53 to 54**
    The *read_attr_hardware()* method. Its argument is a Python sequence of
    Python integer.
**Line 56 to 59**
    The method executed when the *Long_attr* attribute is read. Note that before
    PyTango 7 it sets the attribute value with the PyTango.set_attribute_value
    function. Now the same can be done using the set_value of the attribute
    object
**Line 61 to 62**
    The is_allowed method for the *Long_attr* attribute. This is an optional
    method that is called when the attribute is read or written. Not defining it
    has the same effect as always returning True. The parameter req_type is of
    type :class:`AttReqtype` which tells if the method is called due to a read
    or write request. Since this is a read-only attribute, the method will only
    be called for read requests, obviously.
**Line 64 to 67**
    The method executed when the *Short_attr_rw* attribute is read.
**Line 69 to 72**
    The method executed when the Short_attr_rw attribute is written.
    Note that before PyTango 7 it gets the attribute value with a call to the
    Attribute method *get_write_value* with a list as argument. Now the write
    value can be obtained as the return value of the *get_write_value* call. And
    in case it is a scalar there is no more the need to extract it from the list.
**Line 74 to 75**
    The is_allowed method for the *Short_attr_rw* attribute. This is an optional
    method that is called when the attribute is read or written. Not defining it
    has the same effect as always returning True. The parameter req_type is of
    type :class:`AttReqtype` which tells if the method is called due to a read
    or write request.

General methods
###############

The following array summarizes how the general methods we have in a Tango 
device server are implemented in Python.

+----------------------+-------------------------+-------------+-----------+
|         Name         | Input par (with "self") |return value | mandatory |
+======================+=========================+=============+===========+
|      init_device     |        None             |   None      |  Yes      |
+----------------------+-------------------------+-------------+-----------+
|     delete_device    |        None             |   None      |  No       |
+----------------------+-------------------------+-------------+-----------+
| always_executed_hook |        None             |   None      |  No       |
+----------------------+-------------------------+-------------+-----------+
|    signal_handler    |   :py:obj:`int`         |   None      |  No       |
+----------------------+-------------------------+-------------+-----------+
| read_attr_hardware   | sequence<:py:obj:`int`> |   None      |  No       |
+----------------------+-------------------------+-------------+-----------+

Implementing a command
######################

Commands are defined as described above. Nevertheless, some methods implementing 
them have to be written. These methods names are fixed and depend on command 
name. They have to be called:

    - ``is_<Cmd_name>_allowed(self)``
    - ``<Cmd_name>(self, arg)``

For instance, with a command called *MyCmd*, its is_allowed method has to be 
called `is_MyCmd_allowed` and its execution method has to be called simply *MyCmd*.
The following array gives some more info on these methods.

+-----------------------+-------------------------+--------------------+-----------+
|        Name           | Input par (with "self") | return value       | mandatory |
+=======================+=========================+====================+===========+
| is_<Cmd_name>_allowed |        None             | Python boolean     |  No       |
+-----------------------+-------------------------+--------------------+-----------+
|      Cmd_name         | Depends on cmd type     |Depends on cmd type |  Yes      |
+-----------------------+-------------------------+--------------------+-----------+

Tango has more data types than Python which is more dynamic. The input and
output values of the commands are translated according to the array below.
Note that if PyTango is compiled with :py:mod:`numpy` support the numpy type
will be the used for the input arguments. Also, it is recomended to use numpy
arrays of the appropiate type for output arguments as well, as it is much more
efficient.

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
|                         | ========= ============================================================    | ========= =============================================================== |
|                         | ExtractAs                        Data Type                                | ExtractAs                        Data Type                                |
|                         | ========= ============================================================    | ========= =============================================================== |
|    DEVVAR_CHARARRAY     | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint8`)        | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint8`)        |
|                         | Bytes     :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)       | Bytes     :py:obj:`bytes`                                                 |
|                         | ByteArray :py:obj:`bytearray`                                             | ByteArray :py:obj:`bytearray`                                             |
|                         | String    :py:obj:`str`                                                   | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         | List      :py:class:`list` <:py:obj:`int`>                                | List      :py:class:`list` <:py:obj:`int`>                                |
|                         | Tuple     :py:class:`tuple` <:py:obj:`int`>                               | Tuple     :py:class:`tuple` <:py:obj:`int`>                               |
|                         | ========= ============================================================    | ========= =============================================================== |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | ========= =============================================================== | ========= =============================================================== |
|                         | ExtractAs                        Data Type                                | ExtractAs                        Data Type                                |
|                         | ========= =============================================================== | ========= =============================================================== |
|    DEVVAR_SHORTARRAY    | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint16`)       | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint16`)       |
|                         | Bytes     :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)       | Bytes     :py:obj:`bytes`                                                 |
|                         | ByteArray :py:obj:`bytearray`                                             | ByteArray :py:obj:`bytearray`                                             |
|                         | String    :py:obj:`str`                                                   | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         | List      :py:class:`list` <:py:obj:`int`>                                | List      :py:class:`list` <:py:obj:`int`>                                |
|                         | Tuple     :py:class:`tuple` <:py:obj:`int`>                               | Tuple     :py:class:`tuple` <:py:obj:`int`>                               |
|                         | ========= =============================================================== | ========= =============================================================== |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | ========= =============================================================== | ========= =============================================================== |
|                         | ExtractAs                        Data Type                                | ExtractAs                        Data Type                                |
|                         | ========= =============================================================== | ========= =============================================================== |
|    DEVVAR_LONGARRAY     | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint32`)       | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint32`)       |
|                         | Bytes     :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)       | Bytes     :py:obj:`bytes`                                                 |
|                         | ByteArray :py:obj:`bytearray`                                             | ByteArray :py:obj:`bytearray`                                             |
|                         | String    :py:obj:`str`                                                   | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         | List      :py:class:`list` <:py:obj:`int`>                                | List      :py:class:`list` <:py:obj:`int`>                                |
|                         | Tuple     :py:class:`tuple` <:py:obj:`int`>                               | Tuple     :py:class:`tuple` <:py:obj:`int`>                               |
|                         | ========= =============================================================== | ========= =============================================================== |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | ========= =============================================================== | ========= =============================================================== |
|                         | ExtractAs                        Data Type                                | ExtractAs                        Data Type                                |
|                         | ========= =============================================================== | ========= =============================================================== |
|    DEVVAR_LONG64ARRAY   | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint64`)       | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint64`)       |
|                         | Bytes     :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)       | Bytes     :py:obj:`bytes`                                                 |
|                         | ByteArray :py:obj:`bytearray`                                             | ByteArray :py:obj:`bytearray`                                             |
|                         | String    :py:obj:`str`                                                   | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         | List      :py:class:`list` <int (64 bits) / long (32 bits)>               | List      :py:class:`list` <:py:obj:`int`>                                |
|                         | Tuple     :py:class:`tuple` <int (64 bits) / long (32 bits)>              | Tuple     :py:class:`tuple` <:py:obj:`int`>                               |
|                         | ========= =============================================================== | ========= =============================================================== |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | ========= =============================================================== | ========= =============================================================== |
|                         | ExtractAs                        Data Type                                | ExtractAs                        Data Type                                |
|                         | ========= =============================================================== | ========= =============================================================== |
|    DEVVAR_FLOATARRAY    | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float32`)      | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float32`)      |
|                         | Bytes     :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)       | Bytes     :py:obj:`bytes`                                                 |
|                         | ByteArray :py:obj:`bytearray`                                             | ByteArray :py:obj:`bytearray`                                             |
|                         | String    :py:obj:`str`                                                   | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         | List      :py:class:`list` <:py:obj:`int`>                                | List      :py:class:`list` <:py:obj:`int`>                                |
|                         | Tuple     :py:class:`tuple` <:py:obj:`int`>                               | Tuple     :py:class:`tuple` <:py:obj:`int`>                               |
|                         | ========= =============================================================== | ========= =============================================================== |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | ========= =============================================================== | ========= =============================================================== |
|                         | ExtractAs                        Data Type                                | ExtractAs                        Data Type                                |
|                         | ========= =============================================================== | ========= =============================================================== |
|    DEVVAR_DOUBLEARRAY   | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float64`)      | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float64`)      |
|                         | Bytes     :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)       | Bytes     :py:obj:`bytes`                                                 |
|                         | ByteArray :py:obj:`bytearray`                                             | ByteArray :py:obj:`bytearray`                                             |
|                         | String    :py:obj:`str`                                                   | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         | List      :py:class:`list` <:py:obj:`int`>                                | List      :py:class:`list` <:py:obj:`int`>                                |
|                         | Tuple     :py:class:`tuple` <:py:obj:`int`>                               | Tuple     :py:class:`tuple` <:py:obj:`int`>                               |
|                         | ========= =============================================================== | ========= =============================================================== |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | ========= =============================================================== | ========= =============================================================== |
|                         | ExtractAs                        Data Type                                | ExtractAs                        Data Type                                |
|                         | ========= =============================================================== | ========= =============================================================== |
|    DEVVAR_USHORTARRAY   | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint16`)       | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint16`)       |
|                         | Bytes     :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)       | Bytes     :py:obj:`bytes`                                                 |
|                         | ByteArray :py:obj:`bytearray`                                             | ByteArray :py:obj:`bytearray`                                             |
|                         | String    :py:obj:`str`                                                   | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         | List      :py:class:`list` <:py:obj:`int`>                                | List      :py:class:`list` <:py:obj:`int`>                                |
|                         | Tuple     :py:class:`tuple` <:py:obj:`int`>                               | Tuple     :py:class:`tuple` <:py:obj:`int`>                               |
|                         | ========= =============================================================== | ========= =============================================================== |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | ========= =============================================================== | ========= =============================================================== |
|                         | ExtractAs                        Data Type                                | ExtractAs                        Data Type                                |
|                         | ========= =============================================================== | ========= =============================================================== |
|    DEVVAR_ULONGARRAY    | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint32`)       | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint32`)       |
|                         | Bytes     :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)       | Bytes     :py:obj:`bytes`                                                 |
|                         | ByteArray :py:obj:`bytearray`                                             | ByteArray :py:obj:`bytearray`                                             |
|                         | String    :py:obj:`str`                                                   | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         | List      :py:class:`list` <:py:obj:`int`>                                | List      :py:class:`list` <:py:obj:`int`>                                |
|                         | Tuple     :py:class:`tuple` <:py:obj:`int`>                               | Tuple     :py:class:`tuple` <:py:obj:`int`>                               |
|                         | ========= =============================================================== | ========= =============================================================== |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | ========= =============================================================== | ========= =============================================================== |
|                         | ExtractAs                        Data Type                                | ExtractAs                        Data Type                                |
|                         | ========= =============================================================== | ========= =============================================================== |
|    DEVVAR_ULONG64ARRAY  | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint64`)       | [Numpy]   :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint64`)       |
|                         | Bytes     :py:obj:`bytes` (which is in fact equal to :py:obj:`str`)       | Bytes     :py:obj:`bytes`                                                 |
|                         | ByteArray :py:obj:`bytearray`                                             | ByteArray :py:obj:`bytearray`                                             |
|                         | String    :py:obj:`str`                                                   | String    :py:obj:`str` (decoded with default python encoding *utf-8*!!!) |
|                         | List      :py:class:`list` <int (64 bits) / long (32 bits)>               | List      :py:class:`list` <:py:obj:`int`>                                |
|                         | Tuple     :py:class:`tuple` <int (64 bits) / long (32 bits)>              | Tuple     :py:class:`tuple` <:py:obj:`int`>                               |
|                         | ========= =============================================================== | ========= =============================================================== |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|   DEVVAR_STRINGARRAY    | sequence<:py:obj:`str`>                                                   | sequence<:py:obj:`str`> (decoded with *latin-1*, aka *ISO-8859-1*)        |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | sequence of two elements:                                                 | sequence of two elements:                                                 |
|  DEV_LONGSTRINGARRAY    |                                                                           |                                                                           |
|                         | 0. :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.int32`) or            | 0. :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.int32`) or            |
|                         |    sequence<:py:obj:`int`>                                                |    sequence<:py:obj:`int`>                                                |
|                         | 1. sequence<:py:obj:`str`>                                                | 1.  sequence<:py:obj:`str`> (decoded with *latin-1*, aka *ISO-8859-1*)    |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+
|                         | sequence of two elements:                                                 | sequence of two elements:                                                 |
|  DEV_DOUBLESTRINGARRAY  |                                                                           |                                                                           |
|                         | 0. :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float64`) or          | 0. :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float64`) or          |
|                         |    sequence<:py:obj:`int`>                                                |    sequence<:py:obj:`int`>                                                |
|                         | 1. sequence<:py:obj:`str`>                                                | 1. sequence<:py:obj:`str`> (decoded with *latin-1*, aka *ISO-8859-1*)     |
+-------------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------+

The following code is an example of how you write code executed when a client
calls a command named IOLong::

    def is_IOLong_allowed(self):
        self.debug_stream("in is_IOLong_allowed")
        return self.get_state() == PyTango.DevState.ON

    def IOLong(self, in_data):
        self.info_stream('IOLong', in_data)
        in_data = in_data * 2
        self.info_stream('IOLong returns', in_data)
        return in_data

**Line 1-3**
    the is_IOLong_allowed method determines in which conditions the command
    'IOLong' can be executed. In this case, the command can only be executed if
    the device is in 'ON' state.
**Line 6**
    write a log message to the tango INFO stream (click :ref:`here <logging>` for
    more information about PyTango log system).
**Line 7**
    does something with the input parameter
**Line 8**
    write another log message to the tango INFO stream (click :ref:`here <logging>` for
    more information about PyTango log system).
**Line 9**
    return the output of executing the tango command
    
Implementing an attribute
#########################

Attributes are defined as described in chapter 5.3.2. Nevertheless, some methods
implementing them have to be written. These methods names are fixed and depend 
on attribute name. They have to be called:

    - ``is_<Attr_name>_allowed(self, req_type)``
    - ``read_<Attr_name>(self, attr)``
    - ``write_<Attr_name>(self, attr)``

For instance, with an attribute called *MyAttr*, its is_allowed method has to be 
called *is_MyAttr_allowed*, its read method has to be called *read_MyAttr* and 
its write method has to be called *write_MyAttr*.
The *attr* parameter is an instance of :class:`Attr`.
Unlike the commands, the is_allowed method for attributes receives a parameter
of type :class:`AttReqtype`.
The following table gives some more info on these methods:

.. html:
    +-------------+-------------+--------------------------------------------------------------+
    | data format | data type   |  python type                                                 |
    +=============+=============+==============================================================+
    | SCALAR      | DEV_BOOLEAN | :py:obj:`bool`                                               |
    |-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_UCHAR   | :py:obj:`int`                                                |
    +-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_SHORT   | :py:obj:`int`                                                |
    +-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_USHORT  | :py:obj:`int`                                                |
    +-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_LONG    | :py:obj:`int`                                                |
    +-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_ULONG   | :py:obj:`int`                                                |
    +-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_LONG64  | :py:obj:`int`/ :py:obj:`long`                                |
    +-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_ULONG64 | :py:obj:`int`/ :py:obj:`long`                                |
    +-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_FLOAT   | :py:obj:`float`                                              |
    +-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_DOUBLE  | :py:obj:`float`                                              |
    +-------------+-------------+--------------------------------------------------------------+
    | SCALAR      | DEV_STRING  | :py:obj:`str`                                                |
    +-------------+-------------+--------------------------------------------------------------+
    | SPECTRUM    | DEV_BOOLEAN | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.bool`)      |
    | or IMAGE    |             | or sequence<:py:obj:`bool`>                                  |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_UCHAR   | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint8`)     |
    |             |             | or sequence<:py:obj:`int`>                                   |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_SHORT   | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.int16`)     |
    |             |             | or sequence<:py:obj:`int`>                                   |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_USHORT  | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint16`)    |
    |             |             | or sequence<:py:obj:`int`>                                   |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_LONG    | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.int32`)     |
    |             |             | or sequence<:py:obj:`int`>                                   |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_ULONG   | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint32`)    |
    |             |             | or sequence<:py:obj:`int`>                                   |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_LONG64  | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.int64`)     |
    |             |             | or sequence<:py:obj:`int`>                                   |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_ULONG64 | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.uint64`)    |
    |             |             | or sequence<:py:obj:`int`>                                   |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_FLOAT   | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float32`)   |
    |             |             | or sequence<:py:obj:`float`>                                 |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_DOUBLE  | :py:class:`numpy.ndarray` (dtype= :py:obj:`numpy.float64`)   |
    |             |             | or sequence<:py:obj:`float`>                                 |
    +-------------+-------------+--------------------------------------------------------------+
    |             | DEV_STRING  | sequence<:py:obj:`str`>                                      |
    +-------------+-------------+--------------------------------------------------------------+

For SPECTRUM and IMAGES the actual sequence object used depends on the context 
where the tango data is used, and the availability of :py:mod:`numpy`.

1. for properties the sequence is always a :py:class:`list`. Example:
    
    >>> import PyTango
    >>> db = PyTango.Database()
    >>> s = db.get_property(["TangoSynchrotrons"])
    >>> print type(s)
    <type 'list'>

2. for attribute/command values
    - :py:class:`numpy.ndarray` if PyTango was compiled with :py:mod:`numpy`
      support (default) and :py:mod:`numpy` is installed.
    - :py:class:`list` otherwise
    
The following code is an example of how you write code executed when a client
read an attribute which is called *Long_attr*::
    
    def read_Long_attr(self, the_att):
        self.info_stream("read attribute name Long_attr")
        the_att.set_value(self.attr_long)

**Line 1**
    Method declaration with "the_att" being an instance of the Attribute
    class representing the Long_attr attribute
**Line 2**
    write a log message to the tango INFO stream (click :ref:`here <logging>`
    for more information about PyTango log system).
**Line 3**
    Set the attribute value using the method set_value() with the attribute 
    value as parameter.
    
The following code is an example of how you write code executed when a client 
write the Short_attr_rw attribute::

    def write_Short_attr_rw(self,the_att):
        self.info_stream("In write_Short_attr_rw for attribute ",the_att.get_name())
        self.attr_short_rw = the_att.get_write_value(data)

**Line 1**
       Method declaration with "the_att" being an instance of the Attribute 
       class representing the Short_attr_rw attribute
**Line 2**
    write a log message to the tango INFO stream (click :ref:`here <logging>` for
    more information about PyTango log system).
**Line 3**
    Get the value sent by the client using the method get_write_value() and
    store the value written in the device object. Our attribute is a scalar 
    short attribute so the return value is an int

.. _logging:

Logging
#######

This chapter instructs you on how to use the tango logging API (log4tango) to
create tango log messages on your device server.

The logging system explained here is the Tango Logging Service (TLS). For
detailed information on how this logging system works please check:

    * `3.5 The tango logging service <http://www.esrf.eu/computing/cs/tango/tango_doc/kernel_doc/ds_prog/node4.html#sec:The-Tango-Logging>`_
    * `9.3 The tango logging service <http://www.esrf.eu/computing/cs/tango/tango_doc/kernel_doc/ds_prog/node9.html#SECTION00930000000000000000>`_

The easiest way to start seeing log messages on your device server console is
by starting it with the verbose option. Example::

    python PyDsExp.py PyDs1 -v4

This activates the console tango logging target and filters messages with 
importance level DEBUG or more.
The links above provided detailed information on how to configure log levels 
and log targets. In this document we will focus on how to write log messages on
your device server.

Basic logging
~~~~~~~~~~~~~

The most basic way to write a log message on your device is to use the
:class:`PyTango.DeviceImpl` logging related methods:

    * :meth:`PyTango.DeviceImpl.debug_stream`
    * :meth:`PyTango.DeviceImpl.info_stream`
    * :meth:`PyTango.DeviceImpl.warn_stream`
    * :meth:`PyTango.DeviceImpl.error_stream`
    * :meth:`PyTango.DeviceImpl.fatal_stream`

Example::

    def read_Long_attr(self, the_att):
        self.info_stream("read attribute name Long_attr")
        the_att.set_value(self.attr_long)

This will print a message like::

    1282206864 [-1215867200] INFO test/pydsexp/1 read attribute name Long_attr

every time a client asks to read the 'Long_attr' attribute value.

Logging with print statement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*This feature is only possible since PyTango 7.1.3*

It is possible to use the print statement to log messages into the tango logging
system. This is achieved by using the python's print extend form sometimes
refered to as *print chevron*.

Same example as above, but now using *print chevron*::

    def read_Long_attr(self, the_att):
        print >>self.log_info, "read attribute name Long_attr"
        the_att.set_value(self.attr_long)

Or using the python 3k print function::

    def read_Long_attr(self, the_att):
        print("read attribute name Long_attr", file=self.log_info)
        the_att.set_value(self.attr_long)

Logging with decorators
~~~~~~~~~~~~~~~~~~~~~~~

*This feature is only possible since PyTango 7.1.3*

PyTango provides a set of decorators that place automatic log messages when
you enter and when you leave a python method. For example::

    @PyTango.DebugIt()
    def read_Long_attr(self, the_att):
        the_att.set_value(self.attr_long)

will generate a pair of log messages each time a client asks for the 'Long_attr'
value. Your output would look something like::

    1282208997 [-1215965504] DEBUG test/pydsexp/1 -> read_Long_attr()
    1282208997 [-1215965504] DEBUG test/pydsexp/1 <- read_Long_attr()

Decorators exist for all tango log levels:
    * :class:`PyTango.DebugIt`
    * :class:`PyTango.InfoIt`
    * :class:`PyTango.WarnIt`
    * :class:`PyTango.ErrorIt`
    * :class:`PyTango.FatalIt`

The decorators receive three optional arguments:
    * show_args - shows method arguments in log message (defaults to False)
    * show_kwargs shows keyword method arguments in log message (defaults to False)
    * show_ret - shows return value in log message (defaults to False)

Example::
    
    @PyTango.DebugIt(show_args=True, show_ret=True)
    def IOLong(self, in_data):
        return in_data * 2

will output something like::

    1282221947 [-1261438096] DEBUG test/pydsexp/1 -> IOLong(23)
    1282221947 [-1261438096] DEBUG test/pydsexp/1 46 <- IOLong()

Dynamic devices
###############

*This feature is only possible since PyTango 7.1.2*

Starting from PyTango 7.1.2 it is possible to create devices in a device server
"en caliente". This means that you can create a command in your "management device"
of a device server that creates devices of (possibly) several other tango classes.
There are two ways to create a new device which are described below.

Dynamic device from a known tango class name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you know the tango class name but you don't have access to the :class:`PyTango.DeviceClass`
(or you are too lazy to search how to get it ;-) the way to do it is call 
:meth:`PyTango.Util.create_device` / :meth:`PyTango.Util.delete_device`.
Here is an example of implementing a tango command on one of your devices that 
creates a device of some arbitrary class (the example assumes the tango commands
'CreateDevice' and 'DeleteDevice' receive a parameter of type DevVarStringArray
with two strings. No error processing was done on the code for simplicity sake)::

    class MyDevice(PyTango.Device_4Impl):
        ...
        
        def CreateDevice(self, pars):
            klass_name, dev_name = pars
            util = PyTango.Util.instance()
            util.create_device(klass_name, dev_name, alias=None, cb=None)
        
        def DeleteDevice(self, pars):
            klass_name, dev_name = pars
            util = PyTango.Util.instance()
            util.delete_device(klass_name, dev_name)

An optional callback can be registered that will be executed after the device is
registed in the tango database but before the actual device object is created and its
init_device method is called. You can, for example, initialize some device properties
here.

Dynamic device from a known tango class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have access to the :class:`PyTango.DeviceClass` object that
corresponds to the tango class of the device to be created you can call directly
the :meth:`PyTango.DeviceClass.create_device` / :meth:`PyTango.DeviceClass.delete_device`.
For example, if you wish to create a clone of your device, you can create a 
tango command called Clone::

    class MyDevice(PyTango.Device_4Impl):
        ...
        
        def fill_new_device_properties(self, dev_name):
            prop_names = db.get_device_property_list(self.get_name(), "*")
            prop_values = db.get_device_property(self.get_name(), prop_names.value_string)
            db.put_device_property(dev_name, prop_values)
            
            # do the same for attributes...
            ... 
        
        def Clone(self, dev_name):
            klass = self.get_device_class()
            klass.create_device(dev_name, alias=None, cb=self.fill_new_device_properties)
            
        def DeleteSibling(self, dev_name):
            klass = self.get_device_class()
            klass.delete_device(dev_name)
            
Note that the cb parameter is optional. In the example it is given for
demonstration purposes only.

Dynamic attributes
##################

It is also possible to create dynamic attributes within a Python device server. 
There are several ways to create dynamic attributes. One of the way, is to 
create all the devices within a loop, then to create the dynamic attributes and
finally to make all the devices available for the external world. In C++ device
server, this is typically done within the <Device>Class::device_factory() method.
In Python device server, this method is generic and the user does not have one.
Nevertheless, this generic device_factory method calls a method named dyn_attr()
allowing the user to create his dynamic attributes. It is simply necessary to
re-define this method within your <Device>Class and to create the dynamic 
attribute within this method:

    ``dyn_attr(self, dev_list)``
    
    where dev_list is a list containing all the devices created by the 
    generic device_factory() method.

There is another point to be noted regarding dynamic attribute within Python 
device server. The Tango Python device server core checks that for each 
attribute it exists methods named <attribute_name>_read and/or
<attribute_name>_write and/or is_<attribute_name>_allowed. Using dynamic
attribute, it is not possible to define these methods because attributes name
and number are known only at run-time.
To address this issue, the Device_3Impl::add_attribute() method has a diferent
signature for Python device server which is:

    ``add_attribute(self, attr, r_meth = None, w_meth = None, is_allo_meth = None)``
    
    attr is an instance of the Attr class, r_meth is the method which has to be 
    executed with the attribute is read, w_meth is the method to be executed 
    when the attribute is written and is_allo_meth is the method to be executed
    to implement the attribute state machine. The method passed here as argument
    as to be class method and not object method. Which argument you have to use 
    depends on the type of the attribute (A WRITE attribute does not need a 
    read method). Note, that depending on the number of argument you pass to this
    method, you may have to use Python keyword argument. The necessary methods 
    required by the Tango Python device server core will be created automatically
    as a forward to the methods given as arguments.

Mixing Tango classes (Python and C++) in a Python Tango device server
---------------------------------------------------------------------

Within the same python interpreter, it is possible to mix several Tango classes. 
Here is an example of the main function of a device server with two Tango classes
called IRMiror and PLC::

    import PyTango
    import sys

    if __name__ == '__main__':
        util = PyTango.Util(sys.argv)
        util.add_class(PLCClass, PLC, 'PLC')
        util.add_class(IRMirrorClass, IRMirror, 'IRMirror')
        
        U = PyTango.Util.instance()
        U.server_init()
        U.server_run()

:Line 6: The Tango class PLC is registered in the device server
:Line 7: The Tango class IRMirror is registered in the device server

It is also possible to add C++ Tango class in a Python device server as soon as:
    1. The Tango class is in a shared library
    2. It exist a C function to create the Tango class

For a Tango class called MyTgClass, the shared library has to be called 
MyTgClass.so and has to be in a directory listed in the LD_LIBRARY_PATH 
environment variable. The C function creating the Tango class has to be called 
_create_MyTgClass_class() and has to take one parameter of type "char \*" which 
is the Tango class name. Here is an example of the main function of the same 
device server than before but with one C++ Tango class called SerialLine::

    import PyTango
    import sys
    
    if __name__ == '__main__':
        py = PyTango.Util(sys.argv)
        util.add_class('SerialLine', 'SerialLine', language="c++")
        util.add_class(PLCClass, PLC, 'PLC')
        util.add_class(IRMirrorClass, IRMirror, 'IRMirror')
        
        U = PyTango.Util.instance()
        U.server_init()
        U.server_run()

:Line 6: The C++ class is registered in the device server
:Line 7 and 8: The two Python classes are registered in the device server

Server API
----------

.. toctree::
    :maxdepth: 2

    device
    device_class
    logging
    attribute
    util
