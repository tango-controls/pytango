.. currentmodule:: PyTango

FAQ
===

Answers to general Tango questions can be found at http://www.tango-controls.org/tutorials

Please also check http://www.tango-controls.org/howtos for a list of Tango howtos

Where are the usual bjam files?
-------------------------------

Starting from PyTango 7.0.0 the prefered way to build PyTango is using the standard
python distutils package. This means that:

- you do NOT have to install the additional bjam package
- you do NOT have to change 3 configuration files
- you do NOT need to have 2Gb of RAM to compile PyTango.

Please check the compilation chapter for details on how to build PyTango.

I got a libbost_python error when I try to import PyTango module
----------------------------------------------------------------

doing:
    >>> import PyTango
    ImportError: libboost_python-gcc43-mt-1_38.so.1.38.0: cannot open shared object file: No such file or directory

You must check that you have the correct boost python installed on your computer.
To see which boost python file PyTango needs type::

    $ ldd /usr/lib/python2.5/site-packages/PyTango/_PyTango.so
    linux-vdso.so.1 =>  (0x00007fff48bfe000)
    libtango.so.7 => /home/homer/local/lib/libtango.so.7 (0x00007f393fabb000)
    liblog4tango.so.4 => /home/homer/local/lib/liblog4tango.so.4 (0x00007f393f8a0000)
    **libboost_python-gcc43-mt-1_38.so.1.38.0 => not found**
    libpthread.so.0 => /lib/libpthread.so.0 (0x00007f393f65e000)
    librt.so.1 => /lib/librt.so.1 (0x00007f393f455000)
    libdl.so.2 => /lib/libdl.so.2 (0x00007f393f251000)
    libomniORB4.so.1 => /usr/local/lib/libomniORB4.so.1 (0x00007f393ee99000)
    libomniDynamic4.so.1 => /usr/local/lib/libomniDynamic4.so.1 (0x00007f393e997000)
    libomnithread.so.3 => /usr/local/lib/libomnithread.so.3 (0x00007f393e790000)
    libCOS4.so.1 => /usr/local/lib/libCOS4.so.1 (0x00007f393e359000)
    libgcc_s.so.1 => /lib/libgcc_s.so.1 (0x00007f393e140000)
    libc.so.6 => /lib/libc.so.6 (0x00007f393ddce000)
    libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f393dac1000)
    libm.so.6 => /lib/libm.so.6 (0x00007f393d83b000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f3940a4c000)


My python code uses PyTango 3.0.4 API. How do I change to 7.0.0 API?
--------------------------------------------------------------------

To ease migration effort, PyTango 7 provides an alternative module called
PyTango3.

Changing your python import from::

    import PyTango
    
to::

    import PyTango3 as PyTango
    
should allow you to execute your old PyTango code using the new PyTango 7 library.

Please note that you should as soon as possible migrate the code to Tango 7
since the PyTango team cannot assure the maintainability of the PyTango3 module.

Please find below a basic set of rules to migrate from PyTango 3.0.x to 7:

General rule of thumb for data types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first important thing to be aware of when migrating from PyTango <= 3.0.4 to
PyTango >= 7 is that the data type mapping from tango to python and vice versa is
not always the same. The following table summarizes the differences:

+-------------------------+-------------------------------------------+-------------------------------------------+
|   Tango data type       |              PyTango 7 type               | PyTango <= 3.0.4 type                     |
+=========================+===========================================+===========================================+
|          DEV_VOID       |                    No data                |                    No data                |
+-------------------------+-------------------------------------------+-------------------------------------------+
|       DEV_BOOLEAN       | bool                                      | bool                                      |
+-------------------------+-------------------------------------------+-------------------------------------------+
|         DEV_SHORT       | int                                       | int                                       |
+-------------------------+-------------------------------------------+-------------------------------------------+
|         DEV_LONG        | int                                       | int                                       |
+-------------------------+-------------------------------------------+-------------------------------------------+
|        DEV_LONG64       | long (on a 32 bits computer) or           | long (on a 32 bits computer) or           |
|                         | int (on a 64 bits computer)               | int (on a 64 bits computer)               |
+-------------------------+-------------------------------------------+-------------------------------------------+
|         DEV_FLOAT       | float                                     | float                                     |
+-------------------------+-------------------------------------------+-------------------------------------------+
|       DEV_DOUBLE        | float                                     | float                                     |
+-------------------------+-------------------------------------------+-------------------------------------------+
|        DEV_USHORT       | int                                       | int                                       |
+-------------------------+-------------------------------------------+-------------------------------------------+
|        DEV_ULONG        | int                                       | int                                       |
+-------------------------+-------------------------------------------+-------------------------------------------+
|        DEV_ULONG64      | long (on a 32 bits computer) or           | long (on a 32 bits computer) or           |
|                         | int (on a 64 bits computer)               | int (on a 64 bits computer)               |
+-------------------------+-------------------------------------------+-------------------------------------------+
|        DEV_STRING       | str                                       | str                                       |
+-------------------------+-------------------------------------------+-------------------------------------------+
|    DEVVAR_CHARARRAY     | sequence<int>                             | list<int>                                 |
+-------------------------+-------------------------------------------+-------------------------------------------+
|    DEVVAR_SHORTARRAY    | sequence<int>                             | list<int>                                 |
+-------------------------+-------------------------------------------+-------------------------------------------+
|    DEVVAR_LONGARRAY     | sequence<int>                             | list<int>                                 |
+-------------------------+-------------------------------------------+-------------------------------------------+
|   DEVVAR_LONG64ARRAY    | sequence<long> (on a 32 bits computer) or | list<long> (on a 32 bits computer) or     |
|                         | sequence<int> (on a 64 bits computer)     | list<int> (on a 64 bits computer)         |
+-------------------------+-------------------------------------------+-------------------------------------------+
|    DEVVAR_FLOATARRAY    | sequence<float>                           | list<float>                               |
+-------------------------+-------------------------------------------+-------------------------------------------+
|   DEVVAR_DOUBLEARRAY    | sequence<float>                           | list<float>                               |
+-------------------------+-------------------------------------------+-------------------------------------------+
|   DEVVAR_USHORTARRAY    | sequence<int>                             | list<int>                                 |
+-------------------------+-------------------------------------------+-------------------------------------------+
|   DEVVAR_ULONGARRAY     | sequence<int>                             | list<int>                                 |
+-------------------------+-------------------------------------------+-------------------------------------------+
|  DEVVAR_ULONG64ARRAY    | sequence<long> (on a 32 bits computer) or | list<long> (on a 32 bits computer) or     |
|                         | sequence<int> (on a 64 bits computer)     | list<int> (on a 64 bits computer)         |
+-------------------------+-------------------------------------------+-------------------------------------------+
|   DEVVAR_STRINGARRAY    | sequence<str>                             | list<str>                                 |
+-------------------------+-------------------------------------------+-------------------------------------------+
|                         | A sequence with two elements:             | A list with two elements:                 |
| DEVVAR_LONGSTRINGARRAY  | 1. sequence<int>                          |  1. list<int>                             |
|                         | 2. sequence<str>                          |  2. list<str>                             |
+-------------------------+-------------------------------------------+-------------------------------------------+
|                         | A sequence with two elements:             | A list with two elements:                 |
|DEVVAR_DOUBLESTRINGARRAY | 1. sequence<float>                        |  1. list<float>                           |
|                         | 2. sequence<str>                          |  2. list<str>                             |
+-------------------------+-------------------------------------------+-------------------------------------------+

Note that starting from PyTango 7 you **cannot assume anything** about the concrete 
sequence implementation for the tango array types in PyTango.
This means that the following code (valid in PyTango <= 3.0.4)::

    import PyTango
    dp = PyTango.DeviceProxy("my/device/experiment")
    da = dp.read_attribute("array_attr")
    if isinstance(da.value, list):
        print "array_attr is NOT a scalar attribute"

must be replaced with::

    import operator, types
    import PyTango
    dp = PyTango.DeviceProxy("my/device/experiment")
    da = dp.read_attribute("array_attr")
    if operator.isSequence(da.value) and not type(da.value) in types.StringTypes:
        print "array_attr is NOT a scalar attribute"

Note that the above example is intended for demonstration purposes only. For 
reference, the proper code would be::

    import PyTango
    dp = PyTango.DeviceProxy("my/device/experiment")
    da = dp.read_attribute("array_attr")
    if not da.data_format is PyTango.AttrDataFormat.SCALAR:
        print "array_attr is NOT a scalar attribute"
    
Server
~~~~~~

#. replace `PyTango.PyUtil` with :class:`Util`

#. replace `PyTango.PyDeviceClass` with :class:`DeviceClass`

#. state and status overwrite
    in PyTango <= 3.0.4, in order to overwrite the default state and status in a device
    server, you had to reimplement **State()** and **Status()** methods respectively.

    in PyTango 7 the methods have been renamed to **dev_state()** and **dev_status()** in
    order to match the C++ API.

General
~~~~~~~

#. AttributeValue does **NOT** exist anymore.
    - the result of a read_attribute call on a :class:`DeviceProxy` / :class:`Group`
      is now a :class:`DeviceAttribute` object
    - write_attribute does not accept AttributeValue anymore
    
    (See :class:`DeviceProxy` API documentation for more details)
    
#. command_inout for commands with parameter type DevVar****StringArray don't accept items in second sequence not being strings:
    For example, a tango command 'DevVoid Go(DevVarDoubleArray)' in tango 3.0.4
    could be executed by calling::
        
        dev_proxy.command_inout( 'Go', [[1.0, 2.0], [1, 2, 3]] )
    
    and the second list would internally be converted to ['1', '2', '3'].
    Starting from PyTango 7 this is not allowed anymore. So the above code 
    must be changed to::
    
        dev_proxy.command_inout( 'Go', [[1.0, 2.0], ['1', '2', '3']] )

#. :class:`EventType` enumeration constants changed to match C++ enumeration
    - CHANGE -> CHANGE_EVENT
    - QUALITY -> QUALITY_EVENT
    - PERIODIC -> PERIODIC_EVENT
    - ARCHIVE -> ARCHIVE_EVENT
    - USER -> USER_EVENT
    - ATTR_CONF_EVENT remains

#. Exception handling
    in 3.0.4 :class:`DevFailed` was a tuple of dictionaries. 
    Now :class:`DevFailed` is a tuple of :class:`DevError`.
    This means that code::

        try:
            tango_fail()
        except PyTango.DevFailed as e:
            print e.args[0]['reason']

    needs to be replaced with::

        try:
            tango_fail()
        except PyTango.DevFailed as e:
            print e.args[0].reason


Optional
~~~~~~~~

The following is a list of API improvements. Some where added for performance 
reasons, others to allow for a more pythonic interface, others still to reflect 
more adequately the C++ interface. They are not mandatory since the original 
interface will still be available.

Server side V3 to V4 upgrade
############################

If you want your server to support the V4 interface provided by Tango 7
instead of the V3 provided by Tango 6:

- replace the inheritance of your device class from :class:`Device_3Impl` to :class:`Device_4Impl`
- in the `init_device` method replace the call::
     
     Device_3Impl.init_device(self)

  with::
  
     Device_4Impl.init_device(self)

  or better yet, if your device class only inherits from :class:`Device_4Impl`::
  
     super(<your class>, self).init_device()

Improved server side image attribute read API
#############################################

In PyTango <= 3.0.4, to set the value of an image attribute you needed it
as a flat list. Consider you want to set as value the following image::

    # Image:
    #  | 1  2 |
    #  | 3  4 |
    
In order to tell tango the dimensions of the image you had to specify them as::

    image = [ 1, 2, 3, 4]
    dim_x = 2
    dim_y = 2
    attr.set_value(image, dim_x, dim_y)

In PyTango 8 it is still supported, but the preferred way is to use a
sequence of sequences (instead of a flat sequence), so the dimensions
are inherent and not needed anymore::

    image = [ [1, 2], [3, 4]]
    attr.set_value(image)

If you use a numpy array as the sequence of sequences you can get better
performance::

    image = numpy.array([ [1, 2], [3, 4]], dtype=numpy.int32)
    attr.set_value(image)

Likewise, calls to::

    PyTango.set_attribute_value_date_quality(attr, value, date, quality, dim_x, dim_y)

can be replaced with::

    attr.set_value_date_quality(value, date, quality)

Improved server side attribute write API
########################################

Imagine the following value is written to our IMAGE attribute::

    # Image:
    #  | 1  2 |
    #  | 3  4 |

This is what you would do with PyTango <= 3.0.4::

    flatList = []
    attr.get_write_value(flatList)
    print "flatList =", flatList
    # flatList = [ 1, 2, 3, 4 ]

You can still do it with PyTango 8. However I recommend::

    image = attr.get_write_value()
    print "image =", image
    # image = numpy.array([[1, 2], [3, 4]])

If PyTango 8 is compiled without numpy support, you will get a sequence
of sequences, which makes more sense than a flat list.

If PyTango 8 is compiled with numpy support it does not only makes more sense
but it is also considerably **faster and memory friendlier**.

If PyTango is compiled with numpy support but you prefer a list of lists for
some attribute, you can do::

    image = attr.get_write_value(PyTango.ExtractAs.List)
    print "image =", image
    # image = [[1, 2], [3, 4]]

Also the SCALAR attribute case is much **cleaner** now. Instead of::

    data = []
    attr.get_write_value(data)
    actualData = data[0]

You can just write::

    actualData = attr.get_write_value()

Why is there a "-Wstrict-prototypes" warning when I compile PyTango?
--------------------------------------------------------------------

The PyTango prefered build system (distutils) uses the same flags used to compile
Python to compile PyTango. It happens that Python is compiled as a pure C library
while PyTango is a C++ library. Unfortunately one of the flags used by Python is
the "-Wstrict-prototypes" which makes sence in a C compilation but not in a C++ 
compilation.
For reference here is the complete error message you may have:
    
    `cc1plus: warning: command line option "-Wstrict-prototypes" is valid for Ada/C/ObjC but not for C++`

Do not worry about this warning since the compiler is ignoring the presence of this flag
in the compilation.

Why are there so many warnings when generating the documentation?
-----------------------------------------------------------------
PyTango uses boost python for the binding between C++ and Python and sphinx for
document generation.
When sphinx generates the PyTango API documentation it uses introspection to search
for documentation in the python code. It happens that boost overrides some python
introspection API for functions and methods which sphinx expects to have. Therefore
you should see many warnings of type:

    `(WARNING/2) error while formatting signature for PyTango.Device_4Impl.always_executed_hook: **arg is not a Python function**`

Do not worry since sphinx is able to generate the proper documentation.

