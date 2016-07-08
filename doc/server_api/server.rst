
.. currentmodule:: tango.server

.. _pytango-hlapi:

High level server API
=====================

.. automodule:: tango.server

.. hlist::

   * :class:`~tango.server.Device`
   * :class:`~tango.server.attribute`
   * :class:`~tango.server.command`
   * :class:`~tango.server.pipe`
   * :class:`~tango.server.device_property`
   * :class:`~tango.server.class_property`
   * :func:`~tango.server.run`
   * :func:`~tango.server.server_run`

This module provides a high level device server API. It implements
:ref:`TEP1 <pytango-TEP1>`. It exposes an easier API for developing a Tango
device server.

Here is a simple example on how to write a *Clock* device server using the
high level API::
    
    import time
    from tango.server import run
    from tango.server import Device, DeviceMeta
    from tango.server import attribute, command   


    class Clock(Device):
        __metaclass__ = DeviceMeta

        time = attribute()

        def read_time(self):
            return time.time()

        @command(din_type=str, dout_type=str)
        def strftime(self, format):
            return time.strftime(format)


    if __name__ == "__main__":
        run((Clock,))


Here is a more complete  example on how to write a *PowerSupply* device server
using the high level API. The example contains:

#. a read-only double scalar attribute called *voltage*
#. a read/write double scalar expert attribute *current*
#. a read-only double image attribute called *noise*
#. a *ramp* command
#. a *host* device property
#. a *port* class property

.. code-block:: python
    :linenos:

    from time import time
    from numpy.random import random_sample

    from tango import AttrQuality, AttrWriteType, DispLevel, server_run
    from tango.server import Device, DeviceMeta, attribute, command
    from tango.server import class_property, device_property

    class PowerSupply(Device):
        __metaclass__ = DeviceMeta

        voltage = attribute()

        current = attribute(label="Current", dtype=float,
                            display_level=DispLevel.EXPERT,
                            access=AttrWriteType.READ_WRITE,
                            unit="A", format="8.4f",
                            min_value=0.0, max_value=8.5,
                            min_alarm=0.1, max_alarm=8.4,
                            min_warning=0.5, max_warning=8.0,
                            fget="get_current", fset="set_current",
                            doc="the power supply current")
    
        noise = attribute(label="Noise", dtype=((float,),),
                          max_dim_x=1024, max_dim_y=1024,
                          fget="get_noise")
 
        host = device_property(dtype=str)
        port = class_property(dtype=int, default_value=9788)

        def read_voltage(self):
            self.info_stream("get voltage(%s, %d)" % (self.host, self.port))
            return 10.0

        def get_current(self):
            return 2.3456, time(), AttrQuality.ATTR_WARNING
    
        def set_current(self, current):
            print("Current set to %f" % current)
    
        def get_noise(self):
            return random_sample((1024, 1024))

        @command(dtype_in=float)
        def ramp(self, value):
            print("Ramping up...")

    if __name__ == "__main__":
        server_run((PowerSupply,))

*Pretty cool, uh?*

.. note::
    the ``__metaclass__`` statement is mandatory due to a limitation in the
    *boost-python* library used by PyTango.
    
    If you are using python 3 you can write instead::
        
        class PowerSupply(Device, metaclass=DeviceMeta)
            pass

.. _pytango-hlapi-datatypes:

.. rubric:: Data types

When declaring attributes, properties or commands, one of the most important
information is the data type. It is given by the keyword argument *dtype*.
In order to provide a more *pythonic* interface, this argument is not restricted
to the :obj:`~tango.CmdArgType` options.

For example, to define a *SCALAR* :obj:`~tango.CmdArgType.DevLong`
attribute you have several possibilities:

#. :obj:`int`
#. 'int'
#. 'int32'
#. 'integer' 
#. :obj:`tango.CmdArgType.DevLong`
#. 'DevLong' 
#. :obj:`numpy.int32`

To define a *SPECTRUM* attribute simply wrap the scalar data type in any
python sequence:

* using a *tuple*: ``(:obj:`int`,)`` or
* using a *list*: ``[:obj:`int`]`` or
* any other sequence type

To define an *IMAGE* attribute simply wrap the scalar data type in any
python sequence of sequences:

* using a *tuple*: ``((:obj:`int`,),)`` or
* using a *list*: ``[[:obj:`int`]]`` or
* any other sequence type

Below is the complete table of equivalences.

========================================  ========================================
dtype argument                            converts to tango type                             
========================================  ========================================
 ``None``                                  ``DevVoid``
 ``'None'``                                ``DevVoid``
 ``DevVoid``                               ``DevVoid``
 ``'DevVoid'``                             ``DevVoid``

 ``DevState``                              ``DevState``                           
 ``'DevState'``                            ``DevState``                           

 :py:obj:`bool`                            ``DevBoolean``
 ``'bool'``                                ``DevBoolean``
 ``'boolean'``                             ``DevBoolean``
 ``DevBoolean``                            ``DevBoolean``
 ``'DevBoolean'``                          ``DevBoolean``
 :py:obj:`numpy.bool_`                     ``DevBoolean``

 ``'char'``                                ``DevUChar``
 ``'chr'``                                 ``DevUChar``
 ``'byte'``                                ``DevUChar``
 ``chr``                                   ``DevUChar``
 ``DevUChar``                              ``DevUChar``
 ``'DevUChar'``                            ``DevUChar``
 :py:obj:`numpy.uint8`                     ``DevUChar``

 ``'int16'``                               ``DevShort``
 ``DevShort``                              ``DevShort``
 ``'DevShort'``                            ``DevShort``
 :py:obj:`numpy.int16`                     ``DevShort``

 ``'uint16'``                              ``DevUShort``
 ``DevUShort``                             ``DevUShort``
 ``'DevUShort'``                           ``DevUShort``
 :py:obj:`numpy.uint16`                    ``DevUShort``

 :py:obj:`int`                             ``DevLong``
 ``'int'``                                 ``DevLong``
 ``'int32'``                               ``DevLong``
 ``DevLong``                               ``DevLong``
 ``'DevLong'``                             ``DevLong``
 :py:obj:`numpy.int32`                     ``DevLong``

 ``'uint'``                                ``DevULong``
 ``'uint32'``                              ``DevULong``
 ``DevULong``                              ``DevULong``
 ``'DevULong'``                            ``DevULong``
 :py:obj:`numpy.uint32`                    ``DevULong``

 ``'int64'``                               ``DevLong64``
 ``DevLong64``                             ``DevLong64``
 ``'DevLong64'``                           ``DevLong64``
 :py:obj:`numpy.int64`                     ``DevLong64``
 
 ``'uint64'``                              ``DevULong64``
 ``DevULong64``                            ``DevULong64``
 ``'DevULong64'``                          ``DevULong64``
 :py:obj:`numpy.uint64`                    ``DevULong64``

 ``DevInt``                                ``DevInt``                             
 ``'DevInt'``                              ``DevInt``                             
 
 ``'float32'``                             ``DevFloat``
 ``DevFloat``                              ``DevFloat``
 ``'DevFloat'``                            ``DevFloat``
 :py:obj:`numpy.float32`                   ``DevFloat``
 
 :py:obj:`float`                           ``DevDouble``
 ``'double'``                              ``DevDouble``
 ``'float'``                               ``DevDouble``
 ``'float64'``                             ``DevDouble``
 ``DevDouble``                             ``DevDouble``
 ``'DevDouble'``                           ``DevDouble``
 :py:obj:`numpy.float64`                   ``DevDouble``
 
 :py:obj:`str`                             ``DevString``
 ``'str'``                                 ``DevString``
 ``'string'``                              ``DevString``
 ``'text'``                                ``DevString``
 ``DevString``                             ``DevString``
 ``'DevString'``                           ``DevString``
 
 :py:obj:`bytearray`                       ``DevEncoded``
 ``'bytearray'``                           ``DevEncoded``
 ``'bytes'``                               ``DevEncoded``
 ``DevEncoded``                            ``DevEncoded``
 ``'DevEncoded'``                          ``DevEncoded``

 ``DevVarBooleanArray``                    ``DevVarBooleanArray``
 ``'DevVarBooleanArray'``                  ``DevVarBooleanArray``
 
 ``DevVarCharArray``                       ``DevVarCharArray``
 ``'DevVarCharArray'``                     ``DevVarCharArray``
 
 ``DevVarShortArray``                      ``DevVarShortArray``
 ``'DevVarShortArray'``                    ``DevVarShortArray``
 
 ``DevVarLongArray``                       ``DevVarLongArray``
 ``'DevVarLongArray'``                     ``DevVarLongArray``
 
 ``DevVarLong64Array``                     ``DevVarLong64Array``
 ``'DevVarLong64Array'``                   ``DevVarLong64Array``
 
 ``DevVarULong64Array``                    ``DevVarULong64Array``
 ``'DevVarULong64Array'``                  ``DevVarULong64Array``
 
 ``DevVarFloatArray``                      ``DevVarFloatArray``
 ``'DevVarFloatArray'``                    ``DevVarFloatArray``
 
 ``DevVarDoubleArray``                     ``DevVarDoubleArray``
 ``'DevVarDoubleArray'``                   ``DevVarDoubleArray``
 
 ``DevVarUShortArray``                     ``DevVarUShortArray``
 ``'DevVarUShortArray'``                   ``DevVarUShortArray``
 
 ``DevVarULongArray``                      ``DevVarULongArray``
 ``'DevVarULongArray'``                    ``DevVarULongArray``
 
 ``DevVarStringArray``                     ``DevVarStringArray``
 ``'DevVarStringArray'``                   ``DevVarStringArray``
 
 ``DevVarLongStringArray``                 ``DevVarLongStringArray``
 ``'DevVarLongStringArray'``               ``DevVarLongStringArray``
 
 ``DevVarDoubleStringArray``               ``DevVarDoubleStringArray``
 ``'DevVarDoubleStringArray'``             ``DevVarDoubleStringArray``

 ``DevPipeBlob``                           ``DevPipeBlob``
 ``'DevPipeBlob'``                         ``DevPipeBlob``
========================================  ========================================

.. autoclass:: Device
   :show-inheritance:
   :inherited-members:
   :members:

.. autoclass:: attribute

.. autofunction:: command

.. autoclass:: pipe

.. autoclass:: device_property

.. autoclass:: class_property

.. autofunction:: run

.. autofunction:: server_run
