
.. currentmodule:: PyTango.server

.. _pytango-hlapi:

High Level API
==============

This module provides a high level device server API. It implements
:ref:`TEP1 <pytango-TEP1>`. It exposes an easier API for developing a Tango
device server.

Here is a simple example on how to write a *Clock* device server using the
high level API::
    
    import time
    from PyTango.server import server_run
    from PyTango.server import Device, DeviceMeta
    from PyTango.server import attribute, command   

    class Clock(Device):
        __metaclass__ = DeviceMeta

	time = attribute()

	def read_time(self):
	    return time.time()

	@command(din_type=str, dout_type=str)
	def strftime(self, format):
	    return time.strftime(format)

    if __name__ == "__main__":
        server_run((Clock,))

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

    from PyTango import AttrQuality, AttrWriteType, DispLevel, server_run
    from PyTango.server import Device, DeviceMeta, attribute, command
    from PyTango.server import class_property, device_property

    class PowerSupply(Device):
        __metaclass__ = DeviceMeta

        voltage = attribute()

        current = attribute(label="Current",
                            dtype=float,
                            display_level=DispLevel.EXPERT,
                            access=AttrWriteType.READ_WRITE,
                            unit="A",
                            format="8.4f",
			    min_value=0.0, max_value=8.5,
			    min_alarm=0.1, max_alarm=8.4,
			    min_warning=0.5, max_warning=8.0,
                            fget="get_current",
                            fset="set_current",
                            doc="the power supply current")
	
	noise = attribute(label="Noise",
			  dtype=((float,),),
			  max_dim_x=1024, max_dim_y=1024,
			  fget="get_noise")
 
        host = device_property(dtype=str)
        port = class_property(dtype=int, default_value=9788)

        def read_voltage(self):
            self.info_stream("get voltage(%s, %d)" %(self.host, self.port))
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


*Pretty cool, uh?*

API
---

.. hlist::

   * :class:`~PyTango.server.Device`
   * :class:`~PyTango.server.attribute`
   * :class:`~PyTango.server.command`
   * :class:`~PyTango.server.device_property`
   * :class:`~PyTango.server.class_property`
   * :func:`~PyTango.server.server_run`

.. automodule:: PyTango.server

   .. autoclass:: Device
      :show-inheritance:
      :inherited-members:
      :members:

   .. autoclass:: attribute

   .. autofunction:: command

   .. autoclass:: device_property

   .. autoclass:: class_property

   .. autofunction:: server_run
