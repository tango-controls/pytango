
.. currentmodule:: PyTango.hlapi

.. _pytango-hlapi:

HLAPI
=====

The High Level API for writting Tango device servers.

Introduction
------------

This module provides a high level device server API. It implements
:ref:`TEP1 <pytango-TEP1>`. It exposes an easier API for developing a Tango
device server.

Here is the summary of features which this module exposes and are not available
on the low level :mod:`PyTango` server API:

#. Automatic inheritance from the latest :class:`~PyTango.DeviceImpl`
#. default implementation of :meth:`Device.__init__`
   calls :meth:`Device.init_device`. Around 90% of the
   different device classes which inherit from low level
   :class:`~PyTango.DeviceImpl` only implement `__init__` to call their
   `init_device`
#. has a default implementation of :meth:`Device.init_device`
   which calls :meth:`Device.get_device_properties`. Again,
   90% of existing device classes do that
#. Automatically creates a hidden :class:`~PyTango.DeviceClass` class 
#. recognizes :func:`attribute` members and automatically 
   registers them as tango attributes in the hidden
   :class:`~PyTango.DeviceClass`
#. recognizes :func:`command` decorated functions and
   automatically registers them as tango commands in the hidden
   :class:`~PyTango.DeviceClass`
#. recognizes :func:`device_property` members and
   automatically registers them as tango device properties in the hidden
   :class:`~PyTango.DeviceClass`
#. recognizes :func:`class_property` members and
   automatically registers them as tango class properties in the hidden
   :class:`~PyTango.DeviceClass`
#. read and write attribute methods don't need :class:`~PyTango.Attribute`
   parameter. Access to :class:`~PyTango.Attribute` object is with simple::
   
       self.<attr name>
       
#. read attribute methods can set attribute return value with::
       
       def read_voltage(self):
           return value
       
       # or 
       
       def read_voltage(self):
           self.voltage = value
       
   instead of::
   
       def read_voltage(self, attr):
           attr.set_value(value)

:class:`Device` works very well in conjuction with:

#. :func:`attribute`
#. :class:`command`
#. :meth:`device_property`
#. :meth:`class_property`
#. :meth:`~PyTango.server_run`

Here is an example of a PowerSupply device with:

#. a read-only double scalar `voltage` attribute
#. a read/write double scalar `current` attribute
#. a `ramp` command
#. a `host` device property

.. code-block:: python
    :linenos:

    from time import time
        
    from PyTango import AttrQuality, DebugIt, server_run
    from PyTango.hlapi import Device, DeviceMeta
    from PyTango.hlapi import attribute, command, device_property

    class PowerSupply(Device):
        __metaclass__ = DeviceMeta
        
        voltage = attribute()        

        current = attribute(label="Current", unit="A",
                            fread="read_current",
                            fwrite="write_current")
        
        host = device_property()
        
        def read_voltage(self):
            return 10.0
            
        def read_current(self):
            return 2.5, time(), AttrQuality.ON
        
        @DebugIt()
        def write_current(self):
            new_current = self.current.get_write_value()
        
        @command()
        def ramp(self):
            self.info_stream("Ramping on " + self.host + "...")

    def main():
        classes = PowerSupply,
        server_run(classes)
    
    if __name__ == "__main__":
        main()

And here is the equivalent code using the low-level API:

.. code-block:: python
    :linenos:

    import sys
    import time

    import PyTango

    class PowerSupply(PyTango.Device_4Impl):

        def __init__(self, devclass, name):
            PyTango.Device_4Impl.__init__(self, devclass, name)
            self.init_device()
        
        def init_device(self):
            self.get_device_properties()
        
        def read_voltage(self, attr):
            attr.set_value(10.0)
            
        def read_current(self):
            attr.set_value_date_quality(2.5, time.time(), PyTango.AttrQuality.ON)
        
        @PyTango.DebugIt()
        def write_current(self, attr):
            new_current = attr.get_write_value()
        
        def ramp(self):
            self.info_stream("Ramping on " + self.host + "...")


    class PowerSupplyClass(PyTango.DeviceClass):
        
        class_property_list = {}

        device_property_list = {
            'host':
                [PyTango.DevString, "host of power supply", "localhost"],
        }

        cmd_list = {
            'ramp':
                [ [PyTango.DevVoid, "nothing"],
                  [PyTango.DevVoid, "nothing"] ],
        }

        attr_list = {
            'voltage':
                [[PyTango.DevDouble,
                PyTango.SCALAR,
                PyTango.READ]],
            'current':
                [[PyTango.DevDouble,
                PyTango.SCALAR,
                PyTango.READ_WRITE], 
                { 'label' : 'Current', 'unit' : 'A' }],
        }
        

    def main():
        try:
            py = PyTango.Util(sys.argv)
            py.add_class(PowerSupplyClass,PowerSupply,'PowerSupply')

            U = PyTango.Util.instance()
            U.server_init()
            U.server_run()

        except PyTango.DevFailed,e:
            print '-------> Received a DevFailed exception:',e
        except Exception,e:
            print '-------> An unforeseen exception occured....',e

    if __name__ == "__main__":
        main()
        
       
*Pretty cool, uh?*

API
---

.. automodule:: PyTango.hlapi

   .. autoclass:: Device
      :show-inheritance:
      :inherited-members:
      :members:

   .. autofunction:: attribute

   .. autofunction:: command

   .. autofunction:: device_property

   .. autofunction:: class_property
