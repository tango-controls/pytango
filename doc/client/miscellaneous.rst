

.. currentmodule:: PyTango

API util
--------

.. autoclass:: PyTango.ApiUtil
    :members:

Information classes
-------------------

See also `Event configuration information`_

Attribute
~~~~~~~~~
.. autoclass:: PyTango.AttributeAlarmInfo
    :members:

.. autoclass:: PyTango.AttributeDimension
    :members:

.. autoclass:: PyTango.AttributeInfo
    :members:

.. autoclass:: PyTango.AttributeInfoEx
    :members:
    
see also :class:`PyTango.AttributeInfo`

.. autoclass:: PyTango.DeviceAttributeConfig
    :members:

Command
~~~~~~~

.. autoclass:: PyTango.DevCommandInfo
   :members:

.. autoclass:: PyTango.CommandInfo
   :members:

Other
~~~~~

.. autoclass:: PyTango.DeviceInfo
    :members:

.. autoclass:: PyTango.LockerInfo
    :members:

.. autoclass:: PyTango.PollDevice
    :members:


Storage classes
---------------

Attribute: DeviceAttribute
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DeviceAttribute
    :members:


Command: DeviceData
~~~~~~~~~~~~~~~~~~~

Device data is the type used internally by Tango to deal with command parameters
and return values. You don't usually need to deal with it, as command_inout
will automatically convert the parameters from any other type and the result
value to another type.

You can still use them, using command_inout_raw to get the result in a DeviceData.

You also may deal with it when reading command history.

.. autoclass:: DeviceData
    :members:


Callback related classes
------------------------

If you subscribe a callback in a DeviceProxy, it will be run with a parameter.
This parameter depends will be of one of the following classes depending on
the callback type.

.. autoclass:: PyTango.AttrReadEvent
    :members:

.. autoclass:: PyTango.AttrWrittenEvent
    :members:

.. autoclass:: PyTango.CmdDoneEvent
    :members:


Event related classes
---------------------

Event configuration information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PyTango.AttributeEventInfo
    :members:

.. autoclass:: PyTango.ArchiveEventInfo
    :members:

.. autoclass:: PyTango.ChangeEventInfo
    :members:

.. autoclass:: PyTango.PeriodicEventInfo
    :members:

Event arrived structures
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PyTango.EventData
    :members:

.. autoclass:: PyTango.AttrConfEventData
    :members:

.. autoclass:: PyTango.DataReadyEventData
    :members:


History classes
---------------

.. autoclass:: PyTango.DeviceAttributeHistory
    :show-inheritance:
    :members:

See :class:`DeviceAttribute`.

.. autoclass:: PyTango.DeviceDataHistory
    :show-inheritance:
    :members:

See :class:`DeviceData`.
