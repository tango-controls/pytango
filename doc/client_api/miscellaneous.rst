.. currentmodule:: tango

API util
--------

.. autoclass:: ApiUtil
    :members:

Information classes
-------------------

See also `Event configuration information`_

Attribute
~~~~~~~~~

.. autoclass:: AttributeAlarmInfo
    :members:

.. autoclass:: AttributeDimension
    :members:

.. autoclass:: AttributeInfo
    :members:

.. autoclass:: AttributeInfoEx
    :members:
    
see also :class:`AttributeInfo`

.. autoclass:: DeviceAttributeConfig
    :members:

Command
~~~~~~~

.. autoclass:: DevCommandInfo
   :members:

.. autoclass:: CommandInfo
   :members:

Other
~~~~~

.. autoclass:: DeviceInfo
    :members:

.. autoclass:: LockerInfo
    :members:

.. autoclass:: PollDevice
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

.. autoclass:: AttrReadEvent
    :members:

.. autoclass:: AttrWrittenEvent
    :members:

.. autoclass:: CmdDoneEvent
    :members:


Event related classes
---------------------

Event configuration information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AttributeEventInfo
    :members:

.. autoclass:: ArchiveEventInfo
    :members:

.. autoclass:: ChangeEventInfo
    :members:

.. autoclass:: PeriodicEventInfo
    :members:

Event arrived structures
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: EventData
    :members:

.. autoclass:: AttrConfEventData
    :members:

.. autoclass:: DataReadyEventData
    :members:


History classes
---------------

.. autoclass:: DeviceAttributeHistory
    :show-inheritance:
    :members:

See :class:`DeviceAttribute`.

.. autoclass:: DeviceDataHistory
    :show-inheritance:
    :members:

See :class:`DeviceData`.

