Server green modes
------------------

.. warning::
   Green modes for the server side are still very much experimental.
   If you encounter any issues, please report them on the GitHub issues_ page.

PyTango server API from version 9.2.0 supports two green modes:
:obj:`~tango.GreenMode.Gevent` and :obj:`~tango.GreenMode.Asyncio`.
Both can be used in writing new device servers in an asynchronous way.

gevent mode
~~~~~~~~~~~

This mode lets you convert your existing devices to asynchronous devices
easily. You just add:

 >>> green_mode = tango.GreenMode.Gevent

to your device code. Every method in your device class will be treated as a
coroutine implicitly. This can be beneficial, but also potentially dangerous
as it is a lot harder to debug. You should use this green mode with care.

Another thing to have in mind is that the Tango monitor lock is present - you
can't have two read operations happening concurrently. Any subsequent ones
will always have to wait for the first one to finish.
Greenlets (a task in a background, but handled within the event loop) can be
used.


asyncio mode
~~~~~~~~~~~~

Redirects all user code to an event loop. All user methods become
coroutines, so you should define them with "async" keyword. DS. code written
with asyncio. There's no monitor lock!

.. literalinclude:: ../../examples/asyncio_device/asyncio_device_example.py
    :linenos:
