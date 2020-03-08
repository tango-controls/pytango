Server green modes
------------------

PyTango server API from version 9.2.0 supports two green modes:
:obj:`~tango.GreenMode.Gevent` and :obj:`~tango.GreenMode.Asyncio`.
Both can be used in writing new device servers in an asynchronous way.

gevent mode
~~~~~~~~~~~

This mode lets you convert your existing devices to asynchronous devices
easily. You just add `green_mode = tango.GreenMode.Gevent` line to your device
class. Consider this example::

    class GeventDevice(Device):
        green_mode = tango.GreenMode.Gevent

Every method in your device class will be treated as a
coroutine implicitly. This can be beneficial, but also potentially dangerous
as it is a lot harder to debug. You should use this green mode with care.
:obj:`~tango.GreenMode.Gevent` green mode is useful when you don't want to
change too much in your existing code (or you don't feel comfortable with
writing syntax of asynchronous calls).

Another thing to keep in mind is that when using :obj:`~tango.GreenMode.Gevent`
green mode is that the Tango monitor lock is disabled, so the client requests can
be processed concurrently.

Greenlets can also be used to spawn tasks in the background.


asyncio mode
~~~~~~~~~~~~

The way asyncio green mode on the server side works is it redirects all user
code to an event loop. This means that all user methods become coroutines, so
in Python > 3.5 you should define them with `async` keyword. In Python < 3.5,
you should use a `@coroutine` decorator. This also means that in order to
convert existing code of your devices to :obj:`~tango.GreenMode.Asyncio` green
mode you will have to introduce at least those changes. But, of course, to
truly benefit from this green mode (and asynchronous approach in general),
you should introduce more far-fetched changes!

The main benefit of asynchronous programing approach is that it lets you
control precisely when code is run sequentially without interruptions and
when control can be given back to the event loop. It's especially useful
if you want to perform some long operations and don't want to prevent clients
from accessing other parts of your device (attributes, in particular). This
means that in :obj:`~tango.GreenMode.Asyncio` green mode there is no monitor
lock!

The example below shows how asyncio can be used to write an asynchronous
Tango device:

.. literalinclude:: ../../examples/asyncio_green_mode/asyncio_device_example.py
    :linenos:
