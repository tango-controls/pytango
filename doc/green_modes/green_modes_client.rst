Client green modes
------------------

You can also change the active global green mode at any time in your program::

    >>> from tango import DeviceProxy, GreenMode
    >>> from tango import set_green_mode, get_green_mode

    >>> get_green_mode()
    tango.GreenMode.Synchronous

    >>> dev = DeviceProxy("sys/tg_test/1")
    >>> dev.get_green_mode()
    tango.GreenMode.Synchronous

    >>> set_green_mode(GreenMode.Futures)
    >>> get_green_mode()
    tango.GreenMode.Futures

    >>> dev.get_green_mode()
    tango.GreenMode.Futures

As you can see by the example, the global green mode will affect any previously
created :class:`DeviceProxy` using the default *DeviceProxy* constructor
parameters.

You can specificy green mode on a :class:`DeviceProxy` at creation time.
You can also change the green mode at any time::

    >>> from tango.futures import DeviceProxy

    >>> dev = DeviceProxy("sys/tg_test/1")
    >>> dev.get_green_mode()
    tango.GreenMode.Futures

    >>> dev.set_green_mode(GreenMode.Synchronous)
    >>> dev.get_green_mode()
    tango.GreenMode.Synchronous


.. include:: client_futures_mode.rst

.. include:: client_gevent_mode.rst

.. include:: client_asyncio_mode.rst