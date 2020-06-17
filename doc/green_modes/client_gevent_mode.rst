gevent mode
~~~~~~~~~~~

.. warning::
   Before using gevent mode please note that at the time of writing this
   documentation, *tango.gevent* requires the latest version 1.0 of
   gevent (which has been released the day before :-P).

Using gevent_ cooperative mode in PyTango is relatively easy::

    >>> from tango.gevent import DeviceProxy

    >>> dev = DeviceProxy("sys/tg_test/1")
    >>> dev.get_green_mode()
    tango.GreenMode.Gevent

    >>> print(dev.state())
    RUNNING

The :func:`tango.gevent.DeviceProxy` API is exactly the same as the standard
:class:`~tango.DeviceProxy`. The difference is in the semantics of the methods
that involve synchronous network calls (constructor included) which may block
the execution for a relatively big amount of time.
The list of methods that have been modified to accept *gevent* semantics are,
on the :func:`tango.gevent.DeviceProxy`:

* Constructor
* :meth:`~DeviceProxy.state`
* :meth:`~DeviceProxy.status`
* :meth:`~DeviceProxy.read_attribute`
* :meth:`~DeviceProxy.write_attribute`
* :meth:`~DeviceProxy.write_read_attribute`
* :meth:`~DeviceProxy.read_attributes`
* :meth:`~DeviceProxy.write_attributes`
* :meth:`~DeviceProxy.ping`

So how does this work in fact? I see no difference from using the *standard*
:class:`~tango.DeviceProxy`.
Well, this is, in fact, one of the goals: be able to use a gevent cooperation
without changing the API. Behind the scenes the methods mentioned before have
been modified to be able to work cooperatively with other greenlets.

All of the above methods have been boosted with two extra keyword arguments
*wait* and *timeout* which allow to fine tune the behaviour.
The *wait* parameter is by default set to `True` meaning wait for the request
to finish (the default semantics when not using green mode).
If *wait* is set to `True`, the timeout determines the maximum time to wait for
the method to execute. The default timeout is `None` which means wait forever.
If *wait* is set to `False`, the *timeout* is ignored.

If *wait* is set to `True`, the result is the same as executing the
*standard* method on a :class:`~tango.DeviceProxy`.
If, *wait* is set to `False`, the result will be a
:class:`gevent.event.AsyncResult`. In this case, to get the actual value
you will need to do something like::

    >>> from tango.gevent import DeviceProxy

    >>> dev = DeviceProxy("sys/tg_test/1")
    >>> result = dev.state(wait=False)
    >>> result
    <gevent.event.AsyncResult at 0x1a74050>

    >>> # this will be the blocking code
    >>> state = result.get()
    >>> print(state)
    RUNNING

Here is another example using :meth:`~DeviceProxy.read_attribute`::

    >>> from tango.gevent import DeviceProxy

    >>> dev = DeviceProxy("sys/tg_test/1")
    >>> result = dev.read_attribute('wave', wait=False)
    >>> result
    <gevent.event.AsyncResult at 0x1aff54e>

    >>> dev_attr = result.get()
    >>> print(dev_attr)
    DeviceAttribute[
    data_format = tango.AttrDataFormat.SPECTRUM
          dim_x = 256
          dim_y = 0
     has_failed = False
       is_empty = False
           name = 'wave'
        nb_read = 256
     nb_written = 0
        quality = tango.AttrQuality.ATTR_VALID
    r_dimension = AttributeDimension(dim_x = 256, dim_y = 0)
           time = TimeVal(tv_nsec = 0, tv_sec = 1383923292, tv_usec = 886720)
           type = tango.CmdArgType.DevDouble
          value = array([ -9.61260664e-01,  -9.65924853e-01,  -9.70294813e-01,
            -9.74369212e-01,  -9.78146810e-01,  -9.81626455e-01,
            -9.84807087e-01,  -9.87687739e-01,  -9.90267531e-01,
            ...
            5.15044507e-1])
        w_dim_x = 0
        w_dim_y = 0
    w_dimension = AttributeDimension(dim_x = 0, dim_y = 0)
        w_value = None]

.. note::
   due to the internal workings of gevent, setting the *wait* flag to
   `True` (default) doesn't prevent other greenlets from running in *parallel*.
   This is, in fact, one of the major bonus of working with :mod:`gevent` when
   compared with :mod:`concurrent.futures`
