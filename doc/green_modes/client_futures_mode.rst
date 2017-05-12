futures mode
~~~~~~~~~~~~

Using :mod:`concurrent.futures` cooperative mode in PyTango is relatively easy::

    >>> from tango.futures import DeviceProxy

    >>> dev = DeviceProxy("sys/tg_test/1")
    >>> dev.get_green_mode()
    tango.GreenMode.Futures

    >>> print(dev.state())
    RUNNING

The :func:`tango.futures.DeviceProxy` API is exactly the same as the standard
:class:`~tango.DeviceProxy`. The difference is in the semantics of the methods
that involve synchronous network calls (constructor included) which may block
the execution for a relatively big amount of time.
The list of methods that have been modified to accept *futures* semantics are,
on the :func:`tango.futures.DeviceProxy`:

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
Well, this is, in fact, one of the goals: be able to use a *futures* cooperation
without changing the API. Behind the scenes the methods mentioned before have
been modified to be able to work cooperatively.

All of the above methods have been boosted with two extra keyword arguments
*wait* and *timeout* which allow to fine tune the behaviour.
The *wait* parameter is by default set to `True` meaning wait for the request
to finish (the default semantics when not using green mode).
If *wait* is set to `True`, the timeout determines the maximum time to wait for
the method to execute. The default is `None` which means wait forever. If *wait*
is set to `False`, the *timeout* is ignored.

If *wait* is set to `True`, the result is the same as executing the
*standard* method on a :class:`~tango.DeviceProxy`.
If, *wait* is set to `False`, the result will be a
:class:`concurrent.futures.Future`. In this case, to get the actual value
you will need to do something like::

    >>> from tango.futures import DeviceProxy

    >>> dev = DeviceProxy("sys/tg_test/1")
    >>> result = dev.state(wait=False)
    >>> result
    <Future at 0x16cb310 state=pending>

    >>> # this will be the blocking code
    >>> state = result.result()
    >>> print(state)
    RUNNING

Here is another example using :meth:`~DeviceProxy.read_attribute`::

    >>> from tango.futures import DeviceProxy

    >>> dev = DeviceProxy("sys/tg_test/1")
    >>> result = dev.read_attribute('wave', wait=False)
    >>> result
    <Future at 0x16cbe50 state=pending>

    >>> dev_attr = result.result()
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
           time = TimeVal(tv_nsec = 0, tv_sec = 1383923329, tv_usec = 451821)
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
