asyncio mode
~~~~~~~~~~~~

Asyncio_ mode is similar to gevent but it uses explicit coroutines. You can compare gevent and asyncio examples.

.. literalinclude:: ../../examples/asyncio_green_mode/asyncio_simple_example.py
    :linenos:

Below you can find a TCP server example, which runs in an asynchronous mode and waits for a device's attribute name from a TCP client, then asks the device for a value and replies to the TCP client.

.. literalinclude:: ../../examples/asyncio_green_mode/tcp_server_example.py
    :linenos: