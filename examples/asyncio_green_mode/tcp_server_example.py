"""A simple TCP server for Tango attributes.

It runs on all interfaces on port 8888:

   $ python tango_tcp_server.py
   Serving on 0.0.0.0 port 8888

It can be accessed using netcat:

   $ ncat localhost 8888
   >>> sys/tg_test/1/ampli
   0.0
   >>> sys/tg_test/1/state
   RUNNING
   >>> sys/tg_test/1/nope
   DevFailed[
   DevError[
        desc = Attribute nope is not supported by device sys/tg_test/1
      origin = AttributeProxy::real_constructor()
      reason = API_UnsupportedAttribute
    severity = ERR]
    ]
   >>> ...
"""

import asyncio
from tango.asyncio import AttributeProxy


async def handle_echo(reader, writer):
    # Write the cursor
    writer.write(b'>>> ')
    # Loop over client request
    async for line in reader:
        request = line.decode().strip()
        # Get attribute value using asyncio green mode
        try:
            proxy = await AttributeProxy(request)
            attr_value = await proxy.read()
            reply = str(attr_value.value)
        # Catch exception if something goes wrong
        except Exception as exc:
            reply = str(exc)
        # Reply to client
        writer.write(reply.encode() + b'\n' + b'>>> ')
    # Close communication
    writer.close()


async def start_serving():
    server = await asyncio.start_server(handle_echo, '0.0.0.0', 8888)
    print('Serving on {} port {}'.format(*server.sockets[0].getsockname()))
    return server


async def stop_serving(server):
    server.close()
    await server.wait_closed()


def main():
    # Start the server
    loop = asyncio.get_event_loop()
    server = loop.run_until_complete(start_serving())
    # Serve requests until Ctrl+C is pressed
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    # Close the server
    loop.run_until_complete(stop_serving(server))
    loop.close()


if __name__ == '__main__':
    main()
