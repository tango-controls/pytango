import asyncio
from tango.asyncio import DeviceProxy

async def asyncio_example():
    dev = await DeviceProxy("sys/tg_test/1")
    print(dev.get_green_mode())

    print(await dev.state())

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio_example())
loop.close()
