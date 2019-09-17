import tango
import gc
while True:
    gc.collect()
    print(len(gc.get_objects()))
    prox=tango.DeviceProxy('sys/tg_test/1')
    print(repr(prox))
    prox.write_attribute = prox.write_attribute
    print(prox.get_fqdn())
