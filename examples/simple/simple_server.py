import numpy
from PyTango.server import Server
from common.roi import ROI

class MyClass:

    def __init__(self):
        self.bla = 55.6
        self.ble = 11
        self.bli = False
        self.array = numpy.ones((1000,1000))
        self.buff = bytearray(100000*"Hello ")

    def func1(self):
        return "executed func1"

    def func2(self, v):
        return 2*v

    def func3(self, a, b, c=1, d=3):
        """Just some documentation"""
        return "done func3"

    def func4(self):
        roi1 = ROI(10, 20, 640, 480)
        roi2 = ROI(0, 0, 1024, 768)
        return roi1.__dict__

    def zeros(self, shape, dtype='float'):
        import numpy
        return numpy.zeros(shape, dtype=dtype)

    def func5(self, nap_time):
        import time
        time.sleep(nap_time)
        return "Finished sleep for {0}s".format(nap_time)


my_object = MyClass()

server = Server("Bla")

server.register_object(my_object, "my_object")

server.run()
