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
        self.__rois = {}

    def func1(self):
        return "executed func1"

    def func2(self, v):
        return 2*v

    def func3(self, a, b, c=1, d=3):
        """Just some documentation"""
        return "done func3"

    def add_roi(self, name, roi):
        self.__rois[name] = roi
        server.register_object(roi, name)

    def remove_roi(self, name):
        del self.__rois[name] # no need to unregister object


import logging
logging.basicConfig(level=logging.DEBUG)

my_object = MyClass()
a_roi = ROI(0,0,0,0)

server = Server("Dynamic")

server.register_object(my_object, "dynamic_object")
server.register_object(a_roi, "dummy_roi")
server.run()
