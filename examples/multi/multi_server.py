from PyTango.server import Server

class MyClass:

    def __init__(self):
        self.bla = 55.6
        self.ble = 11
        self.bli = False

    def func1(self):
        return "executed func1"

    def func2(self, v):
        return 2*v
    
    def func3(self, a, b, c=1, d=3):
        """Just some documentation"""
        return "done func3"


class AnotherClass:

    def __init__(self, valid=True):
        self.__valid = valid

    def is_valid(self):
        return self.__valid
    
    def lets_go(self, p1):
        return "lets_go done!", p1

    def fft(self, a, n=None, axis=-1):
        import numpy.fft
        return numpy.fft.fft(a, n=n, axis=axis)

    def array(self, a):
        return a

my_object = MyClass()
another_object = AnotherClass(valid=False)

server = Server("multi", server_type="Server")

server.register_object(my_object, "multi_my_object")
server.register_object(another_object, "multi_another_object")

server.run()
