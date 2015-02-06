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

    def func(self, nap_time):
        import time
        time.sleep(nap_time)
        return "Finished sleep for {0}s".format(nap_time)

server = Server("many", server_type="Server")

N = 100

objs = []
for i in range(N):
    name = "obj%04d" % i
    obj = MyClass()
    objs.append(obj)
    server.register_object(obj, name)

server.run()
