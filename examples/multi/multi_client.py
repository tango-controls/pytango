import PyTango.client

my_object = PyTango.client.Object("multi_my_object")

print("my_object.bla = {0}".format(my_object.bla))
print("my_object.ble = {0}".format(my_object.ble))
print("my_object.bli = {0}".format(my_object.bli))

r1 = my_object.func1()
print("my_object.func1() = {0}".format(r1))

r2 = my_object.func2(96.44)
print("my_object.func2(96.44) = {0}".format(r2))

r3 = my_object.func3(45.86, 'hello', d=False, c='world')
print("my_object.func3(45.86, 'hello', d=False, c='world') = {0}".format(r3))

another_object = PyTango.client.Object("multi_another_object")

r1 = another_object.is_valid()
print("another_object.is_valid() = {0}".format(r1))

r2 = another_object.lets_go("hello, world!")
print("another_object.lets_go('hello, world!') = {0}".format(r2))
