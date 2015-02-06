import PyTango.client

my_object = PyTango.client.Object("my_object")

print("my_object.bla = {0}".format(my_object.bla))
print("my_object.ble = {0}".format(my_object.ble))
print("my_object.bli = {0}".format(my_object.bli))
print("my_object.array = {0}".format(my_object.array))

r1 = my_object.func1()
print("my_object.func1() = {0}".format(r1))

r2 = my_object.func2(96.44)
print("my_object.func2(96.44) = {0}".format(r2))

r3 = my_object.func3(45.86, 'hello', d=False, c='world')
print("my_object.func3(45.86, 'hello', d=False, c='world') = {0}".format(r3))

r4 = my_object.func4()
print("my_object.func4() = {0}".format(r4))

r5 = my_object.zeros((500, 1000))
print("my_object.zeros((500, 1000)) = {0}".format(r5))
