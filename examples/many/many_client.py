from time import time

import tango.client

N = 100

obj_names = ["obj%04d" % i for i in range(N)]

start = time()
objs = map(tango.client.Object, obj_names)
dt = time() - start
print "Took %fs to create %d objects (avg %fs/object)" % (dt, N, dt/N)

start = time()
res = [ obj.func3(1,2) for obj in objs ]
dt = time() - start
print "Took %fs to call func3 on %d objects (avg %fs/object)" % (dt, N, dt/N)
