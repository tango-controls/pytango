# tests for tango_util.cpp
#
from tango._tango import *
u=Util([u'Clock', u'clock'])
assert u.get_trace_level() == 0
u.set_trace_level(4)
assert u.get_trace_level() == 4
u.get_ds_inst_name() == 'clock'
u.get_ds_exec_name() == 'clock'
u.get_ds_name() == 'clock/clock'
u.get_host_name() == 'tcfidell11.dl.ac.uk'
print(u.get_pid_str())
print(u.get_pid())
print(u.get_version_str())
print(u.get_server_version())
u.set_server_version("4")
print(u.get_server_version())
print(u._FileDb())
assert u.get_serial_model() == SerialModel.BY_DEVICE
u.set_serial_model(SerialModel.BY_PROCESS)
assert u.get_serial_model() == SerialModel.BY_PROCESS
print(u.get_database())
u.server_init()
dserv = u.get_dserver_device()
print(dserv)
u.get_dserver_ior(dserv)
#u.get_device_ior(dev)
print(u.get_device_list_by_class('Clock'))
print("done")