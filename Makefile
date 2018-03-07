#!/bin/bash

# Use with the following conda environment:
# $ conda create -n tangobind -c esrf-bcu -c conda-forge python=2 tango pybind11 numpy gxx_linux-64
# In order to isolate the build from the system libraries

objs_py27 = \
	objs_py27/api_util.o \
	objs_py27/archive_event_info.o \
	objs_py27/attr_conf_event_data.o \
	objs_py27/attribute_alarm_info.o \
	objs_py27/attribute_dimension.o \
	objs_py27/attribute_event_info.o \
	objs_py27/attribute_info_ex.o \
	objs_py27/attribute_info.o \
	objs_py27/attribute_proxy.o \
	objs_py27/base_types.o \
	objs_py27/callback.o \
	objs_py27/change_event_info.o \
	objs_py27/command_info.o \
	objs_py27/connection.o \
	objs_py27/constants.o \
	objs_py27/data_ready_event_data.o \
	objs_py27/database.o \
	objs_py27/db.o \
	objs_py27/dev_command_info.o \
	objs_py27/dev_error.o \
	objs_py27/device_attribute_config.o \
	objs_py27/device_attribute_history.o \
	objs_py27/device_attribute.o \
	objs_py27/device_data_history.o \
	objs_py27/device_data.o \
	objs_py27/device_info.o \
	objs_py27/device_pipe.o \
	objs_py27/device_proxy.o \
	#objs_py27/devintr_change_event_data.o \
	objs_py27/enums.o \
	objs_py27/event_data.o \
	objs_py27/group_reply_list.o \
	objs_py27/group_reply.o \
	objs_py27/group.o \
	objs_py27/locker_info.o \
	objs_py27/locking_thread.o \
	objs_py27/periodic_event_info.o \
	objs_py27/pipe_event_data.o \
	objs_py27/pipe_info.o \
	objs_py27/poll_device.o \
	objs_py27/pytango.o \
	objs_py27/pyutils.o \
	objs_py27/time_val.o \
	objs_py27/version.o \
	objs_py27/attr.o \
	objs_py27/attribute.o \
	objs_py27/auto_monitor.o \
	objs_py27/command.o \
	objs_py27/device_impl.o \
	objs_py27/dserver.o \
	objs_py27/encoded_attribute.o \
	objs_py27/fwdAttr.o \
	objs_py27/log4tango.o \
	objs_py27/multi_attribute.o \
	objs_py27/multi_class_attribute.o \
	objs_py27/pipe.o \
	objs_py27/subdev.o \
	objs_py27/tango_util.o \
	objs_py27/user_default_attr_prop.o \
	objs_py27/user_default_pipe_prop.o \
	objs_py27/wattribute.o \
#	objs_py27/device_class.o \
#	objs_py27/exception.o \
#	objs_py27/from_py.o \
#	objs_py27/to_py.o \

CFLAGS += -g -std=c++11 `python-config --cflags` -fPIC -Iext/
CFLAGS += -I$(CONDA_PREFIX)/include/tango
CFLAGS += -I$(CONDA_PREFIX)/lib/python2.7/site-packages/numpy/core/include
CFLAGS += -I$(CONDA_PREFIX)/include

srcs = $(objs_py27:.o=.cpp)


all: pytest

pytest: $(objs_py27)
	$(GXX) -shared  -L$(CONDA_PREFIX)/lib -ltango  ${objs_py27} -o tango/_tango.so

objs_py27/%.o : ext/%.cpp
	@echo -n "Compiling $(<F)... "
	$(GXX) -shared $(CFLAGS) -c $< -o objs_py27/$*.o
	@echo Done!

objs_py27/%.o : ext/server/%.cpp
	@echo -n "Compiling $(<F)... "
	$(GXX) -shared $(CFLAGS) -c $< -o objs_py27/$*.o
	@echo Done!
