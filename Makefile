#!/bin/bash

# Use with the following conda environment:
# $ conda create -n tangobind -c esrf-bcu -c conda-forge python=[2/3] tango pybind11 numpy gxx_linux-64
# In order to isolate the build from the system libraries

objs = \
	objs/api_util.o \
	objs/archive_event_info.o \
	objs/attr_conf_event_data.o \
	objs/attribute_alarm_info.o \
	objs/attribute_dimension.o \
	objs/attribute_event_info.o \
	objs/attribute_info_ex.o \
	objs/attribute_info.o \
	objs/attribute_proxy.o \
	objs/base_types.o \
	objs/callback.o \
	objs/change_event_info.o \
	objs/command_info.o \
	objs/constants.o \
	objs/data_ready_event_data.o \
	objs/database.o \
	objs/db.o \
	objs/dev_command_info.o \
	objs/dev_error.o \
	objs/device_attribute_config.o \
	objs/device_attribute_history.o \
	objs/device_attribute.o \
	objs/device_data_history.o \
	objs/device_data.o \
	objs/device_info.o \
	objs/device_pipe.o \
	objs/device_proxy.o \
	objs/enums.o \
	objs/event_data.o \
	objs/exception.o \
	objs/group_reply_list.o \
	objs/group_reply.o \
	objs/group.o \
	objs/locker_info.o \
	objs/locking_thread.o \
	objs/periodic_event_info.o \
	objs/pipe_event_data.o \
	objs/pipe_info.o \
	objs/poll_device.o \
	objs/pytango.o \
	objs/pyutils.o \
	objs/time_val.o \
	objs/version.o \
	objs/attr.o \
	objs/attribute.o \
	objs/auto_monitor.o \
	objs/command.o \
	objs/dserver.o \
	objs/device_impl.o \
	objs/encoded_attribute.o \
	objs/fwdAttr.o \
	objs/log4tango.o \
	objs/multi_attribute.o \
	objs/multi_class_attribute.o \
	objs/subdev.o \
	objs/tango_util.o \
	objs/user_default_attr_prop.o \
	objs/user_default_pipe_prop.o \
	objs/wattribute.o \
	objs/devintr_change_event_data.o \
	objs/device_class.o \
	objs/pipe.o \
	objs/to_py.o \
	objs/from_py.o \

PYTHON_INCLUDE_DIR=$(shell python -c "import sysconfig; print(sysconfig.get_path('include'))")
CFLAGS += -g -std=c++14 -fvisibility=hidden
CFLAGS += -I$(PYTHON_INCLUDE_DIR)
CFLAGS += -I$(CONDA_PREFIX)/include/tango -Wno-deprecated
CFLAGS += -I$(CONDA_PREFIX)/include
CFLAGS += -Iext

srcs = $(objs:.o=.cpp)

all: pytest

pytest: $(objs)
	$(GXX) -shared  -L$(CONDA_PREFIX)/lib -ltango ${objs} -o tango/_tango.so

objs/%.o : ext/%.cpp
	@echo -n "Compiling $(<F)... "
	$(GXX) -shared $(CFLAGS) -c $< -o objs/$*.o
	@echo Done!

objs/%.o : ext/server/%.cpp
	@echo -n "Compiling $(<F)... "
	$(GXX) -shared $(CFLAGS) -c $< -o objs/$*.o
	@echo Done!
