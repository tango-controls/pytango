################################################################################
##
## This file is part of PyTango, a python binding for Tango
## 
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
## 
## PyTango is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## PyTango is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################

################################################################################
#                    TO BE USED BY DEVELOPERS ONLY
################################################################################

# Makefile to generate the PyTango library
# Needs the following environment variables to be defined:
# - TANGO_ROOT
# - NUMY_ROOT
#
# if target == install also needs: prefix=<install_dir>
# ex: make install prefix=/home/homer/.local/lib/python2.6/site-packages
#

ifndef TANGO_ROOT
TANGO_ROOT=/usr
endif

ifndef NUMPY_ROOT
NUMPY_ROOT=$(shell python -c "import os; import numpy; print os.path.dirname(numpy.__file__)")/core
endif

ifndef prefix
ifdef user
_PY_DIR=$(shell python -c "import os; print os.path.split(os.path.join(os.path.dirname(os.__file__)))[1]")
prefix=$(HOME)/.local/lib/$(_PY_DIR)/site-packages
else
_PY_DIR=$(shell python -c "import os; print os.path.join(os.path.dirname(os.__file__))")
prefix=$(_PY_DIR)/site-packages
endif
endif

SRC_DIR = src

ifndef OBJS_DIR
OBJS_DIR = objs
endif

CC = gcc
CCFLAGS = -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -fPIC $(INCLUDE_DIRS)

LN = g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions
LN_STATIC = g++ -pthread -static -Wl,-O1 -Wl,-Bsymbolic-functions
LN_VER = -Wl,-h -Wl,--strip-all
LN_LIBS = -ltango -llog4tango -lpthread -lrt -ldl -lomniORB4 -lomniDynamic4 -lomnithread -lCOS4 -lboost_python
LN_DIRS = -L$(TANGO_ROOT)/lib

PY_INC = $(shell python-config --includes)
NUMPY_INC = -I$(NUMPY_ROOT)/include
TANGO_INC = -I$(TANGO_ROOT)/include

INCLUDE_DIRS = \
-Isrc \
$(TANGO_INC) \
$(TANGO_INC)/tango \
$(PY_INC) \
$(NUMPY_INC)

LIB_NAME = _PyTango.so
LIB_NAME_STATIC = _PyTangoStatic.so
LIB_SYMB_NAME = $(LIB_NAME).dbg

OBJS = \
$(OBJS_DIR)/api_util.o \
$(OBJS_DIR)/archive_event_info.o \
$(OBJS_DIR)/attr_conf_event_data.o \
$(OBJS_DIR)/attribute_alarm_info.o \
$(OBJS_DIR)/attribute_dimension.o \
$(OBJS_DIR)/attribute_event_info.o \
$(OBJS_DIR)/attribute_info.o \
$(OBJS_DIR)/attribute_info_ex.o \
$(OBJS_DIR)/attribute_proxy.o \
$(OBJS_DIR)/base_types.o \
$(OBJS_DIR)/callback.o \
$(OBJS_DIR)/change_event_info.o \
$(OBJS_DIR)/command_info.o \
$(OBJS_DIR)/connection.o \
$(OBJS_DIR)/constants.o \
$(OBJS_DIR)/database.o \
$(OBJS_DIR)/data_ready_event_data.o \
$(OBJS_DIR)/db.o \
$(OBJS_DIR)/dev_command_info.o \
$(OBJS_DIR)/dev_error.o \
$(OBJS_DIR)/device_attribute_config.o \
$(OBJS_DIR)/device_attribute.o \
$(OBJS_DIR)/device_attribute_history.o \
$(OBJS_DIR)/device_data.o \
$(OBJS_DIR)/device_data_history.o \
$(OBJS_DIR)/device_info.o \
$(OBJS_DIR)/device_proxy.o \
$(OBJS_DIR)/enums.o \
$(OBJS_DIR)/event_data.o \
$(OBJS_DIR)/exception.o \
$(OBJS_DIR)/from_py.o \
$(OBJS_DIR)/group.o \
$(OBJS_DIR)/group_element.o \
$(OBJS_DIR)/group_reply.o \
$(OBJS_DIR)/group_reply_list.o \
$(OBJS_DIR)/locker_info.o \
$(OBJS_DIR)/locking_thread.o \
$(OBJS_DIR)/periodic_event_info.o \
$(OBJS_DIR)/poll_device.o \
$(OBJS_DIR)/pytango.o \
$(OBJS_DIR)/pytgutils.o \
$(OBJS_DIR)/pyutils.o \
$(OBJS_DIR)/time_val.o \
$(OBJS_DIR)/to_py.o \
$(OBJS_DIR)/version.o \
$(OBJS_DIR)/attr.o \
$(OBJS_DIR)/attribute.o \
$(OBJS_DIR)/command.o \
$(OBJS_DIR)/device_class.o \
$(OBJS_DIR)/device_impl.o \
$(OBJS_DIR)/dserver.o \
$(OBJS_DIR)/encoded_attribute.o \
$(OBJS_DIR)/log4tango.o \
$(OBJS_DIR)/multi_attribute.o \
$(OBJS_DIR)/multi_class_attribute.o \
$(OBJS_DIR)/subdev.o \
$(OBJS_DIR)/tango_util.o \
$(OBJS_DIR)/user_default_attr_prop.o \
$(OBJS_DIR)/wattribute.o

INC = callback.h \
defs.h \
device_attribute.h \
exception.h \
fast_from_py.h \
from_py.h \
pytgutils.h \
pyutils.h \
tango_numpy.h \
tgutils.h \
to_py.h \
attr.h \
attribute.h \
command.h \
device_class.h \
device_impl.h

#-----------------------------------------------------------------

all: build

build: init $(LIB_NAME)

init:
	mkdir -p $(OBJS_DIR)

#
# Rule for shared library
#

.SUFFIXES: .o .cpp
.cpp.o:
	$(CC) $(CCFLAGS) -c $< -o $*.o

#
# Rule for API files
#
$(OBJS_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CCFLAGS) -c $< -o $(OBJS_DIR)/$*.o

$(OBJS_DIR)/%.o: $(SRC_DIR)/server/%.cpp
	$(CC) $(CCFLAGS) -c $< -o $(OBJS_DIR)/$*.o

#
#	The shared libs
#

$(LIB_NAME): $(OBJS)
	$(LN) $(OBJS) $(LN_DIRS) $(LN_LIBS) -o $(OBJS_DIR)/$(LIB_NAME) $(LN_VER)
#	$(LN_STATIC) $(OBJS) $(LN_DIRS) $(LN_LIBS) -o $(OBJS_DIR)/$(LIB_NAME_STATIC) $(LN_VER)
#	objcopy --only-keep-debug $(OBJS_DIR)/$(LIB_NAME) $(OBJS_DIR)/$(LIB_SYMB_NAME)
#	objcopy --strip-debug --strip-unneeded $(OBJS_DIR)/$(LIB_NAME)
#	objcopy --add-gnu-debuglink=$(OBJS_DIR)/$(LIB_SYMB_NAME) $(OBJS_DIR)/$(LIB_NAME)
#	chmod -x $(OBJS_DIR)/$(LIB_SYMB_NAME)

clean:
	rm -f *.o core
	rm -f $(PREP).gch
	rm -rf $(OBJS_DIR)

install-py:
	mkdir -p $(prefix)
	rsync -r PyTango $(prefix)

install: build install-py
	rsync $(OBJS_DIR)/$(LIB_NAME) $(prefix)/PyTango
#	rsync $(OBJS_DIR)/$(LIB_SYMB_NAME) $(prefix)/PyTango
    
