# ------------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

################################################################################
#                    TO BE USED BY DEVELOPERS ONLY
################################################################################

# Makefile to generate the PyTango library
# Needs the following environment variables to be defined:
# - TANGO_ROOT
# - LOG4TANGO_ROOT
# - OMNI_ROOT
# - BOOST_ROOT
# - ZMQ_ROOT
# - NUMPY_ROOT
#
# if target == install also needs: prefix=<install_dir>
# ex: make install prefix=/home/homer/.local/lib/python2.6/site-packages
#
# Optional:
# - OBJS_DIR: directory where files will be build (default: objs)
# - PY3K: if defined use python 3 boost python
# - PY_VER: use a specific python version (default is empty) (ex: 3.2)

ifdef PY_VER
PY_EXC=python$(PY_VER)
PY_MAJOR=$(shell $(PY_EXC) -c "import sys; sys.stdout.write(str(sys.version_info[0]))")
PY_MINOR=$(shell $(PY_EXC) -c "import sys; sys.stdout.write(str(sys.version_info[1]))")
else
PY_EXC=python
PY_MAJOR=$(shell $(PY_EXC) -c "import sys; sys.stdout.write(str(sys.version_info[0]))")
PY_MINOR=$(shell $(PY_EXC) -c "import sys; sys.stdout.write(str(sys.version_info[1]))")
PY_VER=$(PY_MAJOR).$(PY_MINOR)
endif

PY_VER_S=$(PY_MAJOR)$(PY_MINOR)

ifndef NUMPY_ROOT
NUMPY_INC = -I$(shell $(PY_EXC) -c "import sys, numpy; sys.stdout.write(numpy.get_include())")
else
NUMPY_INC = -I$(NUMPY_ROOT)/include
endif

PYTANGO_NUMPY_VERSION = $(shell $(PY_EXC) -c "import sys, numpy; sys.stdout.write(numpy.__version__)")

ifndef prefix
ifdef user
_PY_DIR=$(shell $(PY_EXC) -c "import sys, os; sys.stdout.write(os.path.split(os.path.join(os.path.dirname(os.__file__)))[1])")
prefix=$(HOME)/.local/lib/$(_PY_DIR)/site-packages
else
_PY_DIR=$(shell $(PY_EXC) -c "import sys, os; sys.stdout.write(os.path.join(os.path.dirname(os.__file__)))")
prefix=$(_PY_DIR)/site-packages
endif
endif

SRC_DIR = src/boost/cpp

ifndef OBJS_DIR
OBJS_DIR := objs_py$(PY_VER_S)
endif

CC = gcc

PY_INC := $(shell python$(PY_VER)-config --includes)

ifdef optimize
OPTIMIZE_CC = -g -O2
OPTIMIZE_LN = -O2
else
OPTIMIZE_CC = -g -O0
OPTIMIZE_LN = -O0
endif

TANGO_CFLAGS=`pkg-config --cflags-only-other tango`
TANGO_LIBS=`pkg-config --libs-only-l tango`
BOOST_LIB = boost_python-py$(PY_VER_S)

PRE_C_H := precompiled_header.hpp
PRE_C_H_O := $(OBJS_DIR)/$(PRE_C_H).gch
PRE_C := -include$(OBJS_DIR)/$(PRE_C_H)

LN := g++ -pthread -shared -Wl,$(OPTIMIZE_LN) -Wl,-Bsymbolic-functions -z defs
LN_STATIC := g++ -pthread -static -Wl,$(OPTIMIZE_LN) -Wl,-Bsymbolic-functions

LN_VER := -Wl,-h -Wl,--strip-all

LN_LIBS := -l$(BOOST_LIB) -lpython$(PY_VER)

INCLUDE_DIRS =

ifdef TANGO_ROOT
LN_DIRS += -L$(TANGO_ROOT)/lib
INCLUDE_DIRS += -I$(TANGO_ROOT)/include -I$(TANGO_ROOT)/include/tango
LN_LIBS += -ltango -lomniDynamic4 -lCOS4 -llog4tango -lzmq -lomniORB4 -lomnithread
else
LN_DIRS += `pkg-config --libs-only-L tango`
INCLUDE_DIRS += `pkg-config --cflags-only-I tango`
LN_LIBS += `pkg-config --libs-only-l tango`
endif

ifdef LOG4TANGO_ROOT
LN_DIRS += -L$(LOG4TANGO_ROOT)/lib
INCLUDE_DIRS += -I$(LOG4TANGO_ROOT)/include
endif

ifdef OMNI_ROOT
LN_DIRS += -L$(OMNI_ROOT)/lib
INCLUDE_DIRS += -I$(OMNI_ROOT)/include
endif

ifdef BOOST_ROOT
LN_DIRS += -L$(BOOST_ROOT)/lib
endif

ifdef ZMQ_ROOT
LN_DIRS += -L$(ZMQ_ROOT)/lib
INCLUDE_DIRS += -I$(ZMQ_ROOT)/include
endif

INCLUDE_DIRS += \
    -I$(SRC_DIR) \
    -I$(SRC_DIR)/server \
    $(PY_INC) \
    $(NUMPY_INC)

MACROS := -DNDEBUG -DPYTANGO_NUMPY_VERSION=$(PYTANGO_NUMPY_VERSION) -DPYTANGO_HAS_UNIQUE_PTR
CFLAGS := -pthread -fno-strict-aliasing -fwrapv -Wall -fPIC $(OPTIMIZE_CC) $(MACROS) $(TANGO_CFLAGS) $(INCLUDE_DIRS)
LNFLAGS := $(LN_DIRS) $(LN_LIBS)

LIB_NAME := _PyTango.so
LIB_NAME_STATIC := _PyTangoStatic.so
LIB_SYMB_NAME := $(LIB_NAME).dbg

OBJS := \
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

INC := callback.h \
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

ifdef optimize
LINKER=$(LN) $(LNFLAGS) $(OBJS) $(LN_VER) -o $(OBJS_DIR)/$(LIB_NAME).full ; strip --strip-all -o $(OBJS_DIR)/$(LIB_NAME) $(OBJS_DIR)/$(LIB_NAME).full
else
LINKER=$(LN) $(LNFLAGS) $(OBJS) $(LN_VER) -o $(OBJS_DIR)/$(LIB_NAME)
endif

#-----------------------------------------------------------------

all: build

build: init $(PRE_C_H_O) $(LIB_NAME)

init:
	@echo Using python $(PY_VER)
	@echo CFLAGS  = $(CFLAGS)
	@echo LNFLAGS = $(LNFLAGS)
	@echo Preparing build directories... 
	@mkdir -p $(OBJS_DIR)

$(PRE_C_H_O): $(SRC_DIR)/$(PRE_C_H)
	@echo Compiling pre-compiled header...
	@$(CC) $(CFLAGS) -c $< -o $(PRE_C_H_O)

#
# Rule for API files
#
$(OBJS_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo Compiling $(<F) ...
	@$(CC) $(CFLAGS) -c $< -o $(OBJS_DIR)/$*.o $(PRE_C)

$(OBJS_DIR)/%.o: $(SRC_DIR)/server/%.cpp
	@echo Compiling $(<F) ...
	@$(CC) $(CFLAGS) -c $< -o $(OBJS_DIR)/$*.o $(PRE_C)

#
#	The shared libs
#

$(LIB_NAME): $(PRE_C_H_0) $(OBJS)
	@echo Linking shared $(LIB_NAME) ...
	@$(LINKER)

clean:
	@echo Cleaning ...
	@rm -f *.o core
	@rm -f $(SRC_DIR)/*.gch
	@rm -rf $(OBJS_DIR)

install-py:
	@echo Installing python files into $(prefix)/PyTango ...
	@mkdir -p $(prefix)/PyTango
	@rsync -r src/boost/python/ $(prefix)/PyTango/

install: build install-py
	@echo Installing binary files into $(prefix)/PyTango ...
	@rsync $(OBJS_DIR)/$(LIB_NAME) $(prefix)/PyTango
 
