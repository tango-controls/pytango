
# -----------------------------------------------------------------------------
# Common stuff
# -----------------------------------------------------------------------------
COMMON_OUT = _common

CSCC = cscc
CC = c++
LD = c++

SRC = src
SWIG = swig -c++

INTERFACE_FILE=tango.i

# -----------------------------------------------------------------------------
# PERL stuff
# -----------------------------------------------------------------------------
PERL_FLAGS =$(shell perl -MConfig -e 'print join(" ", @Config{qw(ccflags optimize cccdlflags)}, "-I$$Config{archlib}/CORE")')
PERL_LINK_FLAGS = $(shell perl -MConfig -e 'print $$Config{lddlflags}')
PERL_OUT = _perl
PERL_SWIG = $(SWIG) -perl5 -outdir $(PERL_OUT)
PERL_CC = $(CC) $(PERL_FLAGS)
PERL_LD = $(CC) $(PERL_LINK_FLAGS)

# -----------------------------------------------------------------------------
# Python stuff
# -----------------------------------------------------------------------------
PY_FLAGS = -g -fPIC -Wall $(shell python-config --includes) -std=c++0x
PY_LINK_FLAGS = -ltango -llog4tango -lpthread -lrt -ldl -lomniORB4 -lomniDynamic4 -lomnithread -lCOS4 -lzmq
PY_OUT = _python
PY_SWIG = $(SWIG) -python -outdir $(PY_OUT)
PY_CC = $(CC) $(PY_FLAGS)
PY_LD = $(LD) $(PY_LINK_FLAGS) -shared

# -----------------------------------------------------------------------------
# C# stuff
# -----------------------------------------------------------------------------
CSHARP_FLAGS = -fPIC
CSHARP_OUT = _csharp
CSHARP_SWIG = $(SWIG) -csharp -outdir $(CSHARP_OUT)
CSHARP_CC = $(CC) $(CSHARP_FLAGS)
CSHARP_LD = $(CC) -shared

# -----------------------------------------------------------------------------
# TCL stuff
# -----------------------------------------------------------------------------
TCL_FLAGS = -I/usr/include/tcl -fPIC
TCL_OUT = _tcl
TCL_SWIG = $(SWIG) -tcl -outdir $(TCL_OUT)
TCL_CC = $(CC) $(TCL_FLAGS)
TCL_LD = $(LD) -shared

# -----------------------------------------------------------------------------
# GO stuff
# -----------------------------------------------------------------------------
GO_FLAGS = -fPIC
GO_OUT = _go
GO_SWIG = $(SWIG) -go -intgosize 32 -outdir $(GO_OUT)
GO_CC = $(CC) $(GO_FLAGS)
GO_LD = $(CC) -shared

# -----------------------------------------------------------------------------
# commands
# -----------------------------------------------------------------------------

all: perl python csharp tcl

common:
	mkdir -p $(COMMON_OUT)

perl: common
	mkdir -p $(PERL_OUT)
	$(PERL_SWIG) -o $(PERL_OUT)/tango_wrap.cpp $(INTERFACE_FILE)
	$(PERL_CC) -c $(PERL_OUT)/tango_wrap.cpp -o $(PERL_OUT)/tango_wrap.o
	$(PERL_LD) $(COMMON_OUT)/tango_wrap.o -o $(PERL_OUT)/_Tango.so
	
python: common
	mkdir -p $(PY_OUT)
	$(PY_SWIG) -o $(PY_OUT)/tango_wrap.cpp $(INTERFACE_FILE)
	$(PY_CC) -c $(PY_OUT)/tango_wrap.cpp -o $(PY_OUT)/tango_wrap.o
	$(PY_LD) $(PY_OUT)/tango_wrap.o -o $(PY_OUT)/_Tango.so

csharp: common
	mkdir -p $(CSHARP_OUT)
	$(CSHARP_SWIG) -o $(CSHARP_OUT)/example_wrap.c $(INTERFACE_FILE)
	$(CSHARP_CC) -c $(CSHARP_OUT)/example_wrap.c -o $(CSHARP_OUT)/example_wrap.o
	$(CSHARP_LD) $(COMMON_OUT)/example.o $(CSHARP_OUT)/example_wrap.o -o $(CSHARP_OUT)/libexample.so
	
tcl: common
	mkdir -p $(TCL_OUT)
	$(TCL_SWIG) -o $(TCL_OUT)/example_wrap.c $(INTERFACE_FILE)
	$(TCL_CC) -c $(TCL_OUT)/example_wrap.c -o $(TCL_OUT)/example_wrap.o
	$(TCL_LD) $(COMMON_OUT)/example.o $(TCL_OUT)/example_wrap.o -o $(TCL_OUT)/example.so

go: common
	mkdir -p $(GO_OUT)
	$(GO_SWIG) -o $(GO_OUT)/example_wrap.c $(INTERFACE_FILE)
	$(GO_CC) -c $(GO_OUT)/example_wrap.c -o $(GO_OUT)/example_wrap.o $(GO_FLAGS)
	$(GO_LD) $(COMMON_OUT)/example.o $(GO_OUT)/example_wrap.o -o $(GO_OUT)/example.so
	6g -o $(GO_OUT)/example.6 $(GO_OUT)/example.go
	6c -I $(GOROOT)/pkg/linux_amd64 -o $(GO_OUT)/example_gc.6 $(GO_OUT)/example_gc.c
	pack grc $(GO_OUT)/example.a $(GO_OUT)/example.6 $(GO_OUT)/example_gc.6

clean:
	rm -rf $(COMMON_OUT) $(PERL_OUT) $(PY_OUT) $(CSHARP_OUT) $(TCL_OUT) $(GO_OUT)
