TARGET = pytest.so
OFILES = sippytestcmodule.o sippyteststdvector1801.o
HFILES = sipAPIpytest.h 

CC = gcc
CXX = g++
LINK = g++
CPPFLAGS = -DNDEBUG -I. -I/usr/include/python2.6
CFLAGS =  -O2 -g -fPIC -Wall -W
CXXFLAGS =  -O2 -g -fPIC -Wall -W
LFLAGS = -z defs -shared -Wl,--version-script=pytest.exp
LIBS = -lpython2.6 -ltest
.SUFFIXES: .c .o .cpp .cc .cxx .C


.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.C.o:
	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) -o $@ $<

.c.o:
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<

all: $(TARGET)

$(OFILES): $(HFILES)

$(TARGET): $(OFILES)
	@echo '{ global: initpytest; local: *; };' > pytest.exp
	$(LINK) $(LFLAGS) -o $(TARGET) $(OFILES) $(LIBS)

install: $(TARGET)
	@test -d $(DESTDIR)/usr/lib/python2.6/dist-packages || mkdir -p $(DESTDIR)/usr/lib/python2.6/dist-packages
	cp -f $(TARGET) $(DESTDIR)/usr/lib/python2.6/dist-packages/$(TARGET)

clean:
	-rm -f $(TARGET)
	-rm -f sippytestcmodule.o
	-rm -f sippyteststdvector1801.o
	-rm -f pytest.exp
