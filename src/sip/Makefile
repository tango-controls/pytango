TARGET = Tango.so
OFILES = sipTangocmodule.o sipTangoDatabase.o sipTangoDeviceProxy.o sipTangoConnection.o sipTangoDbDevImportInfo.o sipTangoDbDevInfo.o sipTangoDbDatum.o sipTangoCommandInfoList.o sipTangoCommandInfo.o sipTangoDevCommandInfo.o sipTangostdvector1800.o sipTangostdvector0100stdstring.o sipTangostdstring.o
HFILES = sipAPITango.h 

CC = gcc
CXX = g++
LINK = g++
CPPFLAGS = -DNDEBUG -I. -I/usr/include/python2.6
CFLAGS =  -O2 -g -fPIC -Wall -W
CXXFLAGS = -std=c++0x  -O2 -g -fPIC -Wall -W
LFLAGS = -z defs -shared -Wl,--version-script=Tango.exp
LIBS = -ltango -llog4tango -lzmq -lomniORB4 -lomniDynamic4 -lomnithread -lCOS4 -lpython2.6
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
	@echo '{ global: initTango; local: *; };' > Tango.exp
	$(LINK) $(LFLAGS) -o $(TARGET) $(OFILES) $(LIBS)

install: $(TARGET)
	@test -d $(DESTDIR)/usr/lib/python2.6/dist-packages || mkdir -p $(DESTDIR)/usr/lib/python2.6/dist-packages
	cp -f $(TARGET) $(DESTDIR)/usr/lib/python2.6/dist-packages/$(TARGET)

clean:
	-rm -f $(TARGET)
	-rm -f sipTangocmodule.o
	-rm -f sipTangoDatabase.o
	-rm -f sipTangoDeviceProxy.o
	-rm -f sipTangoConnection.o
	-rm -f sipTangoDbDevImportInfo.o
	-rm -f sipTangoDbDevInfo.o
	-rm -f sipTangoDbDatum.o
	-rm -f sipTangoCommandInfoList.o
	-rm -f sipTangoCommandInfo.o
	-rm -f sipTangoDevCommandInfo.o
	-rm -f sipTangostdvector1800.o
	-rm -f sipTangostdvector0100stdstring.o
	-rm -f sipTangostdstring.o
	-rm -f Tango.exp
