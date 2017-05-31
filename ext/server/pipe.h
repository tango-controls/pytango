/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include <boost/python.hpp>
#include <tango.h>
#include "exception.h"
#include "pytgutils.h"
#include "server/device_impl.h"

namespace PyTango { namespace Pipe {

    class _Pipe
    {
    public:
        _Pipe() {}

        virtual ~_Pipe() {}

        void read(Tango::DeviceImpl *, Tango::Pipe &);
        void write(Tango::DeviceImpl *dev, Tango::WPipe &);
        bool is_allowed(Tango::DeviceImpl *, Tango::PipeReqType);

        void set_allowed_name(const std::string &name) { py_allowed_name = name; }
        void set_read_name(const std::string &name)    { read_name = name; }
        void set_write_name(const std::string &name)   { write_name = name; }
        bool _is_method(Tango::DeviceImpl *, const std::string &);
    
    private:
  	std::string py_allowed_name;
	std::string read_name;
	std::string write_name;
    };	


    class PyPipe: public Tango::Pipe,
                  public _Pipe
    {
    public:
        PyPipe(const std::string &name, const Tango::DispLevel level, 
               const Tango::PipeWriteType write=Tango::PIPE_READ):
	    Tango::Pipe(name, level, write)
	{}

        ~PyPipe() {}

        virtual void read(Tango::DeviceImpl *dev)
	{ _Pipe::read(dev, *this); }

	virtual bool is_allowed(Tango::DeviceImpl *dev, Tango::PipeReqType rt)
        { return _Pipe::is_allowed(dev, rt); }
  };

    class PyWPipe: public Tango::WPipe,
                   public _Pipe 
    {
    public:
        PyWPipe(const std::string &name, const Tango::DispLevel level):
	    Tango::WPipe(name, level)
	{}

        ~PyWPipe() {}

        virtual void read(Tango::DeviceImpl *dev)
	{ _Pipe::read(dev, *this); }

        virtual void write(Tango::DeviceImpl *dev)
	{ _Pipe::write(dev, *this); }

	virtual bool is_allowed(Tango::DeviceImpl *dev, Tango::PipeReqType rt)
        { return _Pipe::is_allowed(dev, rt); }
  };

}} // namespace PyTango::Pipe

namespace PyDevicePipe
{
    void set_value(Tango::DevicePipeBlob &, boost::python::object &);

} // namespace PyDevicePipe
