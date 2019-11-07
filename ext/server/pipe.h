/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include <tango.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace PyTango { namespace Pipe {

class _Pipe
{
public:
    _Pipe() {}
    virtual ~_Pipe() {}

    void read(Tango::DeviceImpl* dev, Tango::Pipe&);
    void write(Tango::DeviceImpl* dev, Tango::WPipe&);
    bool is_allowed(Tango::DeviceImpl* dev, Tango::PipeReqType);

    void set_allowed_name(const std::string& name) { py_allowed_name = name; }
    void set_read_name(const std::string& name) { read_name = name; }
    void set_write_name(const std::string& name) { write_name = name; }

private:
    std::string py_allowed_name;
    std::string read_name;
    std::string write_name;

};


class PyPipe: public Tango::Pipe, public _Pipe
{
public:
    PyPipe(const std::string& name, const Tango::DispLevel level,
           const Tango::PipeWriteType write=Tango::PIPE_READ):
           Tango::Pipe(name, level, write) {}
    ~PyPipe() {}

    virtual void read(Tango::DeviceImpl* dev) { _Pipe::read(dev, *this); }
    virtual bool is_allowed(Tango::DeviceImpl* dev, Tango::PipeReqType rt)
        { return _Pipe::is_allowed(dev, rt); }
};

class PyWPipe: public Tango::WPipe, public _Pipe
{
public:
    PyWPipe(const std::string& name, const Tango::DispLevel level):
        Tango::WPipe(name, level) {}
    ~PyWPipe() {}

    virtual void read(Tango::DeviceImpl* dev)
        { _Pipe::read(dev, *this); }

    virtual void write(Tango::DeviceImpl* dev)
        { _Pipe::write(dev, *this); }

    virtual bool is_allowed(Tango::DeviceImpl* dev, Tango::PipeReqType rt)
        { return _Pipe::is_allowed(dev, rt); }
};

void set_value(Tango::Pipe&, py::object&);
py::object get_pipe_write_value(Tango::WPipe&);
void set_value(Tango::DevicePipeBlob& dpb, py::object& py_data);

}} // namespace PyTango::Pipe

//namespace PyDevicePipe
//{
//    void set_value(Tango::DevicePipeBlob&, py::object&);
//
//    void set_value(Tango::DevicePipe& pipe, py::object& py_value);
////        __set_value<Tango::DevicePipe>(pipe, py_value);
//
//} // namespace PyDevicePipe
