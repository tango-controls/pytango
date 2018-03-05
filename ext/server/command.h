/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#ifndef _COMMAND_H_
#define _COMMAND_H_

#include <tango.h>

namespace py = pybind11;

class PyCmd : public Tango::Command
{
public:
    PyCmd(string &name, Tango::CmdArgType in, Tango::CmdArgType out,
          string &in_desc, string &out_desc, Tango::DispLevel  level)
    :Tango::Command(name, in, out, in_desc, out_desc, level), py_allowed_defined(false)
    {};

    PyCmd(const char *name, Tango::CmdArgType in, Tango::CmdArgType out)
    :Tango::Command(name, in, out), py_allowed_defined(false)
    {};

    PyCmd(const char *name, Tango::CmdArgType in, Tango::CmdArgType out,
          const char *in_desc, const char *out_desc, Tango::DispLevel  level)
    :Tango::Command(name, in, out, in_desc, out_desc, level), py_allowed_defined(false)
    {};

    virtual ~PyCmd() {};

    virtual CORBA::Any *execute (Tango::DeviceImpl *dev, const CORBA::Any &any);
    virtual bool is_allowed (Tango::DeviceImpl *dev, const CORBA::Any &any);

    void set_allowed(const string &name) {py_allowed_defined=true; py_allowed_name=name;}

private:
    bool py_allowed_defined;
    string	py_allowed_name;
};

#endif
