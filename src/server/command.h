/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
   
   PyTango is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   PyTango is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

#ifndef _COMMAND_H_
#define _COMMAND_H_

#include <boost/python.hpp>
#include <tango.h>

class PyCmd : public Tango::Command
{
public:
    PyCmd(string &name, Tango::CmdArgType in, Tango::CmdArgType out,
          string &in_desc, string &out_desc, Tango::DispLevel  level)
    :Tango::Command(name,in,out,in_desc,out_desc, level),py_allowed_defined(false)	{};

    PyCmd(const char *name, Tango::CmdArgType in, Tango::CmdArgType out)
    :Tango::Command(name,in,out),py_allowed_defined(false)	{};

    PyCmd(const char *name, Tango::CmdArgType in, Tango::CmdArgType out,
          const char *in_desc, const char *out_desc, Tango::DispLevel  level)
    :Tango::Command(name,in,out,in_desc,out_desc, level),py_allowed_defined(false)	{};

    virtual ~PyCmd() {};

    virtual CORBA::Any *execute (Tango::DeviceImpl *dev, const CORBA::Any &any);
    virtual bool is_allowed (Tango::DeviceImpl *dev, const CORBA::Any &any);

    void set_allowed(const string &name) {py_allowed_defined=true;py_allowed_name=name;}

private:
    bool py_allowed_defined;
    string	py_allowed_name;
};

#endif
