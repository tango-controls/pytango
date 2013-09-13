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

struct DeviceAttributeConfig
{
%TypeHeaderCode
#include <tango.h>
using namespace Tango;
%End
    std::string    name;
    AttrWriteType  writable;
    AttrDataFormat data_format;
    int            data_type;
    int            max_dim_x;
    int            max_dim_y;
    std::string    description;
    std::string    label;
    std::string    unit;
    std::string    standard_unit;
    std::string    display_unit;
    std::string    format;
    std::string    min_value;
    std::string    max_value;
    std::string    min_alarm;
    std::string    max_alarm;
    std::string    writable_attr_name;
    std::vector<std::string>  extensions;

    bool operator==(const DeviceAttributeConfig &);
};
