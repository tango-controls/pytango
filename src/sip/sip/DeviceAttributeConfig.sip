/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

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
