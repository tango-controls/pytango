/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

class DeviceAttribute
{

public :
//
// constructor methods
//
	enum except_flags
	{
		isempty_flag = 0,
		wrongtype_flag,
		failed_flag,
		unknown_format_flag,
		numFlags
	};

	DeviceAttribute();
	DeviceAttribute(const DeviceAttribute&);
};