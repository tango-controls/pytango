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

#include "defs.h"
#include "pyutils.h"

namespace PyTango
{ 
    namespace DevicePipe 
    {
        void
        update_values(Tango::DevicePipe& self, bopy::object& py_value,
                      PyTango::ExtractAs extract_as=PyTango::ExtractAsNumpy);

        template<typename TDevicePipe>
        bopy::object
        convert_to_python(TDevicePipe* self, PyTango::ExtractAs extract_as)
        {
            bopy::object py_value;
            try 
            {
                py_value = bopy::object(
                               bopy::handle<>(
                                   bopy::to_python_indirect<
                                       TDevicePipe*, 
                                       bopy::detail::make_owning_holder>() (self)));
            } 
            catch (...)
            {
                delete self;
                throw;
            }

            update_values(*self, py_value, extract_as);
            return py_value;            
        }
    } 
}
