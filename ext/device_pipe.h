/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include <pybind11/pybind11.h>
#include <tango.h>
#include "defs.h"
#include "pyutils.h"

namespace py = pybind11;

namespace PyTango
{ 
    namespace DevicePipe 
    {
        py::object extract(Tango::DevicePipe& device_pipe);

        py::object extract(Tango::DevicePipeBlob& blob);

        void update_values(Tango::DevicePipe& self, py::object& py_value);

        /// @param self The DevicePipe instance that the new python object
        /// will represent. It must be allocated with new. The new python object
        /// will handle it's destruction. There's never any reason to delete it
        /// manually after a call to this: Even if this function fails, the
        /// responsibility of destroying it will already be in py_value side or
        /// the object will already be destroyed.
        template<typename TDevicePipe>
        py::object convert_to_python(TDevicePipe* self)
        {
            py::object py_value;
            try
            {
                py_value = py::cast(self, py::return_value_policy::move);

//                py_value = py::object(
//                    py::handle(py::to_python_indirect<TDevicePipe*, py::detail::make_owning_holder>() (self)));
            }
            catch (...)
            {
                delete self;
                throw;
            }

            update_values(*self, py_value);
            return py_value;
        }
    } 
}
