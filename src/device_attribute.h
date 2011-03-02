/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   (copyleft) CELLS / ALBA Synchrotron, Bellaterra, Spain
  
   This is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.
  
   This software is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with this program; if not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

#pragma once

#include <boost/python.hpp>
#include <tango/tango.h>
#include <iostream>
#include <string>

#include "pyutils.h"
#include "defs.h"

namespace PyDeviceAttribute {

/// @name Types
/// @{
    typedef std::auto_ptr<std::vector<Tango::DeviceAttribute> > AutoDevAttrVector;
/// @}
    
    template<long tangoTypeConst>
    void _update_array_values_as_tuples(Tango::DeviceAttribute &self, bool isImage, boost::python::object py_value);

    /// Set the value of a DeviceAttribute from python (useful for write*)
    void reset(Tango::DeviceAttribute& self, const Tango::AttributeInfo &attr_info, boost::python::object py_value);
    void reset(Tango::DeviceAttribute & self, const std::string &attr_name, Tango::DeviceProxy &dev_proxy, boost::python::object py_value);

    void update_values(Tango::DeviceAttribute &self, boost::python::object& py_value, PyTango::ExtractAs extract_as=PyTango::ExtractAsNumpy);

    template<typename TDeviceAttribute>
    void update_data_format(Tango::DeviceProxy & dev_proxy, TDeviceAttribute* first, size_t nelems)
    {
        // Older devices do not send arg_format. So we try to discover it from
        // dim_x and dim_y. It is not perfect, sometimes we will get SCALAR for
        // SPECTRUM with size 1 for example. In that case, we must ask tango
        // for the actual value.
        TDeviceAttribute* p = first;
        std::vector<std::string> attr_names;
        for (size_t n =0; n < nelems; ++n, ++p) {
            if ( (p->data_format != Tango::FMT_UNKNOWN) || (p->has_failed()) )
                continue;
            if ( (p->get_dim_x() == 1) && (p->get_dim_y() == 0 ) ) {
                attr_names.push_back(p->name);
            } else if (p->get_dim_y() == 0) {
                p->data_format = Tango::SPECTRUM;
            } else {
                p->data_format = Tango::IMAGE;
            }
        }
        if (attr_names.size()) {
            std::auto_ptr<Tango::AttributeInfoListEx> attr_infos;
            {
                AutoPythonAllowThreads guard;
                p = first;
                try
                {
                    attr_infos.reset(dev_proxy.get_attribute_config_ex(attr_names));
                    for (size_t n=0, m=0; n < nelems; ++n, ++p) {
                        if ((p->data_format == Tango::FMT_UNKNOWN) && (!p->has_failed())) {
                            p->data_format = (*attr_infos)[m++].data_format;
                        }
                    }
                }
                catch(Tango::DevFailed &df)
                {
                    // if we fail to get info about the missing attributes from
                    // the server (because it as shutdown, for example) we assume
                    // that they are SCALAR since dim_x is 1
                    for (size_t n=0; n < nelems; ++n, ++p) {
                        if ((p->data_format == Tango::FMT_UNKNOWN) && (!p->has_failed())) {
                            p->data_format = Tango::SCALAR;
                        }
                    }
                }
            }
        }
        return;
    }

    /// @param self The DeviceAttribute instance that the new python object
    /// will represent. It must be allocated with new. The new python object
    /// will handle it's destruction. There's never any reason to delete it
    /// manually after a call to this: Even if this function fails, the
    /// responsibility of destroying it will already be in py_value side or
    /// the object will already be destroyed.
    template<typename TDeviceAttribute>
    boost::python::object convert_to_python(TDeviceAttribute* self, PyTango::ExtractAs extract_as)
    {
        using namespace boost::python;
        object py_value;
        try {
            py_value = object(
                    handle<>(
                        to_python_indirect<
                            TDeviceAttribute*,
                            detail::make_owning_holder>()(self)));
        } catch (...) {
            delete self;
            throw;
        }

        update_values(*self, py_value, extract_as);
        return py_value;
    }

    /// See the other convert_to_python version. Here we get a vector of
    /// DeviceAttributes. The responsibility to deallocate it is always from
    /// the caller (we will make a copy). This responsibility is unavoidable
    /// as we get a reference to auto_ptr -> the caller must use an auto_ptr,
    /// so the memory will finally be deleted.
    template<typename TDeviceAttribute>
    boost::python::object convert_to_python(const std::auto_ptr<std::vector<TDeviceAttribute> >& dev_attr_vec, Tango::DeviceProxy & dev_proxy, PyTango::ExtractAs extract_as)
    {
        update_data_format(dev_proxy, &(*dev_attr_vec)[0], dev_attr_vec->size());

        // Convert the c++ vector of DeviceAttribute into a pythonic list
        boost::python::list ls;
        typename std::vector<TDeviceAttribute>::const_iterator i, e = dev_attr_vec->end();
        for (i = dev_attr_vec->begin(); i != e; ++i)
            ls.append( convert_to_python(new TDeviceAttribute(*i), extract_as) );
        return ls;
    }

    /// Convert a DeviceAttribute to python (useful for read*)
    /// @param dev_attr Should be a pointer allocated with new. You can forget
    ///                 about deallocating this object (python will do it) even
    ///                 if the call to convert_to_python fails.
    /// @param dev_proxy Device proxy where the value was got. DeviceAttribute
    ///                 sent by older tango versions do not have all the
    ///                 necessary information to extract the values, so we
    ///                 may need to ask.
    /// @param extract_as See ExtractAs enum.
    template<typename TDeviceAttribute>
    boost::python::object convert_to_python(TDeviceAttribute* dev_attr, Tango::DeviceProxy & dev_proxy, PyTango::ExtractAs extract_as)
    {
        update_data_format(dev_proxy, dev_attr, 1);
        return convert_to_python(dev_attr, extract_as);
    }
}
