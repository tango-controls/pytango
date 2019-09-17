/******************************************************************************
0  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#pragma once

#include <tango.h>
#include <pybind11/pybind11.h>
#include <pyutils.h>
#include <tgutils.h>
#include <defs.h>
#include <memory>

namespace py = pybind11;

namespace PyDeviceAttribute {

//    template<long tangoTypeConst> static
//    void _update_array_values_as_tuples(Tango::DeviceAttribute& self, bool isImage, py::object py_value);

//    /// Set the value of a DeviceAttribute from python (useful for write*)
    void reset(Tango::DeviceAttribute& self, const Tango::AttributeInfo &attr_info, py::object py_value);
    void reset(Tango::DeviceAttribute & self, const std::string& attr_name, Tango::DeviceProxy &dev_proxy, py::object py_value);

    void update_values(Tango::DeviceAttribute& self, py::object& py_value);

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
            std::unique_ptr<Tango::AttributeInfoListEx> attr_infos;
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
                catch(Tango::DevFailed &)
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
    py::object convert_to_python(TDeviceAttribute* dev_attr)
    {
        py::object py_value;
        Tango::DeviceAttribute result = std::move(*dev_attr);
        py_value = py::cast(result);
        update_values(result, py_value);
        return py_value;
    }

    /// See the other convert_to_python version. Here we get a vector of
    /// DeviceAttributes. The responsibility to deallocate it is always from
    /// the caller (we will make a copy). This responsibility is unavoidable
    /// as we get a reference to auto_ptr -> the caller must use an auto_ptr,
    /// so the memory will finally be deleted.
    template<typename TDeviceAttribute>
    py::list convert_to_python(const std::unique_ptr<std::vector<TDeviceAttribute> >& dev_attr_vec,
            Tango::DeviceProxy & dev_proxy) {
        py::list ls;
        if (dev_attr_vec->empty()) {
            return ls;
        }
        update_data_format(dev_proxy, &(*dev_attr_vec)[0], (size_t)dev_attr_vec->size());

        // Convert the c++ vector of DeviceAttribute into a pythonic list
        typename std::vector<TDeviceAttribute>::const_iterator i, e = dev_attr_vec->end();
        for (i = dev_attr_vec->begin(); i != e; ++i)
            ls.append(convert_to_python(new TDeviceAttribute(*i)));
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
    template<typename TDeviceAttribute>
    py::object convert_to_python(TDeviceAttribute* dev_attr, Tango::DeviceProxy& dev_proxy)
    {
        update_data_format(dev_proxy, dev_attr, (size_t)1);
        return convert_to_python(dev_attr);
    }

    template<long tangoTypeConst>
    static inline void _fill_scalar_attribute(Tango::DeviceAttribute& dev_attr, const py::object& py_value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType value = py_value.cast<TangoScalarType>();
        dev_attr << const_cast<TangoScalarType&>(value);
    }

    template<>
    inline void _fill_scalar_attribute<Tango::DEV_STRING>(Tango::DeviceAttribute& dev_attr, const py::object& py_value)
    {
        std::string value = py_value.cast<std::string>();
        dev_attr << value;
    }

    template<>
    inline void _fill_scalar_attribute<Tango::DEV_ENCODED>(Tango::DeviceAttribute& dev_attr, const py::object& py_value)
    {
        if (py::len(py_value) != 2) {
            raise_(PyExc_TypeError, "Expecting a tuple of: encoded_format, encoded_data");
        }
        py::tuple tup = py::tuple(py_value);
        py::str encoded_format_str = tup[0];
        py::list encoded_data_obj = tup[1];
        std::string encoded_format = encoded_format_str.cast<std::string>();
        Py_ssize_t encoded_data_len = py::len(encoded_data_obj);
        std::vector<unsigned char> encoded_data;
        for (auto i=0; i<encoded_data_len; i++)
            encoded_data.push_back(encoded_data_obj[i].cast<unsigned char>());
        dev_attr.insert(encoded_format, encoded_data);
    }

}
