/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <algorithm>
#include <tango.h>
#include <pybind11/pybind11.h>
#include <server/device_impl.h>
#include <pyutils.h>
#include <pytgutils.h>
#include <server/attribute.h>
#include <memory>
#include <exception.h>
#include <thread>
#include <server/attr.h>
#include <server/pipe.h>


namespace py = pybind11;

namespace PyDeviceImpl
{
    /* **********************************
     * change event USING set_value
     * **********************************/
    inline void push_change_event(Tango::DeviceImpl& self, std::string& name)
    {
        std::cerr << "push_change_event 1" << std::endl;
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        if ("state" != name_lower && "status" != name_lower)
        {
            Tango::Except::throw_exception(
                "PyDs_InvalidCall",
                "push_change_event without data parameter is only allowed for "
                "state and status attributes.", "DeviceImpl::push_change_event");
        }
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, py::object& data)
    {
        std::cerr << "push_change_event 2" << std::endl;
//        boost::python::extract<Tango::DevFailed> except_convert(data);
//        if (except_convert.check()) {
//            AutoPythonAllowThreads python_guard_ptr;
//            Tango::AutoTangoMonitor tango_guard(&self);
//            Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
//            python_guard_ptr.giveup();
//            attr.fire_change_event(
//                           const_cast<Tango::DevFailed*>( &except_convert() ));
//            return;
//        }
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, data);
        attr.fire_change_event();
    }

    // Special variation for encoded data type
    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, std::string& str_data,
                                  std::string& data)
    {
        std::cerr << "push_change_event 3" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_change_event();
    }

    // Special variation for encoded data type
    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, std::string& str_data,
                                  py::object& data)
    {
        std::cerr << "push_change_event 4" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                                  long x)
    {
        std::cerr << "push_change_event 5" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, data, x);
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                                  long x, long y)
    {
        std::cerr << "push_change_event 6" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, data, x, y);
        attr.fire_change_event();
    }

    /* **********************************
     * change event USING set_value_date_quality
     * **********************************/

    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality)
    {
        std::cerr << "push_change_event 7" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, data, t, quality);
        attr.fire_change_event();
    }

    // Special variation for encoded data type
    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, std::string& str_data,
                                  std::string& data, double t, Tango::AttrQuality quality)
    {
        std::cerr << "push_change_event 8" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_change_event();
    }

    // Special variation for encoded data type
    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, std::string& str_data,
                                 py::object& data, double t, Tango::AttrQuality quality)
    {
        std::cerr << "push_change_event 9" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality, long x)
    {
        std::cerr << "push_change_event 10" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, data, t, quality, x);
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality, long x, long y)
    {
        std::cerr << "push_change_event 11" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, data, t, quality, x, y);
        attr.fire_change_event();
    }

    /* **********************************
     * archive event USING set_value
     * **********************************/
    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, py::object& data)
    {
//        boost::python::extract<Tango::DevFailed> except_convert(data);
//        if (except_convert.check()) {
//            AutoPythonAllowThreads python_guard_ptr;
//            Tango::AutoTangoMonitor tango_guard(&self);
//            Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
//            python_guard_ptr.giveup();
//            attr.fire_archive_event(
//                           const_cast<Tango::DevFailed*>( &except_convert() ));
//            return;
//        }
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, data);
        attr.fire_archive_event();
    }

    // Special variation for encoded data type
    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, std::string& str_data,
                                   std::string& data)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_archive_event();
    }

    // Special variation for encoded data type
    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, std::string& str_data,
                                  py::object& data)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                           long x)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, data, x);
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                           long x, long y)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, data, x, y);
        attr.fire_archive_event();
    }

    /* **********************************
     * archive event USING set_value_date_quality
     * **********************************/

    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, data, t, quality);
        attr.fire_archive_event();
    }

    // Special variation for encoded data type
    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, std::string& str_data,
                                   std::string& data, double t, Tango::AttrQuality quality)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_archive_event();
    }

    // Special variation for encoded data type
    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, std::string& str_data,
                                  py::object& data, double t, Tango::AttrQuality quality)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality, long x)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, data, t, quality, x);
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl& self, std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality, long x, long y)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, data, t, quality, x, y);
        attr.fire_archive_event();
    }

    /* **********************************
     * user event USING set_value
     * **********************************/
    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        attr.fire_event(filt_names, filt_vals);
    }

    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals, py::object& data)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, data);
        attr.fire_event(filt_names, filt_vals);
    }

    // Special variation for encoded data type
    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals,
                           std::string& str_data, std::string& data)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_event(filt_names, filt_vals);
    }

    // Special variation for encoded data type
    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals,
                           std::string& str_data, py::object& data)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_event(filt_names, filt_vals);
    }

    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals, py::object& data,
                           long x)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, data, x);
        attr.fire_event(filt_names, filt_vals);
    }

    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals, py::object& data,
                           long x, long y)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value(attr, data, x, y);
        attr.fire_event(filt_names, filt_vals);
    }

    /* ***************************************
     * user event USING set_value_date_quality
     * **************************************/

    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals,
            py::object& data, double t, Tango::AttrQuality quality)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, data, t, quality);
        attr.fire_event(filt_names, filt_vals);
    }

    // Special variation for encoded data type
    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals,
            std::string& str_data, std::string& data,
            double t, Tango::AttrQuality quality)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_event(filt_names, filt_vals);
    }

    // Special variation for encoded data type
    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals,
            std::string& str_data, py::object& data,
            double t, Tango::AttrQuality quality)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_event(filt_names, filt_vals);
    }

    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals,
            py::object& data, double t, Tango::AttrQuality quality, long x)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, data, t, quality, x);
        attr.fire_event(filt_names, filt_vals);
    }

    inline void push_event(Tango::DeviceImpl& self, std::string& name,
            std::vector<std::string>& filt_names, std::vector<double>& filt_vals,
            py::object& data, double t, Tango::AttrQuality quality, long x, long y)
    {
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        PyAttribute::set_value_date_quality(attr, data, t, quality, x, y);
        attr.fire_event(filt_names, filt_vals);
    }

    /* **********************************
     * data ready event
     * **********************************/
    inline void push_data_ready_event(Tango::DeviceImpl& self, std::string& name, long ctr)
    {
        std::cerr << "push_data_ready_event" << std::endl;
        AutoPythonAllowThreads python_guard_ptr;
        Tango::AutoTangoMonitor tango_guard(&self);
        Tango::Attribute& attr = self.get_device_attr()->get_attr_by_name(name.c_str());
        python_guard_ptr.giveup();
        self.push_data_ready_event(name, ctr);
    }

    /* **********************************
     * pipe event
     * **********************************/
    inline void push_pipe_event(Tango::DeviceImpl& self, std::string& pipe_name, py::object& pipe_data)
    {
//        boost::python::extract<Tango::DevFailed> except_convert(pipe_data);
//        if (except_convert.check()) {
//            self.push_pipe_event(pipe_name, const_cast<Tango::DevFailed*>(&except_convert()));
//            return;
//        }
        Tango::DevicePipeBlob dpb;
        bool reuse = false;
        PyTango::Pipe::set_value(dpb, pipe_data);
        self.push_pipe_event(pipe_name, &dpb, reuse);
    }

//    void check_attribute_method_defined(PyObject *self,
//                                        const std::string& attr_name,
//                                        const std::stringWrap &method_name)
//    {
//        bool exists, is_method;
//
//        is_method_defined(self, method_name, exists, is_method);
//
//        if (!exists)
//        {
//            std::stringstream o;
//            o << "Wrong definition of attribute " << attr_name
//              << "\nThe attribute method " << method_name
//              << " does not exist in your class!" << ends;
//
//            Tango::Except::throw_exception(
//                    "PyDs_WrongCommandDefinition",
//                    o.str(),
//                    "check_attribute_method_defined");
//        }
//
//        if(!is_method)
//        {
//            std::stringstream o;
//            o << "Wrong definition of attribute " << attr_name
//              << "\nThepy::object " << method_name
//              << " exists in your class but is not a Python method" << ends;
//
//            Tango::Except::throw_exception(
//                    "PyDs_WrongCommandDefinition",
//                    o.str(),
//                    "check_attribute_method_defined");
//        }
//    }
//
    void add_attribute(Tango::DeviceImpl& self, const Tango::Attr &c_new_attr,
                       const std::string& read_meth_name,
                       const std::string& write_meth_name,
                       const std::string& is_allowed_meth_name)
    {
        Tango::Attr &new_attr = const_cast<Tango::Attr &>(c_new_attr);

        std::string read_name_met, write_name_met, is_allowed_method;
        std::string attr_name = new_attr.get_name();

        std::cerr << "add_attribute name " << attr_name << std::endl;
        if (read_meth_name == "")
        {
            read_name_met = "read_" + attr_name;
        }
        else
        {
            read_name_met = read_meth_name;
        }
        if (write_meth_name == "")
        {
            write_name_met = "write_" + attr_name;
        }
        else
        {
            write_name_met = write_meth_name;
        }
        if (is_allowed_meth_name == "")
        {
            is_allowed_method = "is_" + attr_name + "_allowed";
        }
        else
        {
            is_allowed_method = is_allowed_meth_name;
        }

        Tango::AttrWriteType attr_write = new_attr.get_writable();

        //
        // Create the attributepy::object according to attribute format
        //

        PyScaAttr *sca_attr_ptr = nullptr;
        PySpecAttr *spec_attr_ptr = nullptr;
        PyImaAttr *ima_attr_ptr= nullptr;
        PyAttr *py_attr_ptr = nullptr;
        Tango::Attr *attr_ptr = nullptr;

        long x, y;
        std::vector<Tango::AttrProperty> &def_prop = new_attr.get_user_default_properties();
        Tango::AttrDataFormat attr_format = new_attr.get_format();
        long attr_type = new_attr.get_type();

        switch (attr_format)
        {
            case Tango::SCALAR:
                sca_attr_ptr = new PyScaAttr(attr_name, attr_type, attr_write, def_prop);
                py_attr_ptr = sca_attr_ptr;
                attr_ptr = sca_attr_ptr;
                break;

            case Tango::SPECTRUM:
                x = (static_cast<Tango::SpectrumAttr &>(new_attr)).get_max_x();
                spec_attr_ptr = new PySpecAttr(attr_name, attr_type, attr_write, x, def_prop);
                py_attr_ptr = spec_attr_ptr;
                attr_ptr = spec_attr_ptr;
                break;

            case Tango::IMAGE:
                x = (static_cast<Tango::ImageAttr &>(new_attr)).get_max_x();
                y = (static_cast<Tango::ImageAttr &>(new_attr)).get_max_y();
                ima_attr_ptr = new PyImaAttr(attr_name, attr_type, attr_write, x, y, def_prop);
                py_attr_ptr = ima_attr_ptr;
                attr_ptr = ima_attr_ptr;
                break;

            default:
                std::stringstream o;
                o << "Attribute " << attr_name << " has an unexpected data format\n"
                  << "Please report this bug to the PyTango development team"
                  << ends;
                Tango::Except::throw_exception(
                        "PyDs_UnexpectedAttributeFormat",
                        o.str(),
                        "cpp_add_attribute");
                break;
        }

        py_attr_ptr->set_read_name(read_name_met);
        py_attr_ptr->set_write_name(write_name_met);
        py_attr_ptr->set_allowed_name(is_allowed_method);

        if (new_attr.get_memorized())
            attr_ptr->set_memorized();
        attr_ptr->set_memorized_init(new_attr.get_memorized_init());

        attr_ptr->set_disp_level(new_attr.get_disp_level());
        attr_ptr->set_polling_period(new_attr.get_polling_period());
        attr_ptr->set_change_event(new_attr.is_change_event(),
                                   new_attr.is_check_change_criteria());
        attr_ptr->set_archive_event(new_attr.is_archive_event(),
                                    new_attr.is_check_archive_criteria());
        attr_ptr->set_data_ready_event(new_attr.is_data_ready_event());
        //
        // Install attribute in Tango.
        //
        self.add_attribute(attr_ptr);
    }

//    py::object* get_attribute_config(Tango::DeviceImpl& self, py::object& py_attr_name_seq)
   py::object get_attribute_config(Tango::DeviceImpl& self, py::object& py_attr_name_seq)
    {
//        Tango::DevVarStringArray par;
//        convert2array(py_attr_name_seq, par);
//
//        Tango::AttributeConfigList *attr_conf_list_ptr =
//            self.get_attribute_config(par);
//
//        boost::python::list ret = to_py(*attr_conf_list_ptr);
//        delete attr_conf_list_ptr;
//
//        return boost::python::incref(ret.ptr());
        return py::none();
    }

    void set_attribute_config(Tango::DeviceImpl& self, py::object& py_attr_conf_list)
    {
//        py::object obj = ...;
//        MyClass *cls = obj.cast<MyClass *>();
//        Tango::AttributeConfigList attr_conf_list;
//        from_py_object(py_attr_conf_list, attr_conf_list);
//        self.set_attribute_config(attr_conf_list);
    }

//    py::object get_attribute_config_2(Tango::DeviceImpl& self, py::object& attr_name_seq)
//    {
//        Tango::DevVarStringArray par;
//        convert2array(attr_name_seq, par);
//
//        Tango::AttributeConfigList_2 *attr_conf_list_ptr =
//            self.get_attribute_config_2(par);
//
//        boost::python::list ret = to_py(*attr_conf_list_ptr);
//        delete attr_conf_list_ptr;
//
//        return boost::python::incref(ret.ptr());
//    }

//    PyObject* get_attribute_config_3(Tango::Device_3Impl &self, object &attr_name_seq)
//    {
//        Tango::DevVarStringArray par;
//        convert2array(attr_name_seq, par);
//
//        Tango::AttributeConfigList_3 *attr_conf_list_ptr =
//            self.get_attribute_config_3(par);
//
//        boost::python::list ret = to_py(*attr_conf_list_ptr);
//        delete attr_conf_list_ptr;
//
//        return boost::python::incref(ret.ptr());
//    }
//
//    void set_attribute_config_3(Tango::Device_3Impl &self, object &py_attr_conf_list)
//    {
//        Tango::AttributeConfigList_3 attr_conf_list;
//        from_py_object(py_attr_conf_list, attr_conf_list);
//        self.set_attribute_config_3(attr_conf_list);
//    }
}

//DeviceImplWrap::DeviceImplWrap(DeviceClassWrap *cl, std::string& name, py::object& pyself)
//  : Tango::Device_5Impl(cl, name)
DeviceImplWrap::DeviceImplWrap(py::object& pyself, DeviceClassWrap *cl, std::string& name)
  : Tango::Device_5Impl(cl, name)
{
    py_self = pyself;
}

DeviceImplWrap::DeviceImplWrap(py::object& pyself, DeviceClassWrap *cl, std::string& name, std::string& descr)
  : Tango::Device_5Impl(cl, name, descr)
{
    py_self = pyself;
}

DeviceImplWrap::DeviceImplWrap(py::object& pyself,
        DeviceClassWrap *cl,
        std::string& name,
        std::string& desc,
        Tango::DevState sta,
        std::string& status)
  : Tango::Device_5Impl(cl, name, desc, sta, status)
{
    py_self = pyself;
}

DeviceImplWrap::~DeviceImplWrap()
{
    delete_device();
}

void DeviceImplWrap::delete_dev()
{
    // Call here the delete_device method. It is defined in DeviceImplWrap
    // class which is already destroyed when the Tango kernel call the
    // delete_device method
    try
    {
        delete_device();
    }
    catch (Tango::DevFailed &e)
    {
        Tango::Except::print_exception(e);
    }
}

void DeviceImplWrap::py_delete_dev()
{
    DeviceImplWrap::delete_dev();
}

void DeviceImplWrap::init_device()
{
    std::cerr << "entering init_device" << std::endl;
    std::thread::id thread_id = std::this_thread::get_id();
    std::cerr << "always executed thread id " << thread_id << std::endl;
    // This method is called by the init command
    AutoPythonGILEnsure __py_lock;
    try
    {
        py_self.attr("init_device")();
    }
    catch(py::error_already_set &eas)
    {
        std::cerr << "1" << std::endl;
        handle_python_exception(eas);
    }
}

void DeviceImplWrap::delete_device()
{
    AutoPythonGILEnsure __py_lock;
    try {
        if (is_method_callable(py_self, "delete_device")) {
            py_self.attr("delete_device")();
        }
        else {
            Tango::Device_5Impl::delete_device();
        }
    }
    catch(py::error_already_set &eas) {
        std::cerr << "2" << std::endl;
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

void DeviceImplWrap::default_delete_device()
{
    this->Tango::Device_5Impl::delete_device();
}

bool DeviceImplWrap::_is_attribute_polled(const std::string& att_name)
{
    return this->is_attribute_polled(att_name);
}

bool DeviceImplWrap::_is_command_polled(const std::string& cmd_name)
{
    std::cerr << &py_self << std::endl;
    return this->is_command_polled(cmd_name);
}

int DeviceImplWrap::_get_attribute_poll_period(const std::string& att_name)
{
    return this->get_attribute_poll_period(att_name);
}

int DeviceImplWrap::_get_command_poll_period(const std::string& cmd_name)
{
    return this->get_command_poll_period(cmd_name);
}

void DeviceImplWrap::_poll_attribute(const std::string& att_name, int period)
{
    this->poll_attribute(att_name, period);
}

void DeviceImplWrap::_poll_command(const std::string& cmd_name, int period)
{
    this->poll_command(cmd_name, period);
}

void DeviceImplWrap::_stop_poll_attribute(const std::string& att_name)
{
    this->stop_poll_attribute(att_name);
}

void DeviceImplWrap::_stop_poll_command(const std::string& cmd_name)
{
    this->stop_poll_command(cmd_name);
}

void DeviceImplWrap::always_executed_hook()
{
    std::thread::id thread_id = std::this_thread::get_id();
    AutoPythonGILEnsure __py_lock;
    try {
        if (is_method_callable(py_self, "always_executed_hook")) {
            py_self.attr("always_executed_hook")();
        }
        else {
            Tango::Device_5Impl::always_executed_hook();
        }
    }
    catch(py::error_already_set &eas) {
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

void DeviceImplWrap::default_always_executed_hook()
{
    this->Tango::Device_5Impl::always_executed_hook();
}

void DeviceImplWrap::read_attr_hardware(std::vector<long> &attr_list)
{
    std::thread::id thread_id = std::this_thread::get_id();
    AutoPythonGILEnsure __py_lock;
    try {
        if (is_method_callable(py_self, "read_attr_hardware")) {
            py_self.attr("read_attr_hardware")(attr_list);
        }
        else {
            Tango::Device_5Impl::read_attr_hardware(attr_list);
        }
    }
    catch(py::error_already_set &eas) {
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

void DeviceImplWrap::default_read_attr_hardware(std::vector<long> &attr_list)
{
    this->Tango::Device_5Impl::read_attr_hardware(attr_list);
}

void DeviceImplWrap::write_attr_hardware(std::vector<long> &attr_list)
{
    AutoPythonGILEnsure __py_lock;
    try {
        if (is_method_callable(py_self, "write_attr_hardware")) {
            py_self.attr("write_attr_hardware")(attr_list);
        }
        else {
            Tango::Device_5Impl::write_attr_hardware(attr_list);
        }
    }
    catch(py::error_already_set &eas) {
        std::cerr << "5" << std::endl;
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

void DeviceImplWrap::default_write_attr_hardware(std::vector<long> &attr_list)
{
    this->Tango::Device_5Impl::write_attr_hardware(attr_list);
}

Tango::DevState DeviceImplWrap::dev_state()
{
    AutoPythonGILEnsure __py_lock;
    try {
        if (is_method_callable(py_self, "dev_state")) {
            py::object ret = py_self.attr("dev_state")();
            return ret.cast<Tango::DevState>();
        }
        else {
            return Tango::Device_5Impl::dev_state();
        }
    }
    catch(py::error_already_set &eas) {
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

Tango::DevState DeviceImplWrap::default_dev_state()
{
    return this->Tango::Device_5Impl::dev_state();
}

Tango::ConstDevString DeviceImplWrap::dev_status()
{
    AutoPythonGILEnsure __py_lock;
    try {
        if (is_method_callable(py_self, "dev_status")) {
            py::object ret = py_self.attr("dev_status")();
            this->the_status = ret.cast<std::string>();
            return this->the_status.c_str();
        }
        else {
            this->the_status = Tango::Device_5Impl::dev_status();
            return this->the_status.c_str();
        }
    }
    catch(py::error_already_set &eas) {
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

Tango::ConstDevString DeviceImplWrap::default_dev_status()
{
    this->the_status = this->Tango::Device_5Impl::dev_status();
    return this->the_status.c_str();
}

void DeviceImplWrap::signal_handler(long signo)
{
    try
    {
        if (is_method_callable(py_self, "signal_handler")) {
            py_self.attr("signal_handler")(signo);
        }
        else {
            Tango::Device_5Impl::signal_handler(signo);
        }
    }
    catch(Tango::DevFailed &df)
    {
        long nb_err = df.errors.length();
        df.errors.length(nb_err + 1);

        df.errors[nb_err].reason = CORBA::string_dup(
            "PyDs_UnmanagedSignalHandlerException");
        df.errors[nb_err].desc = CORBA::string_dup(
            "An unmanaged Tango::DevFailed exception occurred in signal_handler");
        df.errors[nb_err].origin = CORBA::string_dup("DeviceImpl.signal_handler");
        df.errors[nb_err].severity = Tango::ERR;

        Tango::Except::print_exception(df);
    }
}

void DeviceImplWrap::default_signal_handler(long signo)
{
    this->Tango::Device_5Impl::signal_handler(signo);
}

void export_device_impl(py::module &m) {
    py::class_<DeviceImplWrap>(m, "DeviceImpl")
        .def(py::init([](py::object& pyself, DeviceClassWrap *cppdev, std::string& name) {
            DeviceImplWrap* cpp = new DeviceImplWrap(pyself, cppdev, name);
            return cpp;
        }))
        .def(py::init([](py::object& pyself, DeviceClassWrap *cppdev, std::string& name, std::string& descr) {
            DeviceImplWrap* cpp = new DeviceImplWrap(pyself, cppdev, name, descr);
            return cpp;
        }))
       .def(py::init([](py::object& pyself, DeviceClassWrap *cppdev, std::string& name,
            std::string& descr, Tango::DevState dstate, std::string& status) {
            DeviceImplWrap* cpp = new DeviceImplWrap(pyself, cppdev, name, descr, dstate, status);
            return cpp;
        }))
        .def("set_state", [](DeviceImplWrap& self, const Tango::DevState& new_state) -> void {
            self.set_state(new_state);
        })
        .def("get_state", [](DeviceImplWrap& self) -> Tango::DevState {
            return self.get_state();
        })
        .def("get_prev_state", [](DeviceImplWrap& self) -> Tango::DevState {
            return self.get_prev_state();
        })
        .def("get_name", [](DeviceImplWrap& self) -> std::string {
            return self.get_name();
        })
        .def("get_device_attr", [](DeviceImplWrap& self) {
            return self.get_device_attr();
        })
        .def("register_signal", [](DeviceImplWrap& self, long signo, bool own_handler) -> void {
            self.register_signal(signo, own_handler);
        }, py::arg("signo"), py::arg("own_handler")=false)
        .def("unregister_signal", [](DeviceImplWrap& self, long signo) -> void {
            self.unregister_signal(signo);
        })
        .def("get_status", [](DeviceImplWrap& self) -> std::string {
            return self.get_status();
        })
        .def("set_status", [](DeviceImplWrap& self, const std::string& new_status) -> void {
            self.set_status(new_status);
        })
        .def("append_status", [](DeviceImplWrap& self, const std::string& stat, bool new_line) -> void {
            self.append_status(stat, new_line);
        }, py::arg("stat"), py::arg("new_line")=false)
       .def("dev_state", [](DeviceImplWrap& self) -> Tango::DevState {
           return self.default_dev_state();
        })
        .def("dev_status", [](DeviceImplWrap& self) -> Tango::ConstDevString {
            return self.default_dev_status();
        })
        .def("get_attribute_config", [](DeviceImplWrap& self, py::object& py_attr_name_seq) {
            return PyDeviceImpl::get_attribute_config(self, py_attr_name_seq);
        })
        .def("set_change_event", [](DeviceImplWrap& self, std::string& attr_name, bool implemented, bool detect ) -> void {
            self.set_change_event(attr_name, implemented, detect);
        }, py::arg("attr_name"), py::arg("implemented"), py::arg("detect")=true)
        .def("set_archive_event", [](DeviceImplWrap& self, std::string& attr_name, bool implemented, bool detect) -> void {
            self.set_archive_event(attr_name, implemented, detect);
        }, py::arg("attr_name"), py::arg("implemented"), py::arg("detect")=true)
        .def("_add_attribute", [](DeviceImplWrap& self, const Tango::Attr &new_attr,
                       const std::string& read_meth_name,
                       const std::string& write_meth_name,
                       const std::string& is_allowed_meth_name) -> void {
            PyDeviceImpl::add_attribute(self, new_attr, read_meth_name, write_meth_name, is_allowed_meth_name);
        })
        .def("_remove_attribute", [](DeviceImplWrap& self, std::string& att_name, bool freeit, bool clean_db) -> void {
            self.remove_attribute(att_name, freeit, clean_db);
        }, py::arg("att_name"), py::arg("freeit")=false, py::arg("clean_db")=true)

        //@TODO .def("get_device_class")
        //@TODO .def("get_db_device")

        .def("is_attribute_polled", [](DeviceImplWrap& self, const std::string& att_name) -> bool {
            return self._is_attribute_polled(att_name);
        })
        .def("is_command_polled", [](DeviceImplWrap& self, const std::string& cmd_name) -> bool {
            std::cerr << "is the command " << cmd_name << " polled" << std::endl;
            return self._is_command_polled(cmd_name);
        })
        .def("get_attribute_poll_period", [](DeviceImplWrap& self, const std::string& att_name) -> int {
            return self._get_attribute_poll_period(att_name);
        })
        .def("get_command_poll_period", [](DeviceImplWrap& self, const std::string& cmd_name) -> int {
            return self._get_command_poll_period(cmd_name);
        })
        .def("poll_attribute", [](DeviceImplWrap& self, const std::string& att_name, int period) -> void {
            self._poll_attribute(att_name, period);
        })
        .def("poll_command", [](DeviceImplWrap& self, const std::string& cmd_name, int period) -> void {
            self._poll_command(cmd_name, period);
        })
        .def("stop_poll_attribute", [](DeviceImplWrap& self, const std::string& att_name) -> void {
            self._stop_poll_attribute(att_name);
        })
        .def("stop_poll_command", [](DeviceImplWrap& self, const std::string& cmd_name) -> void {
            self._stop_poll_command(cmd_name);
        })
        .def("get_exported_flag", [](DeviceImplWrap& self) -> bool {
            return self.get_exported_flag();
        })
        .def("get_poll_ring_depth", [](DeviceImplWrap& self) -> long {
            return self.get_poll_ring_depth();
        })
        .def("get_poll_old_factor", [](DeviceImplWrap& self) -> long {
            return self.get_poll_old_factor();
        })
        .def("is_polled", [](DeviceImplWrap& self) -> bool {
            return self.is_polled();
        })
        .def("get_polled_cmd", [](DeviceImplWrap& self) -> std::vector<std::string> {
            return self.get_polled_cmd();
        })
        .def("get_polled_attr", [](DeviceImplWrap& self) -> std::vector<std::string> {
            return self.get_polled_attr();
        })
        .def("get_non_auto_polled_cmd", [](DeviceImplWrap& self) -> std::vector<std::string> {
            return self.get_non_auto_polled_cmd();
        })
        .def("get_non_auto_polled_attr", [](DeviceImplWrap& self) -> std::vector<std::string> {
            return self.get_non_auto_polled_attr();
        })

        //@TODO .def("get_poll_obj_list", &PyDeviceImpl::get_poll_obj_list)

        .def("stop_polling", [](DeviceImplWrap& self, bool stop) -> void {
            self.stop_polling(stop);
        }, py::arg("stop")=true)
        .def("check_command_exists", [](DeviceImplWrap& self, const std::string& cmd) -> void {
            self.check_command_exists(cmd);
        })

        //@TODO .def("get_command", &PyDeviceImpl::get_command)

        .def("get_dev_idl_version", [](DeviceImplWrap& self) {
            self.get_dev_idl_version();
        })
        .def("get_cmd_poll_ring_depth", [](DeviceImplWrap& self, std::string& cmd) -> long {
            return self.get_cmd_poll_ring_depth(cmd);
        })
        .def("get_attr_poll_ring_depth", [](DeviceImplWrap& self, std::string& attr) -> long {
            return self.get_attr_poll_ring_depth(attr);
        })
        .def("is_device_locked", [](DeviceImplWrap& self) -> bool {
            return self.is_device_locked();
        })
        .def("init_logger", [](DeviceImplWrap& self) -> void {
            self.init_logger();
        })
        .def("start_logging", [](DeviceImplWrap& self) -> void {
            self.start_logging();
        })
        .def("stop_logging", [](DeviceImplWrap& self) -> void {
            self.stop_logging();
        })
        .def("set_exported_flag", [](DeviceImplWrap& self, bool exp) -> void {
            self.set_exported_flag(exp);
        })
        .def("set_poll_ring_depth", [](DeviceImplWrap& self, long depth) -> void {
            self.set_poll_ring_depth(depth);
        })
        .def("push_change_event", [](DeviceImplWrap& self, std::string& attr_name) -> void {
            PyDeviceImpl::push_change_event(self, attr_name);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, py::object& data) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, std::string& str_data, std::string& data) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, str_data, data);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, std::string& str_data, py::object& data) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, str_data, data);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, py::object& data, long x) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, x);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, py::object& data, long x, long y) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, x, y);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, py::object& data, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, quality);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, std::string& str_data, std::string& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, str_data, data, t, quality);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, std::string& str_data, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, str_data, data, t, quality);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality, long x) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, t, quality, x);
        })
        .def("push_change_event",[](DeviceImplWrap& self, std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality, long x, long y) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, t, quality, x, y);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, py::object& data) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, std::string& str_data, std::string& data) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, str_data, data);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, std::string& str_data, py::object& data) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, str_data, data);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, py::object& data, long x) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, x);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, py::object& data, long x, long y) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, x, y);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, t, quality);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, std::string& str_data, std::string& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, str_data, data, t, quality);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, std::string& str_data, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, str_data, data, t, quality);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality, long x) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, t, quality, x);
        })
        .def("push_archive_event", [](DeviceImplWrap& self, std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality, long x, long y) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, t, quality, x, y);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, py::object& data) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, std::string& str_data, std::string& data) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, str_data, data);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, std::string& str_data, py::object& data) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, str_data, data);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, py::object& data, long x) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, x);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, py::object& data, long x, long y) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, x, y);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, t, quality);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, std::string& str_data, std::string& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, str_data, data, t, quality);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, std::string& str_data, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, str_data, data, t, quality);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, py::object& data, double t, Tango::AttrQuality quality, long x) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, t, quality, x);
        })
        .def("push_event", [](DeviceImplWrap& self, std::string& attr_name, std::vector<std::string>& filt_names, std::vector<double>& filt_vals, py::object& data, double t, Tango::AttrQuality quality, long x, long y) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, t, quality, x, y);
        })
        .def("push_data_ready_event", [](DeviceImplWrap& self, std::string& attr_name, long ctr) -> void {
            PyDeviceImpl::push_data_ready_event(self, attr_name, ctr);
        })

//         @TODO
//        .def("push_att_conf_event", [](DeviceImplWrap& self) -> void {
//            self.push_att_conf_event();
//            should be this??????????????
//            self.push_att_conf_event(Attribute *);
//        })

        .def("_push_pipe_event", [](DeviceImplWrap& self, std::string& pipe_name, py::object& pipe_data) -> void {
            PyDeviceImpl::push_pipe_event(self, pipe_name, pipe_data);
        })
        .def("get_logger", [](DeviceImplWrap& self) {
            return self.get_logger();
        })
        .def("__debug_stream", [](DeviceImplWrap& self, const std::string& msg) {
            if (self.get_logger()->is_debug_enabled()) {
                self.get_logger()->debug_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("__info_stream", [](DeviceImplWrap& self, const std::string& msg) -> void {
            if (self.get_logger()->is_info_enabled()) {
                self.get_logger()->info_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("__warn_stream", [](DeviceImplWrap& self, const std::string& msg) -> void {
            if (self.get_logger()->is_warn_enabled()) {
                self.get_logger()->warn_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("__error_stream", [](DeviceImplWrap& self, const std::string& msg) -> void {
            if (self.get_logger()->is_error_enabled()) {
                self.get_logger()->error_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("__fatal_stream", [](DeviceImplWrap& self, const std::string& msg) -> void {
            if (self.get_logger()->is_fatal_enabled()) {
                self.get_logger()->fatal_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("get_min_poll_period", [](DeviceImplWrap& self) {
            self.get_min_poll_period();
        })
        .def("get_cmd_min_poll_period", [](DeviceImplWrap& self) {
            self.get_cmd_min_poll_period();
        })
        .def("get_attr_min_poll_period", [](DeviceImplWrap& self) {
            self.get_attr_min_poll_period();
        })
        .def("is_there_subscriber", [](DeviceImplWrap& self, const std::string& att_name, Tango::EventType event_type) -> bool{
            self.is_there_subscriber(att_name, event_type);
        })
        .def("init_device", [](DeviceImplWrap& self) {
            self.init_device();
        })
        .def("delete_device", [](DeviceImplWrap& self) {
            self.default_delete_device();
        })
        .def("always_executed_hook", [](DeviceImplWrap& self) {
            self.default_always_executed_hook();
        })
        .def("read_attr_hardware", [](DeviceImplWrap& self, std::vector<long> &attr_list) {
            self.default_read_attr_hardware(attr_list);
        })
        .def("write_attr_hardware", [](DeviceImplWrap& self, std::vector<long> &attr_list) {
            self.default_write_attr_hardware(attr_list);
        })
        .def("signal_handler", [](DeviceImplWrap& self, long signo) {
            self.default_signal_handler(signo);
        })
//        .def("get_attribute_config_2", &PyDevice_2Impl::get_attribute_config_2)
//        .def("get_attribute_config_3", &PyDevice_3Impl::get_attribute_config_3)
//        .def("set_attribute_config_3", &PyDevice_3Impl::set_attribute_config_3)
    ;
}
