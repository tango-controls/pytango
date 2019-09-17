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

namespace py = pybind11;

#define SAFE_PUSH(dev, attr, attr_name) \
    std::string __att_name = attr_name; \
    AutoPythonAllowThreads python_guard_ptr; \
    Tango::AutoTangoMonitor tango_guard(&dev); \
    Tango::Attribute & attr = dev.get_device_attr()->get_attr_by_name(__att_name.c_str()); \
    python_guard_ptr.giveup();

#define SAFE_PUSH_CHANGE_EVENT(dev, attr_name, data) \
{ \
    SAFE_PUSH(dev, attr, attr_name) \
    PyAttribute::set_value(attr, data); \
    attr.fire_change_event(); \
}

#define SAFE_PUSH_CHANGE_EVENT_VARGS(dev, attr_name, data, ...) \
{ \
    SAFE_PUSH(dev, attr, attr_name) \
    PyAttribute::set_value(attr, data, __VA_ARGS__); \
    attr.fire_change_event(); \
}

#define SAFE_PUSH_CHANGE_EVENT_DATE_QUALITY(dev, attr_name, data, date, quality) \
{ \
    SAFE_PUSH(dev, attr, attr_name) \
    PyAttribute::set_value_date_quality(attr, data, date, quality); \
    attr.fire_change_event(); \
}

#define SAFE_PUSH_CHANGE_EVENT_DATE_QUALITY_VARGS(dev, attr_name, data, date, quality, ...) \
{ \
    SAFE_PUSH(dev, attr, attr_name) \
    PyAttribute::set_value_date_quality(attr, data, date, quality, __VA_ARGS__); \
    attr.fire_change_event(); \
}

#define SAFE_PUSH_ARCHIVE_EVENT(dev, attr_name, data) \
{ \
    SAFE_PUSH(dev, attr, attr_name) \
    PyAttribute::set_value(attr, data); \
    attr.fire_archive_event(); \
}

#define SAFE_PUSH_ARCHIVE_EVENT_VARGS(dev, attr_name, data, ...) \
{ \
    SAFE_PUSH(dev, attr, attr_name) \
    PyAttribute::set_value(attr, data, __VA_ARGS__); \
    attr.fire_archive_event(); \
}

#define SAFE_PUSH_ARCHIVE_EVENT_DATE_QUALITY(dev, attr_name, data, date, quality) \
{ \
    SAFE_PUSH(dev, attr, attr_name) \
    PyAttribute::set_value_date_quality(attr, data, date, quality); \
    attr.fire_archive_event(); \
}

#define SAFE_PUSH_ARCHIVE_EVENT_DATE_QUALITY_VARGS(dev, attr_name, data, date, quality, ...) \
{ \
    SAFE_PUSH(dev, attr, attr_name) \
    PyAttribute::set_value_date_quality(attr, data, date, quality, __VA_ARGS__); \
    attr.fire_archive_event(); \
}
/*
 * do we have to do this conversion or is it pybind11 automatic
 */
#define AUX_SAFE_PUSH_EVENT(dev, attr_name, filt_names, filt_vals) \
    std::vector<std::string> filt_names_; \
    std::vector<double> filt_vals_; \
    SAFE_PUSH(dev, attr, attr_name)
//    from_sequence<StdStringVector>::convert(filt_names, filt_names_); \
//    from_sequence<StdDoubleVector>::convert(filt_vals, filt_vals_); \

#define SAFE_PUSH_EVENT(dev, attr_name, filt_names, filt_vals, data) \
{ \
    AUX_SAFE_PUSH_EVENT(dev, attr_name, filt_names, filt_vals) \
    PyAttribute::set_value(attr, data); \
    attr.fire_event(filt_names_, filt_vals_); \
}

#define SAFE_PUSH_EVENT_VARGS(dev, attr_name, filt_names, filt_vals, data, ...) \
{ \
    AUX_SAFE_PUSH_EVENT(dev,attr_name, filt_names, filt_vals) \
    PyAttribute::set_value(attr, data, __VA_ARGS__); \
    attr.fire_event(filt_names_, filt_vals_); \
}

#define SAFE_PUSH_EVENT_DATE_QUALITY(dev, attr_name, filt_names, filt_vals, data, date, quality) \
{ \
    AUX_SAFE_PUSH_EVENT(dev, attr_name, filt_names, filt_vals) \
    PyAttribute::set_value_date_quality(attr, data, date, quality); \
    attr.fire_event(filt_names_, filt_vals_); \
}

#define SAFE_PUSH_EVENT_DATE_QUALITY_VARGS(dev, attr_name, filt_names, filt_vals, data, date, quality, ...) \
{ \
    AUX_SAFE_PUSH_EVENT(dev,attr_name, filt_names, filt_vals) \
    PyAttribute::set_value_date_quality(attr, data, date, quality, __VA_ARGS__); \
    attr.fire_event(filt_names_, filt_vals_); \
}

namespace PyDeviceImpl
{
    /* **********************************
     * change event USING set_value
     * **********************************/
    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name)
    {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        if ("state" != name_lower && "status" != name_lower)
        {
            Tango::Except::throw_exception(
                "PyDs_InvalidCall",
                "push_change_event without data parameter is only allowed for "
                "state and status attributes.", "DeviceImpl::push_change_event");
        }
        SAFE_PUSH(self, attr, name)
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, py::object& data)
    {
//        boost::python::extract<Tango::DevFailed> except_convert(data);
//        if (except_convert.check()) {
//            SAFE_PUSH(self, attr, name);
//            attr.fire_change_event(
//                           const_cast<Tango::DevFailed*>( &except_convert() ));
//            return;
//        }
        SAFE_PUSH_CHANGE_EVENT(self, name, data);
    }

    // Special variation for encoded data type
    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, const std::string& str_data,
                                  const std::string& data)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_change_event();
    }

    // Special variation for encoded data type
    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, const std::string& str_data,
                                  py::object& data)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                                  long x)
    {
        SAFE_PUSH_CHANGE_EVENT_VARGS(self, name, data, x);
    }

    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                                  long x, long y)
    {
        SAFE_PUSH_CHANGE_EVENT_VARGS(self, name, data, x, y);
    }

    /* **********************************
     * change event USING set_value_date_quality
     * **********************************/

    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH_CHANGE_EVENT_DATE_QUALITY(self, name, data, t, quality)
    }

    // Special variation for encoded data type
    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, const std::string& str_data,
                                  const std::string& data, double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_change_event();
    }

    // Special variation for encoded data type
    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, const std::string& str_data,
                                 py::object& data, double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality, long x)
    {
        SAFE_PUSH_CHANGE_EVENT_DATE_QUALITY_VARGS(self, name, data, t, quality, x)
    }

    inline void push_change_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality, long x, long y)
    {
        SAFE_PUSH_CHANGE_EVENT_DATE_QUALITY_VARGS(self, name, data, t, quality, x, y)
    }

    /* **********************************
     * archive event USING set_value
     * **********************************/
    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name)
    {
        SAFE_PUSH(self, attr, name)
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, py::object& data)
    {
//        boost::python::extract<Tango::DevFailed> except_convert(data);
//        if (except_convert.check()) {
//            SAFE_PUSH(self, attr, name);
//            attr.fire_archive_event(
//                           const_cast<Tango::DevFailed*>( &except_convert() ));
//            return;
//        }
        SAFE_PUSH_ARCHIVE_EVENT(self, name, data);
    }

    // Special variation for encoded data type
    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, const std::string& str_data,
                                   const std::string& data)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_archive_event();
    }

    // Special variation for encoded data type
    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, const std::string& str_data,
                                  py::object& data)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                           long x)
    {
        SAFE_PUSH_ARCHIVE_EVENT_VARGS(self, name, data, x);
    }

    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                           long x, long y)
    {
        SAFE_PUSH_ARCHIVE_EVENT_VARGS(self, name, data, x, y);
    }

    /* **********************************
     * archive event USING set_value_date_quality
     * **********************************/

    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH_ARCHIVE_EVENT_DATE_QUALITY(self, name, data, t, quality)
    }

    // Special variation for encoded data type
    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, const std::string& str_data,
                                   const std::string& data, double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_archive_event();
    }

    // Special variation for encoded data type
    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, const std::string& str_data,
                                  py::object& data, double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality, long x)
    {
        SAFE_PUSH_ARCHIVE_EVENT_DATE_QUALITY_VARGS(self, name, data, t, quality, x)
    }

    inline void push_archive_event(Tango::DeviceImpl& self, const std::string& name, py::object& data,
                                  double t, Tango::AttrQuality quality, long x, long y)
    {
        SAFE_PUSH_ARCHIVE_EVENT_DATE_QUALITY_VARGS(self, name, data, t, quality, x, y)
    }

    /* **********************************
     * user event USING set_value
     * **********************************/
    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                          py::object& filt_names, py::object& filt_vals)
    {
        AUX_SAFE_PUSH_EVENT(self, name, filt_names, filt_vals)
        attr.fire_event(filt_names_, filt_vals_);
    }

    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                          py::object& filt_names, py::object& filt_vals, py::object& data)
    {
        SAFE_PUSH_EVENT(self, name, filt_names, filt_vals, data)
    }

    // Special variation for encoded data type
    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                          py::object& filt_names, py::object& filt_vals,
                           const std::string& str_data, const std::string& data)
    {
        AUX_SAFE_PUSH_EVENT(self, name, filt_names, filt_vals)
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_event(filt_names_, filt_vals_);
    }

    // Special variation for encoded data type
    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                          py::object& filt_names, py::object& filt_vals,
                           const std::string& str_data, py::object& data)
    {
        AUX_SAFE_PUSH_EVENT(self, name, filt_names, filt_vals)
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_event(filt_names_, filt_vals_);
    }

    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                          py::object& filt_names, py::object& filt_vals, py::object& data,
                           long x)
    {
        SAFE_PUSH_EVENT_VARGS(self, name, filt_names, filt_vals, data, x)
    }

    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                          py::object& filt_names, py::object& filt_vals, py::object& data,
                           long x, long y)
    {
        SAFE_PUSH_EVENT_VARGS(self, name, filt_names, filt_vals, data, x, y)
    }

    /* ***************************************
     * user event USING set_value_date_quality
     * **************************************/

    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                           py::object& filt_names, py::object& filt_vals, py::object& data,
                           double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH_EVENT_DATE_QUALITY(self, name, filt_names, filt_vals, data, t, quality)
    }

    // Special variation for encoded data type
    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                           py::object& filt_names, py::object& filt_vals,
                           const std::string& str_data, const std::string& data,
                           double t, Tango::AttrQuality quality)
    {
        AUX_SAFE_PUSH_EVENT(self, name, filt_names, filt_vals)
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_event(filt_names_, filt_vals_);
    }

    // Special variation for encoded data type
    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                           py::object& filt_names, py::object& filt_vals,
                           const std::string& str_data, py::object& data,
                           double t, Tango::AttrQuality quality)
    {
        AUX_SAFE_PUSH_EVENT(self, name, filt_names, filt_vals)
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_event(filt_names_, filt_vals_);
    }

    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                           py::object& filt_names, py::object& filt_vals, py::object& data,
                           double t, Tango::AttrQuality quality, long x)
    {
        SAFE_PUSH_EVENT_DATE_QUALITY_VARGS(self, name, filt_names, filt_vals, data, t, quality, x)
    }

    inline void push_event(Tango::DeviceImpl& self, const std::string& name,
                           py::object& filt_names, py::object& filt_vals, py::object& data,
                           double t, Tango::AttrQuality quality, long x, long y)
    {
        SAFE_PUSH_EVENT_DATE_QUALITY_VARGS(self, name, filt_names, filt_vals, data, t, quality, x, y)
    }

    /* **********************************
     * data ready event
     * **********************************/
    inline void push_data_ready_event(Tango::DeviceImpl& self, const std::string& name, long ctr)
    {
        SAFE_PUSH(self, attr, name)
        self.push_data_ready_event(__att_name, ctr); //__att_name from SAFE_PUSH
    }

    /* **********************************
     * pipe event
     * **********************************/
    inline void push_pipe_event(Tango::DeviceImpl& self, const std::string& pipe_name, py::object& pipe_data)
    {
//        std::string __pipe_name;
//        from_str_to_char(pipe_name.ptr(), __pipe_name);
//        boost::python::extract<Tango::DevFailed> except_convert(pipe_data);
//        if (except_convert.check()) {
//            self.push_pipe_event(__pipe_name, const_cast<Tango::DevFailed*>(&except_convert()));
//            return;
//        }
//        Tango::DevicePipeBlob dpb;
//        bool reuse = false;
//        PyDevicePipe::set_value(dpb, pipe_data);
//        self.push_pipe_event(__pipe_name, &dpb, reuse);
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

        std::cout << "add_attribute name " << attr_name << std::endl;
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

        PyScaAttr *sca_attr_ptr = NULL;
        PySpecAttr *spec_attr_ptr = NULL;
        PyImaAttr *ima_attr_ptr= NULL;
        PyAttr *py_attr_ptr = NULL;
        Tango::Attr *attr_ptr = NULL;

        long x, y;
        vector<Tango::AttrProperty> &def_prop = new_attr.get_user_default_properties();
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
}

Device_5ImplWrap::Device_5ImplWrap(py::object& pyself, DeviceClass *cl, std::string& name)
  : Tango::Device_5Impl(cl, name)
{
    py_self = pyself;
}

Device_5ImplWrap::Device_5ImplWrap(py::object& pyself, DeviceClass *cl, std::string& name, std::string& descr)
  : Tango::Device_5Impl(cl, name, descr)
{
    py_self = pyself;
}

Device_5ImplWrap::Device_5ImplWrap(py::object& pyself, DeviceClass *cl,
                                   std::string& name,
                                   std::string& desc,
                                   Tango::DevState sta,
                                   std::string& status)
  : Tango::Device_5Impl(cl, name, desc, sta, status)
{
    py_self = pyself;
}

Device_5ImplWrap::~Device_5ImplWrap()
{
    delete_device();
}

void Device_5ImplWrap::delete_dev()
{
    // Call here the delete_device method. It is defined in Device_5ImplWrap
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

void Device_5ImplWrap::py_delete_dev()
{
    Device_5ImplWrap::delete_dev();
}

void Device_5ImplWrap::init_device()
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

void Device_5ImplWrap::delete_device()
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

void Device_5ImplWrap::default_delete_device()
{
    this->Tango::Device_5Impl::delete_device();
}

bool Device_5ImplWrap::_is_attribute_polled(const std::string& att_name)
{
    return this->is_attribute_polled(att_name);
}

bool Device_5ImplWrap::_is_command_polled(const std::string& cmd_name)
{
    std::cout << &py_self << std::endl;
    return this->is_command_polled(cmd_name);
}

int Device_5ImplWrap::_get_attribute_poll_period(const std::string& att_name)
{
    return this->get_attribute_poll_period(att_name);
}

int Device_5ImplWrap::_get_command_poll_period(const std::string& cmd_name)
{
    return this->get_command_poll_period(cmd_name);
}

void Device_5ImplWrap::_poll_attribute(const std::string& att_name, int period)
{
    this->poll_attribute(att_name, period);
}

void Device_5ImplWrap::_poll_command(const std::string& cmd_name, int period)
{
    this->poll_command(cmd_name, period);
}

void Device_5ImplWrap::_stop_poll_attribute(const std::string& att_name)
{
    this->stop_poll_attribute(att_name);
}

void Device_5ImplWrap::_stop_poll_command(const std::string& cmd_name)
{
    this->stop_poll_command(cmd_name);
}

void Device_5ImplWrap::always_executed_hook()
{
    std::thread::id thread_id = std::this_thread::get_id();
    std::cerr << "always executed thread id " << thread_id << std::endl;
    AutoPythonGILEnsure __py_lock;
    try {
        if (is_method_callable(py_self, "always_executed_hook")) {
            std::cout << "always_executed_hook is callable" << std::endl;
            py_self.attr("always_executed_hook")();
        }
        else {
            std::cout << "always_executed_hook is NOT callable" << std::endl;
            Tango::Device_5Impl::always_executed_hook();
        }
    }
    catch(py::error_already_set &eas) {
        std::cerr << "3" << std::endl;
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

void Device_5ImplWrap::default_always_executed_hook()
{
    this->Tango::Device_5Impl::always_executed_hook();
}

void Device_5ImplWrap::read_attr_hardware(std::vector<long> &attr_list)
{
    std::thread::id thread_id = std::this_thread::get_id();
    std::cerr << "read_attr_hardware thread id " << thread_id << std::endl;
    AutoPythonGIL __py_lock;
    try {
        if (is_method_callable(py_self, "read_attr_hardware")) {
            py_self.attr("read_attr_hardware")(attr_list);
        }
        else {
            Tango::Device_5Impl::read_attr_hardware(attr_list);
        }
    }
    catch(py::error_already_set &eas) {
        std::cerr << "4" << std::endl;
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

void Device_5ImplWrap::default_read_attr_hardware(vector<long> &attr_list)
{
    this->Tango::Device_5Impl::read_attr_hardware(attr_list);
}

void Device_5ImplWrap::write_attr_hardware(std::vector<long> &attr_list)
{
    std::thread::id thread_id = std::this_thread::get_id();
    std::cerr << "write_attr_hardware thread id " << thread_id << std::endl;
    AutoPythonGIL __py_lock;
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

void Device_5ImplWrap::default_write_attr_hardware(vector<long> &attr_list)
{
    this->Tango::Device_5Impl::write_attr_hardware(attr_list);
}

Tango::DevState Device_5ImplWrap::dev_state()
{
    AutoPythonGILEnsure __py_lock;
    try {
        if (is_method_callable(py_self, "dev_state")) {
            std::cerr << "doing this devstate" << std::endl;
            py::object ret = py_self.attr("dev_state")();
            return ret.cast<Tango::DevState>();
        }
        else {
            std::cerr << "OR doing this devstate" << std::endl;
            return Tango::Device_5Impl::dev_state();
        }
    }
    catch(py::error_already_set &eas) {
        std::cerr << "6" << std::endl;
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

Tango::DevState Device_5ImplWrap::default_dev_state()
{
    return this->Tango::Device_5Impl::dev_state();
}

Tango::ConstDevString Device_5ImplWrap::dev_status()
{
    std::cerr << "Entering 5ImplWrap dev_status" << std::endl;
    AutoPythonGILEnsure __py_lock;
    try {
        if (is_method_callable(py_self, "dev_status")) {
            py::object ret = py_self.attr("dev_status")();
            this->the_status = ret.cast<std::string>();
            std::cerr << this->the_status << std::endl;
            return this->the_status.c_str();
        }
        else {
            std::cerr << "OR OR this dev_status?" << std::endl;
            this->the_status = Tango::Device_5Impl::dev_status();
            return this->the_status.c_str();
        }
    }
    catch(py::error_already_set &eas) {
        std::cerr << "dev_status exception" << std::endl;
        handle_python_exception(eas);
    }
    catch(...) {
        Tango::Except::throw_exception("CppException",
                "An unexpected C++ exception occurred",
                "delete_device");
    }
}

Tango::ConstDevString Device_5ImplWrap::default_dev_status()
{
    this->the_status = this->Tango::Device_5Impl::dev_status();
    return this->the_status.c_str();
}

void Device_5ImplWrap::signal_handler(long signo)
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
        df.errors[nb_err].origin = CORBA::string_dup("Device_5Impl.signal_handler");
        df.errors[nb_err].severity = Tango::ERR;

        Tango::Except::print_exception(df);
    }
}

void Device_5ImplWrap::default_signal_handler(long signo)
{
    this->Tango::Device_5Impl::signal_handler(signo);
}

// Trampoline class define method for each virtual function
class PyDevice_5Impl : public Tango::Device_5Impl {
public:
    void init_device() override {
        PYBIND11_OVERLOAD_PURE(
            void,               // Return type
            Tango::Device_5Impl,  // Parent class
            init_device,        // Name of function in C++ (must match Python name)
        );
    }
    void delete_device() override {
        PYBIND11_OVERLOAD(
            void,               // Return type
            Tango::Device_5Impl,  // Parent class
            delete_device,      // Name of function in C++ (must match Python name)
        );
    }
    void always_executed_hook() override {
        PYBIND11_OVERLOAD(
            void,                  // Return type
            Tango::Device_5Impl,     // Parent class
            always_executed_hook,  // Name of function in C++ (must match Python name)
        );
    }
    void read_attr_hardware(std::vector<long> &attr_list) override {
        PYBIND11_OVERLOAD(
            void,                // Return type
            Tango::Device_5Impl,   // Parent class
            read_attr_hardware,  // Name of function in C++ (must match Python name)
            attr_list
        );
    }
    void write_attr_hardware(std::vector<long> &attr_list) override {
        PYBIND11_OVERLOAD(
            void,                 // Return type
            Tango::Device_5Impl,    // Parent class
            write_attr_hardware,  // Name of function in C++ (must match Python name)
            attr_list
        );
    }
    Tango::DevState dev_state() override {
        PYBIND11_OVERLOAD(
            Tango::DevState,     // Return type
            Tango::Device_5Impl,   // Parent class
            dev_state,           // Name of function in C++ (must match Python name)
        );
    }
    Tango::ConstDevString dev_status() override {
        PYBIND11_OVERLOAD(
            Tango::ConstDevString, // Return type
            Tango::Device_5Impl,     // Parent class
            dev_status,            // Name of function in C++ (must match Python name)
        );
    }
    void signal_handler(long signo) override {
        PYBIND11_OVERLOAD(
            void,                // Return type
            Tango::Device_5Impl, // Parent class
            signal_handler,      // Name of function in C++ (must match Python name)
            signo                // first argument
        );
    }
};

void export_device_impl(py::module &m) {

    py::class_<Tango::Device_5Impl, PyDevice_5Impl>(m, "BaseDevice_5Impl")
    ;
    py::class_<Device_5ImplWrap, Tango::Device_5Impl>(m, "Device_5Impl")
        .def(py::init([](DeviceClass *cppdev, std::string& name, py::object pyself) {
            std::thread::id thread_id = std::this_thread::get_id();
            std::cout << "c++/python Device_5ImplWrap thread id " << thread_id << std::endl;
            Device_5ImplWrap* cpp = new Device_5ImplWrap(pyself, cppdev, name);
            return cpp;
        }))
        .def(py::init([](py::object pyself, DeviceClass *cppdev, std::string& name, std::string& descr) {
            Device_5ImplWrap* cpp = new Device_5ImplWrap(pyself, cppdev, name, descr);
            return cpp;
        }))
       .def(py::init([](py::object pyself, DeviceClass *cppdev, std::string& name,
            std::string& descr, Tango::DevState dstate, std::string& status) {
            Device_5ImplWrap* cpp = new Device_5ImplWrap(pyself, cppdev, name, descr, dstate, status);
            return cpp;
        }))
        .def("set_state", [](Device_5ImplWrap& self, const Tango::DevState &new_state) -> void {
            self.set_state(new_state);
        })
        .def("get_state", [](Device_5ImplWrap& self) -> Tango::DevState {
            return self.get_state();
        })
        .def("get_prev_state", [](Device_5ImplWrap& self) -> Tango::DevState {
            return self.get_prev_state();
        })
        .def("get_name", [](Device_5ImplWrap& self) -> std::string {
            return self.get_name();
        })
        .def("get_device_attr", [](Device_5ImplWrap& self) {
            return self.get_device_attr();
        })
        .def("register_signal", [](Device_5ImplWrap& self, long signo, bool own_handler) -> void {
            self.register_signal(signo, own_handler);
        }, py::arg("signo"), py::arg("own_handler")=false)
        .def("unregister_signal", [](Device_5ImplWrap& self, long signo) -> void {
            self.unregister_signal(signo);
        })
        .def("get_status", [](Device_5ImplWrap& self) -> std::string {
            return self.get_status();
        })
        .def("set_status", [](Device_5ImplWrap& self, const std::string& new_status) -> void {
            self.set_status(new_status);
        })
        .def("append_status", [](Device_5ImplWrap& self, const std::string& stat, bool new_line) -> void {
            self.append_status(stat, new_line);
        }, py::arg("stat"), py::arg("new_line")=false)
       .def("dev_state", [](Device_5ImplWrap& self) -> Tango::DevState {
           return self.default_dev_state();
        })
        .def("dev_status", [](Device_5ImplWrap& self) -> Tango::ConstDevString {
            return self.default_dev_status();
        })
        .def("get_attribute_config", [](Device_5ImplWrap& self, py::object& py_attr_name_seq) {
            return PyDeviceImpl::get_attribute_config(self, py_attr_name_seq);
        })
        .def("set_change_event", [](Device_5ImplWrap& self, std::string& attr_name, bool implemented, bool detect ) -> void {
            self.set_change_event(attr_name, implemented, detect);
        }, py::arg("attr_name"), py::arg("implemented"), py::arg("detect")=true)
        .def("set_archive_event", [](Device_5ImplWrap& self, std::string& attr_name, bool implemented, bool detect) -> void {
            self.set_archive_event(attr_name, implemented, detect);
        }, py::arg("attr_name"), py::arg("implemented"), py::arg("detect")=true)
        .def("_add_attribute", [](Device_5ImplWrap& self, const Tango::Attr &new_attr,
                       const std::string& read_meth_name,
                       const std::string& write_meth_name,
                       const std::string& is_allowed_meth_name) -> void {
            PyDeviceImpl::add_attribute(self, new_attr, read_meth_name, write_meth_name, is_allowed_meth_name);
        })
        .def("_remove_attribute", [](Device_5ImplWrap& self, std::string& att_name, bool freeit, bool clean_db) -> void {
            self.remove_attribute(att_name, freeit, clean_db);
        }, py::arg("att_name"), py::arg("freeit")=false, py::arg("clean_db")=true)

        //@TODO .def("get_device_class")
        //@TODO .def("get_db_device")

        .def("is_attribute_polled", [](Device_5ImplWrap& self, const std::string& att_name) -> bool {
            return self._is_attribute_polled(att_name);
        })
        .def("is_command_polled", [](Device_5ImplWrap& self, const std::string& cmd_name) -> bool {
            std::cout << "is the command " << cmd_name << " polled" << std::endl;
            return self._is_command_polled(cmd_name);
        })
        .def("get_attribute_poll_period", [](Device_5ImplWrap& self, const std::string& att_name) -> int {
            return self._get_attribute_poll_period(att_name);
        })
        .def("get_command_poll_period", [](Device_5ImplWrap& self, const std::string& cmd_name) -> int {
            return self._get_command_poll_period(cmd_name);
        })
        .def("poll_attribute", [](Device_5ImplWrap& self, const std::string& att_name, int period) -> void {
            self._poll_attribute(att_name, period);
        })
        .def("poll_command", [](Device_5ImplWrap& self, const std::string& cmd_name, int period) -> void {
            self._poll_command(cmd_name, period);
        })
        .def("stop_poll_attribute", [](Device_5ImplWrap& self, const std::string& att_name) -> void {
            self._stop_poll_attribute(att_name);
        })
        .def("stop_poll_command", [](Device_5ImplWrap& self, const std::string& cmd_name) -> void {
            self._stop_poll_command(cmd_name);
        })
        .def("get_exported_flag", [](Device_5ImplWrap& self) -> bool {
            return self.get_exported_flag();
        })
        .def("get_poll_ring_depth", [](Device_5ImplWrap& self) -> long {
            return self.get_poll_ring_depth();
        })
        .def("get_poll_old_factor", [](Device_5ImplWrap& self) -> long {
            return self.get_poll_old_factor();
        })
        .def("is_polled", [](Device_5ImplWrap& self) -> bool {
            return self.is_polled();
        })
        .def("get_polled_cmd", [](Device_5ImplWrap& self) -> std::vector<std::string> {
            return self.get_polled_cmd();
        })
        .def("get_polled_attr", [](Device_5ImplWrap& self) -> std::vector<std::string> {
            return self.get_polled_attr();
        })
        .def("get_non_auto_polled_cmd", [](Device_5ImplWrap& self) -> std::vector<std::string> {
            return self.get_non_auto_polled_cmd();
        })
        .def("get_non_auto_polled_attr", [](Device_5ImplWrap& self) -> std::vector<std::string> {
            return self.get_non_auto_polled_attr();
        })

        //@TODO .def("get_poll_obj_list", &PyDeviceImpl::get_poll_obj_list)

        .def("stop_polling", [](Device_5ImplWrap& self, bool stop) -> void {
            self.stop_polling(stop);
        }, py::arg("stop")=true)
        .def("check_command_exists", [](Device_5ImplWrap& self, const std::string& cmd) -> void {
            self.check_command_exists(cmd);
        })

        //@TODO .def("get_command", &PyDeviceImpl::get_command)

        .def("get_dev_idl_version", [](Device_5ImplWrap& self) {
            self.get_dev_idl_version();
        })
        .def("get_cmd_poll_ring_depth", [](Device_5ImplWrap& self, std::string& cmd) -> long {
            return self.get_cmd_poll_ring_depth(cmd);
        })
        .def("get_attr_poll_ring_depth", [](Device_5ImplWrap& self, std::string& attr) -> long {
            return self.get_attr_poll_ring_depth(attr);
        })
        .def("is_device_locked", [](Device_5ImplWrap& self) -> bool {
            return self.is_device_locked();
        })
        .def("init_logger", [](Device_5ImplWrap& self) -> void {
            self.init_logger();
        })
        .def("start_logging", [](Device_5ImplWrap& self) -> void {
            self.start_logging();
        })
        .def("stop_logging", [](Device_5ImplWrap& self) -> void {
            self.stop_logging();
        })
        .def("set_exported_flag", [](Device_5ImplWrap& self, bool exp) -> void {
            self.set_exported_flag(exp);
        })
        .def("set_poll_ring_depth", [](Device_5ImplWrap& self, long depth) -> void {
            self.set_poll_ring_depth(depth);
        })
        .def("push_change_event", [](Device_5ImplWrap& self, const std::string& attr_name) -> void {
            PyDeviceImpl::push_change_event(self, attr_name);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, py::object& data) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, const std::string& str_data, const std::string& data) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, str_data, data);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, const std::string& str_data, py::object& data) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, str_data, data);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, long x) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, x);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, long x, long y) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, x, y);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, quality);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, const std::string& str_data, const std::string& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, str_data, data, t, quality);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, const std::string& str_data, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, str_data, data, t, quality);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality, long x) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, t, quality, x);
        })
        .def("push_change_event",[](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality, long x, long y) -> void {
            PyDeviceImpl::push_change_event(self, attr_name, data, t, quality, x, y);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& data) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, const std::string& str_data, const std::string& data) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, str_data, data);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, const std::string& str_data, py::object& data) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, str_data, data);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, long x) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, x);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, long x, long y) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, x, y);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, t, quality);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, const std::string& str_data, const std::string& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, str_data, data, t, quality);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, const std::string& str_data, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, str_data, data, t, quality);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality, long x) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, t, quality, x);
        })
        .def("push_archive_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& data, double t, Tango::AttrQuality quality, long x, long y) -> void {
            PyDeviceImpl::push_archive_event(self, attr_name, data, t, quality, x, y);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, py::object& data) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, const std::string& str_data, const std::string& data) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, str_data, data);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, const std::string& str_data, py::object& data) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, str_data, data);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, py::object& data, long x) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, x);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, py::object& data, long x, long y) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, x, y);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, t, quality);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, const std::string& str_data, const std::string& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, str_data, data, t, quality);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, const std::string& str_data, py::object& data, double t, Tango::AttrQuality quality) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, str_data, data, t, quality);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, py::object& data, double t, Tango::AttrQuality quality, long x) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, t, quality, x);
        })
        .def("push_event", [](Device_5ImplWrap& self, const std::string& attr_name, py::object& filt_names, py::object& filt_vals, py::object& data, double t, Tango::AttrQuality quality, long x, long y) -> void {
            PyDeviceImpl::push_event(self, attr_name, filt_names, filt_vals, data, t, quality, x, y);
        })
        .def("push_data_ready_event", [](Device_5ImplWrap& self, const std::string& attr_name, long ctr) -> void {
            PyDeviceImpl::push_data_ready_event(self, attr_name, ctr);
        })

//         @TODO
//        .def("push_att_conf_event", [](Device_5ImplWrap& self) -> void {
//            self.push_att_conf_event();
//            should be this?????????????
//            self.push_att_conf_event(Attribute *);
//        })

        .def("push_pipe_event", [](Device_5ImplWrap& self, const std::string& pipe_name, py::object& pipe_data) -> void {
            PyDeviceImpl::push_pipe_event(self, pipe_name, pipe_data);
        })
        .def("get_logger", [](Device_5ImplWrap& self) {
            return self.get_logger();
        })
        .def("__debug_stream", [](Device_5ImplWrap& self, const std::string& msg) {
            if (self.get_logger()->is_debug_enabled()) {
                self.get_logger()->debug_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("__info_stream", [](Device_5ImplWrap& self, const std::string& msg) -> void {
            if (self.get_logger()->is_info_enabled()) {
                self.get_logger()->info_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("__warn_stream", [](Device_5ImplWrap& self, const std::string& msg) -> void {
            if (self.get_logger()->is_warn_enabled()) {
                self.get_logger()->warn_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("__error_stream", [](Device_5ImplWrap& self, const std::string& msg) -> void {
            if (self.get_logger()->is_error_enabled()) {
                self.get_logger()->error_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("__fatal_stream", [](Device_5ImplWrap& self, const std::string& msg) -> void {
            if (self.get_logger()->is_fatal_enabled()) {
                self.get_logger()->fatal_stream() << log4tango::LogInitiator::_begin_log << msg;
            }
        })
        .def("get_min_poll_period", [](Device_5ImplWrap& self) {
            self.get_min_poll_period();
        })
        .def("get_cmd_min_poll_period", [](Device_5ImplWrap& self) {
            self.get_cmd_min_poll_period();
        })
        .def("get_attr_min_poll_period", [](Device_5ImplWrap& self) {
            self.get_attr_min_poll_period();
        })
        .def("is_there_subscriber", [](Device_5ImplWrap& self, const std::string& att_name, Tango::EventType event_type) -> bool{
            self.is_there_subscriber(att_name, event_type);
        })
        .def("init_device", [](Device_5ImplWrap& self) {
            self.init_device();
        })
        .def("delete_device", [](Device_5ImplWrap& self) {
            self.default_delete_device();
        })
        .def("always_executed_hook", [](Device_5ImplWrap& self) {
            std::cout << "does it do this executed hook code" << std::endl;
            self.default_always_executed_hook();
        })
        .def("read_attr_hardware", [](Device_5ImplWrap& self, std::vector<long> &attr_list) {
            self.default_read_attr_hardware(attr_list);
        })
        .def("write_attr_hardware", [](Device_5ImplWrap& self, std::vector<long> &attr_list) {
            self.default_write_attr_hardware(attr_list);
        })
        .def("signal_handler", [](Device_5ImplWrap& self, long signo) {
            self.default_signal_handler(signo);
        })
    ;
}
