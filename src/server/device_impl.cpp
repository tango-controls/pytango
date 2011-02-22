#include <boost/python.hpp>
#include <boost/python/return_value_policy.hpp>
#include <string>
#include <tango.h>

#include "defs.h"
#include "pytgutils.h"
#include "exception.h"
#include "server/device_impl.h"
#include "server/attr.h"
#include "server/attribute.h"
#include "to_py.h"

extern const char *param_must_be_seq;

using namespace boost::python;

#define __AUX_DECL_CALL_DEVICE_METHOD \
    AutoPythonGIL __py_lock;

#define __AUX_CATCH_PY_EXCEPTION \
    catch(boost::python::error_already_set &eas) \
    { handle_python_exception(eas); }

#define __AUX_CATCH_EXCEPTION(name) \
    catch(...)                                                                \
    { Tango::Except::throw_exception("CppException",                          \
                                     "An unexpected C++ exception occurred",  \
                                     #name );                                 \
    }

#define CALL_DEVICE_METHOD(cls, name) \
    __AUX_DECL_CALL_DEVICE_METHOD \
    try {                                                                     \
        if (override name = this->get_override( #name) ) { name (); }         \
        else { cls :: name (); }                                              \
    }                                                                         \
    __AUX_CATCH_PY_EXCEPTION \
    __AUX_CATCH_EXCEPTION(name)

#define CALL_DEVICE_METHOD_VARGS(cls, name, ...) \
    __AUX_DECL_CALL_DEVICE_METHOD \
    try {                                                                     \
        if (override name = this->get_override( #name) )                      \
        { name (__VA_ARGS__); }                                               \
        else { cls :: name (__VA_ARGS__); }                                   \
    }                                                                         \
    __AUX_CATCH_PY_EXCEPTION \
    __AUX_CATCH_EXCEPTION(name)

#define CALL_DEVICE_METHOD_RET(cls, name) \
    __AUX_DECL_CALL_DEVICE_METHOD \
    try {                                                                     \
        if (override name = this->get_override( #name) ) { return name (); }  \
        else { return cls :: name (); }                                       \
    }                                                                         \
    __AUX_CATCH_PY_EXCEPTION \
    __AUX_CATCH_EXCEPTION(name)

#define CALL_DEVICE_METHOD_VARGS_RET(cls, name, ...) \
    __AUX_DECL_CALL_DEVICE_METHOD \
    try {                                                                     \
        if (override name = this->get_override( #name) )                      \
        { return name (__VA_ARGS__); }                                        \
        else { return cls :: name (__VA_ARGS__); }                            \
    }                                                                         \
    __AUX_CATCH_PY_EXCEPTION \
    __AUX_CATCH_EXCEPTION(name)

#define SAFE_PUSH(dev, attr, attr_name) \
    char *__att_name_ptr = PyString_AsString(attr_name.ptr()); \
    AutoPythonAllowThreads python_guard_ptr; \
    Tango::AutoTangoMonitor tango_guard(&dev); \
    Tango::Attribute & attr = dev.get_device_attr()->get_attr_by_name(__att_name_ptr); \
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

#define AUX_SAFE_PUSH_EVENT(dev, attr_name, filt_names, filt_vals) \
    StdStringVector filt_names_; \
    StdDoubleVector filt_vals_; \
    from_sequence<StdStringVector>::convert(filt_names, filt_names_); \
    from_sequence<StdDoubleVector>::convert(filt_vals, filt_vals_); \
    SAFE_PUSH(dev, attr, attr_name)

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
    inline PyObject* get_polled_cmd(Tango::DeviceImpl &self)
    {
        return to_list<std::vector<std::string> >::convert(self.get_polled_cmd());
    }

    inline PyObject* get_polled_attr(Tango::DeviceImpl &self)
    {
        return to_list<std::vector<std::string> >::convert(self.get_polled_attr());
    }
    
    inline PyObject* get_non_auto_polled_cmd(Tango::DeviceImpl &self)
    {
        return to_list<std::vector<std::string> >::convert(self.get_non_auto_polled_cmd());
    }

    inline PyObject* get_non_auto_polled_attr(Tango::DeviceImpl &self)
    {
        return to_list<std::vector<std::string> >::convert(self.get_non_auto_polled_attr());
    }
    
    /* **********************************
     * change event USING set_value
     * **********************************/
    inline void push_change_event(Tango::DeviceImpl &self, str &name)
    {
        str name_lower = name.lower();
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

    inline void push_change_event(Tango::DeviceImpl &self, str &name, object &data)
    {
        boost::python::extract<Tango::DevFailed> except_convert(data);
        if (except_convert.check()) {
            SAFE_PUSH(self, attr, name);
            attr.fire_change_event(
                           const_cast<Tango::DevFailed*>( &except_convert() ));
            return;
        }
        SAFE_PUSH_CHANGE_EVENT(self, name, data);
    }

    // Special variantion for encoded data type
    inline void push_change_event(Tango::DeviceImpl &self, str &name, str &str_data,
                                  str &data)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl &self, str &name, object &data,
                                  long x)
    {
        SAFE_PUSH_CHANGE_EVENT_VARGS(self, name, data, x);
    }

    inline void push_change_event(Tango::DeviceImpl &self, str &name, object &data,
                                  long x, long y)
    {
        SAFE_PUSH_CHANGE_EVENT_VARGS(self, name, data, x, y);
    }

    /* **********************************
     * change event USING set_value_date_quality
     * **********************************/

    inline void push_change_event(Tango::DeviceImpl &self, str &name, object &data,
                                  double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH_CHANGE_EVENT_DATE_QUALITY(self, name, data, t, quality)
    }

    // Special variantion for encoded data type
    inline void push_change_event(Tango::DeviceImpl &self, str &name, str &str_data,
                                  str &data, double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_change_event();
    }

    inline void push_change_event(Tango::DeviceImpl &self, str &name, object &data,
                                  double t, Tango::AttrQuality quality, long x)
    {
        SAFE_PUSH_CHANGE_EVENT_DATE_QUALITY_VARGS(self, name, data, t, quality, x)
    }

    inline void push_change_event(Tango::DeviceImpl &self, str &name, object &data,
                                  double t, Tango::AttrQuality quality, long x, long y)
    {
        SAFE_PUSH_CHANGE_EVENT_DATE_QUALITY_VARGS(self, name, data, t, quality, x, y)
    }

    /* **********************************
     * archive event USING set_value
     * **********************************/
    inline void push_archive_event(Tango::DeviceImpl &self, str &name)
    {
        SAFE_PUSH(self, attr, name)
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl &self, str &name, object &data)
    {
        boost::python::extract<Tango::DevFailed> except_convert(data);
        if (except_convert.check()) {
            SAFE_PUSH(self, attr, name);
            attr.fire_archive_event(
                           const_cast<Tango::DevFailed*>( &except_convert() ));
            return;
        }
        SAFE_PUSH_ARCHIVE_EVENT(self, name, data);
    }

    // Special variantion for encoded data type
    inline void push_archive_event(Tango::DeviceImpl &self, str &name, str &str_data,
                                   str &data)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl &self, str &name, object &data,
                           long x)
    {
        SAFE_PUSH_ARCHIVE_EVENT_VARGS(self, name, data, x);
    }

    inline void push_archive_event(Tango::DeviceImpl &self, str &name, object &data,
                           long x, long y)
    {
        SAFE_PUSH_ARCHIVE_EVENT_VARGS(self, name, data, x, y);
    }

    /* **********************************
     * archive event USING set_value_date_quality
     * **********************************/

    inline void push_archive_event(Tango::DeviceImpl &self, str &name, object &data,
                                  double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH_ARCHIVE_EVENT_DATE_QUALITY(self, name, data, t, quality)
    }

    // Special variantion for encoded data type
    inline void push_archive_event(Tango::DeviceImpl &self, str &name, str &str_data,
                                   str &data, double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH(self, attr, name)
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_archive_event();
    }

    inline void push_archive_event(Tango::DeviceImpl &self, str &name, object &data,
                                  double t, Tango::AttrQuality quality, long x)
    {
        SAFE_PUSH_ARCHIVE_EVENT_DATE_QUALITY_VARGS(self, name, data, t, quality, x)
    }

    inline void push_archive_event(Tango::DeviceImpl &self, str &name, object &data,
                                  double t, Tango::AttrQuality quality, long x, long y)
    {
        SAFE_PUSH_ARCHIVE_EVENT_DATE_QUALITY_VARGS(self, name, data, t, quality, x, y)
    }

    /* **********************************
     * user event USING set_value
     * **********************************/
    inline void push_event(Tango::DeviceImpl &self, str &name,
                           object &filt_names, object &filt_vals)
    {
        AUX_SAFE_PUSH_EVENT(self, name, filt_names, filt_vals)
        attr.fire_event(filt_names_, filt_vals_);
    }

    inline void push_event(Tango::DeviceImpl &self, str &name,
                           object &filt_names, object &filt_vals, object &data)
    {
        SAFE_PUSH_EVENT(self, name, filt_names, filt_vals, data)
    }

    // Special variantion for encoded data type
    inline void push_event(Tango::DeviceImpl &self, str &name,
                           object &filt_names, object &filt_vals,
                           str &str_data, str &data)
    {
        AUX_SAFE_PUSH_EVENT(self, name, filt_names, filt_vals)
        PyAttribute::set_value(attr, str_data, data);
        attr.fire_event(filt_names_, filt_vals_);
    }

    inline void push_event(Tango::DeviceImpl &self, str &name,
                           object &filt_names, object &filt_vals, object &data,
                           long x)
    {
        SAFE_PUSH_EVENT_VARGS(self, name, filt_names, filt_vals, data, x)
    }

    inline void push_event(Tango::DeviceImpl &self, str &name,
                           object &filt_names, object &filt_vals, object &data,
                           long x, long y)
    {
        SAFE_PUSH_EVENT_VARGS(self, name, filt_names, filt_vals, data, x, y)
    }

    /* ***************************************
     * user event USING set_value_date_quality
     * **************************************/

    inline void push_event(Tango::DeviceImpl &self, str &name,
                           object &filt_names, object &filt_vals, object &data,
                           double t, Tango::AttrQuality quality)
    {
        SAFE_PUSH_EVENT_DATE_QUALITY(self, name, filt_names, filt_vals, data, t, quality)
    }

    // Special variantion for encoded data type
    inline void push_event(Tango::DeviceImpl &self, str &name,
                           object &filt_names, object &filt_vals,
                           str &str_data, str &data,
                           double t, Tango::AttrQuality quality)
    {
        AUX_SAFE_PUSH_EVENT(self, name, filt_names, filt_vals)
        PyAttribute::set_value_date_quality(attr, str_data, data, t, quality);
        attr.fire_event(filt_names_, filt_vals_);
    }

    inline void push_event(Tango::DeviceImpl &self, str &name,
                           object &filt_names, object &filt_vals, object &data,
                           double t, Tango::AttrQuality quality, long x)
    {
        SAFE_PUSH_EVENT_DATE_QUALITY_VARGS(self, name, filt_names, filt_vals, data, t, quality, x)
    }

    inline void push_event(Tango::DeviceImpl &self, str &name,
                           object &filt_names, object &filt_vals, object &data,
                           double t, Tango::AttrQuality quality, long x, long y)
    {
        SAFE_PUSH_EVENT_DATE_QUALITY_VARGS(self, name, filt_names, filt_vals, data, t, quality, x, y)
    }

    void check_attribute_method_defined(PyObject *self,
                                        const std::string &attr_name,
                                        const std::string &method_name)
    {
        bool exists, is_method;

        is_method_defined(self, method_name, exists, is_method);

        if (!exists)
        {
            TangoSys_OMemStream o;
            o << "Wrong definition of attribute " << attr_name
              << "\nThe attribute method " << method_name
              << " does not exist in your class!" << ends;

            Tango::Except::throw_exception(
                    (const char *)"PyDs_WrongCommandDefinition",
                    o.str(),
                    (const char *)"check_attribute_method_defined");
        }

        if(!is_method)
        {
            TangoSys_OMemStream o;
            o << "Wrong definition of attribute " << attr_name
              << "\nThe object " << method_name
              << " exists in your class but is not a Python method" << ends;

            Tango::Except::throw_exception(
                    (const char *)"PyDs_WrongCommandDefinition",
                    o.str(),
                    (const char *)"check_attribute_method_defined");
        }
    }

    void add_attribute(Tango::DeviceImpl &self, const Tango::Attr &c_new_attr)
    {
        Tango::Attr &new_attr = const_cast<Tango::Attr &>(c_new_attr);

        std::string
            attr_name = new_attr.get_name(),
            is_allowed_method = "is_" + attr_name + "_allowed",
            read_name_met = "read_" + attr_name,
            write_name_met = "write_" + attr_name;

        Tango::AttrWriteType attr_write = new_attr.get_writable();

        //
        // Create the attribute object according to attribute format
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
                TangoSys_OMemStream o;
                o << "Attribute " << attr_name << " has an unexpected data format\n"
                  << "Please report this bug to the PyTango development team"
                  << ends;
                Tango::Except::throw_exception(
                        (const char *)"PyDs_UnexpectedAttributeFormat",
                        o.str(),
                        (const char *)"cpp_add_attribute");
                break;
        }

        py_attr_ptr->set_read_name(read_name_met);
        py_attr_ptr->set_write_name(write_name_met);
        py_attr_ptr->set_allowed_name(is_allowed_method);

        //
        // Install attribute in Tango.
        //
        self.add_attribute(attr_ptr);
    }

    void remove_attribute(Tango::DeviceImpl &self, const char *att_name)
    {
        string str(att_name);
        self.remove_attribute(str, false);
    }

    inline void debug(Tango::DeviceImpl &self, const string &msg)
    {
        self.get_logger()->debug(msg);
    }

    inline void info(Tango::DeviceImpl &self, const string &msg)
    {
        self.get_logger()->info(msg);
    }

    inline void warn(Tango::DeviceImpl &self, const string &msg)
    {
        self.get_logger()->warn(msg);
    }

    inline void error(Tango::DeviceImpl &self, const string &msg)
    {
        self.get_logger()->error(msg);
    }

    inline void fatal(Tango::DeviceImpl &self, const string &msg)
    {
        self.get_logger()->fatal(msg);
    }
    
    PyObject* get_attribute_config(Tango::DeviceImpl &self, object &py_attr_name_seq)
    {
        Tango::DevVarStringArray par;
        convert2array(py_attr_name_seq, par);
        
        Tango::AttributeConfigList *attr_conf_list_ptr = 
            self.get_attribute_config(par);
        
        boost::python::list ret = to_py(*attr_conf_list_ptr);
        delete attr_conf_list_ptr;
        
        return boost::python::incref(ret.ptr());
    }
    
    void set_attribute_config(Tango::DeviceImpl &self, object &py_attr_conf_list)
    {
        Tango::AttributeConfigList attr_conf_list;
        from_py_object(py_attr_conf_list, attr_conf_list);
        self.set_attribute_config(attr_conf_list);
    }
}

DeviceImplWrap::DeviceImplWrap(PyObject *self, CppDeviceClass *cl,
                               std::string &st)
    :Tango::DeviceImpl(cl,st), m_self(self)
{
    Py_INCREF(m_self);
}

DeviceImplWrap::DeviceImplWrap(PyObject *self, CppDeviceClass *cl,
                               const char *name,
                               const char *desc /* = "A Tango device" */,
                               Tango::DevState sta /* = Tango::UNKNOWN */,
                               const char *status /* = StatusNotSet */)
    :Tango::DeviceImpl(cl, name, desc, sta, status), m_self(self)
{
    Py_INCREF(m_self);
}

void DeviceImplWrap::init_device()
{
    this->get_override("init_device")();
}

Device_2ImplWrap::Device_2ImplWrap(PyObject *self, CppDeviceClass *cl,
                                   std::string &st)
    :Tango::Device_2Impl(cl,st),m_self(self)
{
    Py_INCREF(m_self);
}

Device_2ImplWrap::Device_2ImplWrap(PyObject *self, CppDeviceClass *cl,
                                   const char *name,
                                   const char *desc /* = "A Tango device" */,
                                   Tango::DevState sta /* = Tango::UNKNOWN */,
                                   const char *status /* = StatusNotSet */)
    :Tango::Device_2Impl(cl, name, desc, sta, status), m_self(self)
{
    Py_INCREF(m_self);
}

void Device_2ImplWrap::init_device()
{
    this->get_override("init_device")();
}

PyDeviceImplBase::PyDeviceImplBase(PyObject *self):the_self(self)
{
    Py_INCREF(the_self);
}

namespace PyDevice_2Impl
{
    PyObject* get_attribute_config_2(Tango::Device_2Impl &self, object &attr_name_seq)
    {
        Tango::DevVarStringArray par;
        convert2array(attr_name_seq, par);
        
        Tango::AttributeConfigList_2 *attr_conf_list_ptr = 
            self.get_attribute_config_2(par);
        
        boost::python::list ret = to_py(*attr_conf_list_ptr);
        delete attr_conf_list_ptr;
        
        return boost::python::incref(ret.ptr());
    }

    /* Postponed: Tango (7.1.1) has no set_attribute_config_2 !!!
    void set_attribute_config_2(Tango::Device_2Impl &self, object &py_attr_conf_list)
    {
        Tango::AttributeConfigList_2 attr_conf_list;
        from_py_object(py_attr_conf_list, attr_conf_list);
        self.set_attribute_config_2(attr_conf_list);
    }
    */
}

PyDeviceImplBase::~PyDeviceImplBase()
{}

void PyDeviceImplBase::py_delete_dev()
{
    try
    {
        Py_DECREF(the_self);
    }
    catch(error_already_set &eas)
    {
        handle_python_exception(eas);
    }
}

Device_3ImplWrap::Device_3ImplWrap(PyObject *self, CppDeviceClass *cl,
                                   std::string &st)
    :Tango::Device_3Impl(cl,st),
    PyDeviceImplBase(self)
{
    _init();
}

Device_3ImplWrap::Device_3ImplWrap(PyObject *self, CppDeviceClass *cl,
                                   const char *name,
                                   const char *desc /* = "A Tango device" */,
                                   Tango::DevState sta /* = Tango::UNKNOWN */,
                                   const char *status /* = StatusNotSet */)
    :Tango::Device_3Impl(cl, name, desc, sta, status),
    PyDeviceImplBase(self)
{
    _init();
}

void Device_3ImplWrap::_init()
{
    // Make sure the wrapper contains a valid pointer to the self
    // I found out this is needed by inspecting the boost wrapper_base.hpp code
    initialize_wrapper(the_self, this);

    // Tell Tango that this is a Python device.
    // Humm, we should try to avoid this in the future
    this->set_py_device(true);

    Tango::Device_3ImplExt *tmp_ptr = ext_3;
    Py_Device_3ImplExt *new_ext = new Py_Device_3ImplExt(this);
    ext_3 = new_ext;
    delete tmp_ptr;
}

void Device_3ImplWrap::init_device()
{
    AutoPythonGIL __py_lock;
    try
    {
        this->get_override("init_device")();
    }
    catch(boost::python::error_already_set &eas)
    {
        handle_python_exception(eas);
    }
}

void Device_3ImplWrap::delete_device()
{
    CALL_DEVICE_METHOD(Device_3Impl, delete_device)
}

void Device_3ImplWrap::default_delete_device()
{
    this->Tango::Device_3Impl::delete_device();
}

void Device_3ImplWrap::delete_dev()
{
    // Call here the delete_device method. It is defined in Device_3ImplWrap
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

void Device_3ImplWrap::py_delete_dev()
{
    Device_3ImplWrap::delete_dev();
    PyDeviceImplBase::py_delete_dev();
}

void Device_3ImplWrap::always_executed_hook()
{
    CALL_DEVICE_METHOD(Device_3Impl, always_executed_hook)
}

void Device_3ImplWrap::default_always_executed_hook()
{
    this->Tango::Device_3Impl::always_executed_hook();
}

void Device_3ImplWrap::read_attr_hardware(vector<long> &attr_list)
{
    CALL_DEVICE_METHOD_VARGS(Device_3Impl, read_attr_hardware, attr_list)
}

void Device_3ImplWrap::default_read_attr_hardware(vector<long> &attr_list)
{
    this->Tango::Device_3Impl::read_attr_hardware(attr_list);
}

void Device_3ImplWrap::write_attr_hardware(vector<long> &attr_list)
{
    CALL_DEVICE_METHOD_VARGS(Device_3Impl, write_attr_hardware, attr_list)
}

void Device_3ImplWrap::default_write_attr_hardware(vector<long> &attr_list)
{
    this->Tango::Device_3Impl::write_attr_hardware(attr_list);
}

Tango::DevState Device_3ImplWrap::dev_state()
{
    CALL_DEVICE_METHOD_RET(Device_3Impl, dev_state)
    // Keep the compiler quiet
    return Tango::UNKNOWN;
}

Tango::DevState Device_3ImplWrap::default_dev_state()
{
    return this->Tango::Device_3Impl::dev_state();
}

Tango::ConstDevString Device_3ImplWrap::dev_status()
{
    CALL_DEVICE_METHOD_RET(Device_3Impl, dev_status)
    // Keep the compiler quiet
    return "Impossible state";
}

Tango::ConstDevString Device_3ImplWrap::default_dev_status()
{
    return this->Tango::Device_3Impl::dev_status();
}

void Device_3ImplWrap::signal_handler(long signo)
{
    CALL_DEVICE_METHOD_VARGS(Device_3Impl, signal_handler, signo)
}

void Device_3ImplWrap::default_signal_handler(long signo)
{
    this->Tango::Device_3Impl::signal_handler(signo);
}

namespace PyDevice_3Impl
{
    PyObject* get_attribute_config_3(Tango::Device_3Impl &self, object &attr_name_seq)
    {
        Tango::DevVarStringArray par;
        convert2array(attr_name_seq, par);
        
        Tango::AttributeConfigList_3 *attr_conf_list_ptr = 
            self.get_attribute_config_3(par);
        
        boost::python::list ret = to_py(*attr_conf_list_ptr);
        delete attr_conf_list_ptr;
        
        return boost::python::incref(ret.ptr());
    }
    
    void set_attribute_config_3(Tango::Device_3Impl &self, object &py_attr_conf_list)
    {
        Tango::AttributeConfigList_3 attr_conf_list;
        from_py_object(py_attr_conf_list, attr_conf_list);
        self.set_attribute_config_3(attr_conf_list);
    }

}

Device_4ImplWrap::Device_4ImplWrap(PyObject *self, CppDeviceClass *cl,
                                   std::string &st)
    :Tango::Device_4Impl(cl,st),
    PyDeviceImplBase(self)
{
    _init();
}

Device_4ImplWrap::Device_4ImplWrap(PyObject *self, CppDeviceClass *cl,
                                   const char *name,
                                   const char *desc /* = "A Tango device" */,
                                   Tango::DevState sta /* = Tango::UNKNOWN */,
                                   const char *status /* = StatusNotSet */)
    :Tango::Device_4Impl(cl, name, desc, sta, status),
    PyDeviceImplBase(self)
{
    _init();
}

void Device_4ImplWrap::_init()
{
    // Make sure the wrapper contains a valid pointer to the self
    // I found out this is needed by inspecting the boost wrapper_base.hpp code
    initialize_wrapper(the_self, this);

    // Tell Tango that this is a Python device.
    // Humm, we should try to avoid this in the future
    this->set_py_device(true);

    Tango::Device_3ImplExt *tmp_ptr = ext_3;
    Py_Device_3ImplExt *new_ext = new Py_Device_3ImplExt(this);
    ext_3 = new_ext;
    delete tmp_ptr;
}

void Device_4ImplWrap::init_device()
{
    AutoPythonGIL __py_lock;
    try
    {
        this->get_override("init_device")();
    }
    catch(boost::python::error_already_set &eas)
    {
        handle_python_exception(eas);
    }
}

void Device_4ImplWrap::delete_device()
{
    CALL_DEVICE_METHOD(Device_4Impl, delete_device)
}

void Device_4ImplWrap::default_delete_device()
{
    this->Tango::Device_4Impl::delete_device();
}

void Device_4ImplWrap::delete_dev()
{
    // Call here the delete_device method. It is defined in Device_4ImplWrap
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

void Device_4ImplWrap::py_delete_dev()
{
    Device_4ImplWrap::delete_dev();
    PyDeviceImplBase::py_delete_dev();
}

void Device_4ImplWrap::always_executed_hook()
{
    CALL_DEVICE_METHOD(Device_4Impl, always_executed_hook)
}

void Device_4ImplWrap::default_always_executed_hook()
{
    this->Tango::Device_4Impl::always_executed_hook();
}

void Device_4ImplWrap::read_attr_hardware(vector<long> &attr_list)
{
    CALL_DEVICE_METHOD_VARGS(Device_4Impl, read_attr_hardware, attr_list)
}

void Device_4ImplWrap::default_read_attr_hardware(vector<long> &attr_list)
{
    this->Tango::Device_4Impl::read_attr_hardware(attr_list);
}

void Device_4ImplWrap::write_attr_hardware(vector<long> &attr_list)
{
    CALL_DEVICE_METHOD_VARGS(Device_4Impl, write_attr_hardware, attr_list)
}

void Device_4ImplWrap::default_write_attr_hardware(vector<long> &attr_list)
{
    this->Tango::Device_4Impl::write_attr_hardware(attr_list);
}

Tango::DevState Device_4ImplWrap::dev_state()
{
    CALL_DEVICE_METHOD_RET(Device_4Impl, dev_state)
    // Keep the compiler quiet
    return Tango::UNKNOWN;
}

Tango::DevState Device_4ImplWrap::default_dev_state()
{
    return this->Tango::Device_4Impl::dev_state();
}

Tango::ConstDevString Device_4ImplWrap::dev_status()
{
    CALL_DEVICE_METHOD_RET(Device_4Impl, dev_status)
    // Keep the compiler quiet
    return "Impossible state";
}

Tango::ConstDevString Device_4ImplWrap::default_dev_status()
{
    return this->Tango::Device_4Impl::dev_status();
}

void Device_4ImplWrap::signal_handler(long signo)
{
    try
    {
        CALL_DEVICE_METHOD_VARGS(Device_4Impl, signal_handler, signo)
    }
    catch(Tango::DevFailed &df)
    {
        long nb_err = df.errors.length();
        df.errors.length(nb_err + 1);
        
        df.errors[nb_err].reason = CORBA::string_dup(
            "PyDs_UnmanagedSignalHandlerException");
        df.errors[nb_err].desc = CORBA::string_dup(
            "An unmanaged Tango::DevFailed exception occurred in signal_handler");
        df.errors[nb_err].origin = CORBA::string_dup("Device_4Impl.signal_handler");
        df.errors[nb_err].severity = Tango::ERR;

        Tango::Except::print_exception(df);
    }
}

void Device_4ImplWrap::default_signal_handler(long signo)
{
    this->Tango::Device_4Impl::signal_handler(signo);
}

///////////////////////////////////////////////////////////////////////////////
// Ext
///////////////////////////////////////////////////////////////////////////////

Py_Device_3ImplExt::Py_Device_3ImplExt(PyDeviceImplBase *ptr)
    :Tango::Device_3ImplExt(), my_dev(ptr)
{}

Py_Device_3ImplExt::~Py_Device_3ImplExt()
{}

void Py_Device_3ImplExt::delete_dev()
{
    my_dev->py_delete_dev();
}

#if ((defined sun) || (defined WIN32))
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(register_signal_overload,
                                       Tango::DeviceImpl::register_signal, 1, 1)
#else
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(register_signal_overload,
                                       Tango::DeviceImpl::register_signal, 1, 2)
#endif

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(append_status_overload,
                                       Tango::DeviceImpl::append_status, 1, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(set_change_event_overload,
                                       Tango::DeviceImpl::set_change_event, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(set_archive_event_overload,
                                       Tango::DeviceImpl::set_archive_event, 2, 3)

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(push_data_ready_event_overload,
                                       Tango::DeviceImpl::push_data_ready_event, 1, 2)


void export_device_impl()
{
 
    // The following function declarations are necessary to be able to cast
    // the function parameters from string& to const string&, otherwise python
    // will not recognize the method calls
    long (Tango::DeviceImpl::*get_cmd_poll_ring_depth_)(std::string &) =
        &Tango::DeviceImpl::get_cmd_poll_ring_depth;
    long (Tango::DeviceImpl::*get_attr_poll_ring_depth_)(std::string &) =
        &Tango::DeviceImpl::get_attr_poll_ring_depth;

    
    void (Tango::DeviceImpl::*stop_polling1)() = &Tango::DeviceImpl::stop_polling;
    void (Tango::DeviceImpl::*stop_polling2)(bool) = &Tango::DeviceImpl::stop_polling;
    
    class_<Tango::DeviceImpl, DeviceImplWrap, boost::noncopyable>("DeviceImpl",
        init<CppDeviceClass *, const char *,
             optional<const char *, Tango::DevState, const char *> >())

        .def("init_device", pure_virtual(&Tango::DeviceImpl::init_device))
        .def("set_state", &Tango::DeviceImpl::set_state)
        .def("get_state", &Tango::DeviceImpl::get_state,
            return_value_policy<copy_non_const_reference>())
        .def("get_prev_state", &Tango::DeviceImpl::get_prev_state,
            return_value_policy<copy_non_const_reference>())
        .def("get_name", &Tango::DeviceImpl::get_name,
            return_value_policy<copy_non_const_reference>())
        .def("get_device_attr", &Tango::DeviceImpl::get_device_attr,
            return_value_policy<reference_existing_object>())
        .def("register_signal",
            &Tango::DeviceImpl::register_signal,
            register_signal_overload())
        .def("unregister_signal", &Tango::DeviceImpl::unregister_signal)
        .def("get_status", &Tango::DeviceImpl::get_status,
            return_value_policy<copy_non_const_reference>())
        .def("set_status", &Tango::DeviceImpl::set_status)
        .def("append_status", &Tango::DeviceImpl::append_status,
            append_status_overload())
        .def("dev_state", &Tango::DeviceImpl::dev_state)
        .def("dev_status", &Tango::DeviceImpl::dev_status)
        .def("get_attribute_config", &PyDeviceImpl::get_attribute_config)
        .def("set_change_event",
            &Tango::DeviceImpl::set_change_event,
            set_change_event_overload())
        .def("set_archive_event",
            &Tango::DeviceImpl::set_archive_event,
            set_archive_event_overload())
        .def("_add_attribute", &PyDeviceImpl::add_attribute)
        .def("_remove_attribute", &PyDeviceImpl::remove_attribute)
        //@TODO .def("get_device_class")
        //@TODO .def("get_device_attr")
        //@TODO .def("get_db_device")
        
        
        .def("get_exported_flag", &Tango::DeviceImpl::get_exported_flag)
        .def("get_poll_ring_depth", &Tango::DeviceImpl::get_poll_ring_depth)
        .def("get_poll_old_factor", &Tango::DeviceImpl::get_poll_old_factor)
        .def("is_polled", (bool (Tango::DeviceImpl::*) ())&Tango::DeviceImpl::is_polled)
        .def("get_polled_cmd", &PyDeviceImpl::get_polled_cmd)
        .def("get_polled_attr", &PyDeviceImpl::get_polled_attr)
        .def("get_non_auto_polled_cmd", &PyDeviceImpl::get_non_auto_polled_cmd)
        .def("get_non_auto_polled_attr", &PyDeviceImpl::get_non_auto_polled_attr)
        //@TODO .def("get_poll_obj_list", &PyDeviceImpl::get_poll_obj_list)
        .def("stop_polling", stop_polling1)
        .def("stop_polling", stop_polling2)
        .def("check_command_exists", &Tango::DeviceImpl::check_command_exists)
        //@TODO .def("get_command", &PyDeviceImpl::get_command)
        .def("get_dev_idl_version", &Tango::DeviceImpl::get_dev_idl_version)
        .def("get_cmd_poll_ring_depth",
            (long (Tango::DeviceImpl::*) (const std::string &))
            get_cmd_poll_ring_depth_)
        .def("get_attr_poll_ring_depth",
            (long (Tango::DeviceImpl::*) (const std::string &))
            get_attr_poll_ring_depth_)
        .def("is_device_locked", &Tango::DeviceImpl::is_device_locked)
        
        .def("init_logger", &Tango::DeviceImpl::init_logger)
        .def("start_logging", &Tango::DeviceImpl::start_logging)
        .def("stop_logging", &Tango::DeviceImpl::stop_logging)
        
        //.def("set_exported_flag", &Tango::DeviceImpl::set_exported_flag)
        //.def("set_poll_ring_depth", &Tango::DeviceImpl::set_poll_ring_depth)

        .def("push_change_event",
            (void (*) (Tango::DeviceImpl &, str &))
            &PyDeviceImpl::push_change_event,
            (arg_("self"), arg_("attr_name")))

        .def("push_change_event",
            (void (*) (Tango::DeviceImpl &, str &, object &))
            &PyDeviceImpl::push_change_event)

        .def("push_change_event",
            (void (*) (Tango::DeviceImpl &, str &, str &, str &))
            &PyDeviceImpl::push_change_event)

        .def("push_change_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, long))
            &PyDeviceImpl::push_change_event)

        .def("push_change_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, long, long))
            &PyDeviceImpl::push_change_event)

        .def("push_change_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, double, Tango::AttrQuality))
            &PyDeviceImpl::push_change_event)

        .def("push_change_event",
            (void (*) (Tango::DeviceImpl &, str &, str &, str &, double, Tango::AttrQuality))
            &PyDeviceImpl::push_change_event)

        .def("push_change_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, double, Tango::AttrQuality, long))
            &PyDeviceImpl::push_change_event)

        .def("push_change_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, double, Tango::AttrQuality, long, long))
            &PyDeviceImpl::push_change_event)

        .def("push_archive_event",
            (void (*) (Tango::DeviceImpl &, str &))
            &PyDeviceImpl::push_archive_event,
            (arg_("self"), arg_("attr_name")))

        .def("push_archive_event",
            (void (*) (Tango::DeviceImpl &, str &, object &))
            &PyDeviceImpl::push_archive_event)

        .def("push_archive_event",
            (void (*) (Tango::DeviceImpl &, str &, str &, str &))
            &PyDeviceImpl::push_archive_event)

        .def("push_archive_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, long))
            &PyDeviceImpl::push_archive_event)

        .def("push_archive_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, long, long))
            &PyDeviceImpl::push_archive_event)

        .def("push_archive_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, double, Tango::AttrQuality))
            &PyDeviceImpl::push_archive_event)

        .def("push_archive_event",
            (void (*) (Tango::DeviceImpl &, str &, str &, str &, double, Tango::AttrQuality))
            &PyDeviceImpl::push_archive_event)

        .def("push_archive_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, double, Tango::AttrQuality, long))
            &PyDeviceImpl::push_archive_event)

        .def("push_archive_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, double, Tango::AttrQuality, long, long))
            &PyDeviceImpl::push_archive_event)

        .def("push_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, object &))
            &PyDeviceImpl::push_event)

        .def("push_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, object &, object &))
            &PyDeviceImpl::push_event)

        .def("push_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, object &, str &, str &))
            &PyDeviceImpl::push_event)

        .def("push_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, object &, object &, long))
            &PyDeviceImpl::push_event)

        .def("push_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, object &, object &, long, long))
            &PyDeviceImpl::push_event)

        .def("push_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, object &, object &, double, Tango::AttrQuality))
            &PyDeviceImpl::push_event)

        .def("push_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, object &, str &, str &, double, Tango::AttrQuality))
            &PyDeviceImpl::push_event)

        .def("push_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, object &, object &, double, Tango::AttrQuality, long))
            &PyDeviceImpl::push_event)

        .def("push_event",
            (void (*) (Tango::DeviceImpl &, str &, object &, object &, object &, double, Tango::AttrQuality, long, long))
            &PyDeviceImpl::push_event)

        .def("push_data_ready_event", &Tango::DeviceImpl::push_data_ready_event,
            push_data_ready_event_overload())

        .def("get_logger", &Tango::DeviceImpl::get_logger, return_internal_reference<>())
        .def("__debug_stream", &PyDeviceImpl::debug)
        .def("__info_stream", &PyDeviceImpl::info)
        .def("__warn_stream", &PyDeviceImpl::warn)
        .def("__error_stream", &PyDeviceImpl::error)
        .def("__fatal_stream", &PyDeviceImpl::fatal)
    ;

    class_<Tango::Device_2Impl, Device_2ImplWrap,
           bases<Tango::DeviceImpl>,
           boost::noncopyable>
           ("Device_2Impl",
            init<CppDeviceClass *, const char *,
                 optional<const char *, Tango::DevState, const char *> >())
        .def("get_attribute_config_2", &PyDevice_2Impl::get_attribute_config_2)
        //@TODO .def("read_attribute_history_2", &PyDevice_2Impl::read_attribute_history_2)
    ;

    class_<Tango::Device_3Impl, Device_3ImplWrap,
           bases<Tango::Device_2Impl>,
           boost::noncopyable>
           ("Device_3Impl",
            init<CppDeviceClass *, const char *,
                 optional<const char *, Tango::DevState, const char *> >())
        .def("init_device", pure_virtual(&Tango::Device_3Impl::init_device))
        .def("delete_device", &Tango::Device_3Impl::delete_device,
            &Device_3ImplWrap::default_delete_device)
        .def("always_executed_hook", &Tango::Device_3Impl::always_executed_hook,
            &Device_3ImplWrap::default_always_executed_hook)
        .def("read_attr_hardware", &Tango::Device_3Impl::read_attr_hardware,
            &Device_3ImplWrap::default_read_attr_hardware)
        .def("write_attr_hardware", &Tango::Device_3Impl::write_attr_hardware,
            &Device_3ImplWrap::default_write_attr_hardware)
        .def("dev_state", &Tango::Device_3Impl::dev_state,
            &Device_3ImplWrap::default_dev_state)
        .def("dev_status", &Tango::Device_3Impl::dev_status,
            &Device_3ImplWrap::default_dev_status)
        .def("signal_handler", &Tango::Device_3Impl::signal_handler,
            &Device_3ImplWrap::default_signal_handler)
        .def("get_attribute_config_3", &PyDevice_3Impl::get_attribute_config_3)
        .def("set_attribute_config_3", &PyDevice_3Impl::set_attribute_config_3)
    ;

    class_<Tango::Device_4Impl, Device_4ImplWrap,
           bases<Tango::Device_3Impl>,
           boost::noncopyable>
           ("Device_4Impl",
            init<CppDeviceClass *, const char *,
                 optional<const char *, Tango::DevState, const char *> >())
        .def("init_device", pure_virtual(&Tango::Device_4Impl::init_device))
        .def("delete_device", &Tango::Device_4Impl::delete_device,
            &Device_4ImplWrap::default_delete_device)
        .def("always_executed_hook", &Tango::Device_4Impl::always_executed_hook,
            &Device_4ImplWrap::default_always_executed_hook)
        .def("read_attr_hardware", &Tango::Device_4Impl::read_attr_hardware,
            &Device_4ImplWrap::default_read_attr_hardware)
        .def("write_attr_hardware", &Tango::Device_4Impl::write_attr_hardware,
            &Device_4ImplWrap::default_write_attr_hardware)
        .def("dev_state", &Tango::Device_4Impl::dev_state,
            &Device_4ImplWrap::default_dev_state)
        .def("dev_status", &Tango::Device_4Impl::dev_status,
            &Device_4ImplWrap::default_dev_status)
        .def("signal_handler", &Tango::Device_4Impl::signal_handler,
            &Device_4ImplWrap::default_signal_handler)
    ;
}
