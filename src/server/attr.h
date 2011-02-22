#ifndef _ATTR_H_
#define _ATTR_H_

#include <boost/python.hpp>
#include <tango.h>

#include "exception.h"
#include "pytgutils.h"
#include "server/device_impl.h"

#define __AUX_DECL_CALL_ATTR_METHOD \
    PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev); \
    AutoPythonGIL __py_lock;

#define __AUX_CATCH_PY_EXCEPTION \
    catch(boost::python::error_already_set &eas) \
    { handle_python_exception(eas); }

#define CALL_ATTR_METHOD(dev, name) \
    __AUX_DECL_CALL_ATTR_METHOD \
    try { boost::python::call_method<void>(__dev_ptr->the_self, name); } \
    __AUX_CATCH_PY_EXCEPTION

#define CALL_ATTR_METHOD_VARGS(dev, name, ...) \
    __AUX_DECL_CALL_ATTR_METHOD \
    try { boost::python::call_method<void>(__dev_ptr->the_self, name, __VA_ARGS__); } \
    __AUX_CATCH_PY_EXCEPTION

#define CALL_ATTR_METHOD_RET(retType, ret, dev, name) \
    __AUX_DECL_CALL_ATTR_METHOD \
    try { ret = boost::python::call_method<retType>(__dev_ptr->the_self, name); } \
    __AUX_CATCH_PY_EXCEPTION

#define CALL_ATTR_METHOD_VARGS_RET(retType, ret, dev, name, ...) \
    __AUX_DECL_CALL_ATTR_METHOD \
    try { ret = boost::python::call_method<retType>(__dev_ptr->the_self, name, __VA_ARGS__); } \
    __AUX_CATCH_PY_EXCEPTION

#define RET_CALL_ATTR_METHOD(retType, dev, name) \
    __AUX_DECL_CALL_ATTR_METHOD \
    try { return boost::python::call_method<retType>(__dev_ptr->the_self, name); } \
    __AUX_CATCH_PY_EXCEPTION

#define RET_CALL_ATTR_METHOD_VARGS(retType, dev, name, ...) \
    __AUX_DECL_CALL_ATTR_METHOD \
    try { return boost::python::call_method<retType>(__dev_ptr->the_self, name, __VA_ARGS__); } \
    __AUX_CATCH_PY_EXCEPTION

class PyAttr
{
public:
    /**
     * Constructor
     */
    PyAttr()
    {}

    /**
     * Desctructor
     */
    virtual ~PyAttr()
    {}

    /**
     * Read one attribute. This method forward the action to the python method.
     *
     * @param[in] dev The device on which the attribute has to be read
     * @param[in, out] att the attribute
     */
    inline void read(Tango::DeviceImpl *dev,Tango::Attribute &att)
    {
        if (!_is_method(dev, read_name))
        {
            TangoSys_OMemStream o;
            o << read_name << " method not found";
            Tango::Except::throw_exception("PyTango_ReadAttributeMethodNotFound",
                o.str(), "PyTango::Attr::read");
        }
        
        CALL_ATTR_METHOD_VARGS(dev, read_name.c_str(), boost::ref(att))
    }

    /**
     * Write one attribute. This method forward the action to the python method.
     *
     * @param[in] dev The device on which the attribute has to be written
     * @param[in, out] att the attribute
     */
    inline void write(Tango::DeviceImpl *dev,Tango::WAttribute &att)
    {
        if (!_is_method(dev, write_name))
        {
            TangoSys_OMemStream o;
            o << write_name << " method not found";
            Tango::Except::throw_exception("PyTango_WriteAttributeMethodNotFound",
                o.str(), "PyTango::Attr::write");
        }
        CALL_ATTR_METHOD_VARGS(dev, write_name.c_str(), boost::ref(att))
    }

    /**
     * Decide if it is allowed to read/write the attribute
     *
     * @param[in] dev The device on which the attribute has to be read/written
     * @param[in] ty The requets type (read or write)
     *
     * @return a boolean set to true if it is allowed to execute
     *         the command. Otherwise, returns false
     */
    inline bool is_allowed(Tango::DeviceImpl *dev,Tango::AttReqType ty)
    {
        if (_is_method(dev, py_allowed_name))
        {
            RET_CALL_ATTR_METHOD_VARGS(bool, dev, py_allowed_name.c_str(), ty)
        }
        // keep compiler quiet
        return true;
    }

    /**
     * Sets the is_allowed method name for this attribute
     *
     * @param[in] name the is_allowed method name
     */
    inline void set_allowed_name(const std::string &name)
    {
        py_allowed_name = name;
    }

    /**
     * Sets the read method name for this attribute
     *
     * @param[in] name the read method name
     */
    inline void set_read_name(const std::string &name)
    {
        read_name = name;
    }

    /**
     * Sets the write method name for this attribute
     *
     * @param[in] name the write method name
     */
    inline void set_write_name(const std::string &name)
    {
        write_name = name;
    }

    /**
     * Transfer user property from a vector of AttrProperty
     * to a UserDefaultAttrProp
     *
     * @param[in] user_prop the AttrProperty vector
     * @param[out] def_prop  the UserDefaultAttrProp instance
     */
    void set_user_prop(std::vector<Tango::AttrProperty> &user_prop,
                       Tango::UserDefaultAttrProp &def_prop);

    inline bool _is_method(Tango::DeviceImpl *dev, const std::string &name)
    {   
        AutoPythonGIL __py_lock;
        PyDeviceImplBase *__dev_ptr = dynamic_cast<PyDeviceImplBase *>(dev);
        PyObject *__dev_py = __dev_ptr->the_self;
        return is_method_defined(__dev_py, name);
    }
    
private:

    /** the name of the is allowed python method */
    std::string py_allowed_name;
    
    /** the name of the read attribute python method */
    std::string read_name;

    /** the name of the write attribute python method */
    std::string write_name;
};

/**
 * The python class representing a scalar attribute
 */
class PyScaAttr: public Tango::Attr,
                 public PyAttr
{
public:

    /**
     * Python Scalar Attribute constructor
     *
     * @param[in] na The attribute name
     * @param[in] type  The attribute data type
     * @param[in] w The attribute writable type
     */
    PyScaAttr(const std::string &na, long type, Tango::AttrWriteType w)
        : Tango::Attr(na.c_str(), type, w)
    {}

    /**
     * Python Scalar Attribute constructor
     *
     * @param[in] na The attribute name
     * @param[in] type  The attribute data type
     * @param[in] w The attribute writable type
     * @param[in] ww The attribute max dim x
     * @param[in] user_prop The attribute user default properties
     */
    PyScaAttr(const std::string &na, long type, Tango::AttrWriteType w,
               std::vector<Tango::AttrProperty> &user_prop)
        : Tango::Attr(na.c_str(), type, w)
    {
        if (user_prop.size() == 0)
            return;

        Tango::UserDefaultAttrProp def_prop;
        set_user_prop(user_prop,def_prop);
        set_default_properties(def_prop);
    }

    /**
     * Python Scalar Attribute destructor
     */
    ~PyScaAttr() {};

    /**
     * Decide if it is allowed to read/write the attribute
     *
     * @param[in] dev The device on which the attribute has to be read/written
     * @param[in] ty The requets type (read or write)
     *
     * @return a boolean set to true if it is allowed to execute
     *         the command. Otherwise, returns false
     */
    inline virtual bool is_allowed(Tango::DeviceImpl *dev, Tango::AttReqType ty)
    {
        return PyAttr::is_allowed(dev, ty);
    }

    /**
     * Read one attribute. This method forward the action to the python method.
     *
     * @param[in] dev The device on which the attribute has to be read
     * @param[in, out] att the attribute
     */
    inline virtual void read(Tango::DeviceImpl *dev, Tango::Attribute &att)
    {
        return PyAttr::read(dev, att);
    }

    /**
     * Write one attribute. This method forward the action to the python method.
     *
     * @param[in] dev The device on which the attribute has to be written
     * @param[in, out] att the attribute
     */
    virtual void write(Tango::DeviceImpl *dev, Tango::WAttribute &att)
    {
        return PyAttr::write(dev, att);
    }
};

//------------------------------------------------------------------------------------------------


class PySpecAttr: public Tango::SpectrumAttr,
                  public PyAttr
{
public:
    /**
     * Python Spectrum Attribute constructor
     *
     * @param[in] na The attribute name
     * @param[in] type  The attribute data type
     * @param[in] w The attribute writable type
     * @param[in] xx The attribute max dim x
     */
    PySpecAttr(const std::string &na, long type, Tango::AttrWriteType w, long xx)
        : Tango::SpectrumAttr(na.c_str(), type, w, xx)
    {}

    /**
     * Python Spectrum Attribute constructor
     *
     * @param[in] na The attribute name
     * @param[in] type  The attribute data type
     * @param[in] w The attribute writable type
     * @param[in] xx The attribute max dim x
     * @param[in] user_prop The attribute user default properties
     */
    PySpecAttr(const std::string &na, long type, Tango::AttrWriteType w, long xx,
               std::vector<Tango::AttrProperty> &user_prop)
        : Tango::SpectrumAttr(na.c_str(), type, w, xx)
    {
        if (user_prop.size() == 0)
            return;

        Tango::UserDefaultAttrProp def_prop;
        set_user_prop(user_prop,def_prop);
        set_default_properties(def_prop);
    }

    /**
     * Python Spectrum Attribute destructor
     */
    ~PySpecAttr()
    {}

    /**
     * Decide if it is allowed to read/write the attribute
     *
     * @param[in] dev The device on which the attribute has to be read/written
     * @param[in] ty The requets type (read or write)
     *
     * @return a boolean set to true if it is allowed to execute
     *         the command. Otherwise, returns false
     */
    inline virtual bool is_allowed(Tango::DeviceImpl *dev, Tango::AttReqType ty)
    {
        return PyAttr::is_allowed(dev, ty);
    }

    /**
     * Read one attribute. This method forward the action to the python method.
     *
     * @param[in] dev The device on which the attribute has to be read
     * @param[in, out] att the attribute
     */
    inline virtual void read(Tango::DeviceImpl *dev, Tango::Attribute &att)
    {
        return PyAttr::read(dev, att);
    }

    /**
     * Write one attribute. This method forward the action to the python method.
     *
     * @param[in] dev The device on which the attribute has to be written
     * @param[in, out] att the attribute
     */
    virtual void write(Tango::DeviceImpl *dev, Tango::WAttribute &att)
    {
        return PyAttr::write(dev, att);
    }
};

//------------------------------------------------------------------------------------------------

class PyImaAttr: public Tango::ImageAttr,
                 public PyAttr
{
public:
    /**
     * Python Image Attribute constructor
     *
     * @param[in] na The attribute name
     * @param[in] type  The attribute data type
     * @param[in] w The attribute writable type
     * @param[in] xx The attribute max dim x
     * @param[in] yy The attribute max dim y
     */
    PyImaAttr(const std::string &na, long type, Tango::AttrWriteType w, long xx, long yy)
        : Tango::ImageAttr(na.c_str(), type, w, xx, yy)
    {}

    /**
     * Python Image Attribute constructor
     *
     * @param[in] na The attribute name
     * @param[in] type  The attribute data type
     * @param[in] w The attribute writable type
     * @param[in] xx The attribute max dim x
     * @param[in] yy The attribute max dim y
     * @param[in] user_prop The attribute user default properties
     */
    PyImaAttr(const std::string &na, long type, Tango::AttrWriteType w, long xx, long yy,
               std::vector<Tango::AttrProperty> &user_prop)
        : Tango::ImageAttr(na.c_str(), type, w, xx, yy)
    {
        if (user_prop.size() == 0)
            return;

        Tango::UserDefaultAttrProp def_prop;
        set_user_prop(user_prop,def_prop);
        set_default_properties(def_prop);
    }

    /**
     * Python Image Attribute destructor
     */
    ~PyImaAttr()
    {}

    /**
     * Decide if it is allowed to read/write the attribute
     *
     * @param[in] dev The device on which the attribute has to be read/written
     * @param[in] ty The requets type (read or write)
     *
     * @return a boolean set to true if it is allowed to execute
     *         the command. Otherwise, returns false
     */
    inline virtual bool is_allowed(Tango::DeviceImpl *dev, Tango::AttReqType ty)
    {
        return PyAttr::is_allowed(dev, ty);
    }

    /**
     * Read one attribute. This method forward the action to the python method.
     *
     * @param[in] dev The device on which the attribute has to be read
     * @param[in, out] att the attribute
     */
    inline virtual void read(Tango::DeviceImpl *dev, Tango::Attribute &att)
    {
        return PyAttr::read(dev, att);
    }

    /**
     * Write one attribute. This method forward the action to the python method.
     *
     * @param[in] dev The device on which the attribute has to be written
     * @param[in, out] att the attribute
     */
    virtual void write(Tango::DeviceImpl *dev, Tango::WAttribute &att)
    {
        return PyAttr::write(dev, att);
    }
};

#endif
