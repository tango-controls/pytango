#ifndef _DEVICE_IMPL_H
#define _DEVICE_IMPL_H

#include <boost/python.hpp>
#include <tango.h>

#include <server/device_class.h>

/**
 * A wrapper around the Tango::DeviceImpl class
 */
class DeviceImplWrap : public Tango::DeviceImpl,
                       public boost::python::wrapper<Tango::DeviceImpl>
{
public:
    /** a reference to itself */
    PyObject *m_self;

    /**
     * Constructor
     *
     * @param[in] self
     * @param[in] cl
     * @param[in] st
     */
    DeviceImplWrap(PyObject *self, CppDeviceClass *cl, std::string &st);

    /**
     * Constructor
     *
     * @param[in] self
     * @param[in] cl
     * @param[in] name
     * @param[in] desc
     * @param[in] sta
     * @param[in] status
     */
    DeviceImplWrap(PyObject *self, CppDeviceClass *cl, const char *name,
                   const char *desc = "A Tango device",
                   Tango::DevState sta = Tango::UNKNOWN,
                   const char *status = StatusNotSet);

    /**
     * Destructor
     */
    virtual ~DeviceImplWrap()
    {}

    /**
     * Invokes the actual init_device
     */
    void init_device();
};

/**
 * A wrapper around the Tango::Device_2Impl class
 */
class Device_2ImplWrap : public Tango::Device_2Impl,
                         public boost::python::wrapper<Tango::Device_2Impl>
{
public:
    /** a reference to itself */
    PyObject *m_self;

    /**
     * Constructor
     *
     * @param[in] self
     * @param[in] cl
     * @param[in] st
     */
    Device_2ImplWrap(PyObject *self, CppDeviceClass *cl, std::string &st);

    /**
     * Constructor
     *
     * @param[in] self
     * @param[in] cl
     * @param[in] name
     * @param[in] desc
     * @param[in] sta
     * @param[in] status
     */
    Device_2ImplWrap(PyObject *self, CppDeviceClass *cl, const char *name,
                     const char *desc = "A Tango device",
                     Tango::DevState sta = Tango::UNKNOWN,
                     const char *status = StatusNotSet);

    /**
     * Destructor
     */
    virtual ~Device_2ImplWrap()
    {}

    /**
     * Invokes the actual init_device
     */
    void init_device();
};

class PyDeviceImplBase
{
public:
    /** a reference to itself */
    PyObject *the_self;

    PyDeviceImplBase(PyObject *self);

    virtual ~PyDeviceImplBase();

    virtual void py_delete_dev();
};

/**
 * A wrapper around the Tango::Device_3Impl class
 */
class Device_3ImplWrap : public Tango::Device_3Impl,
                         public PyDeviceImplBase,
                         public boost::python::wrapper<Tango::Device_3Impl>
{
public:
    /**
     * Constructor
     *
     * @param[in] self
     * @param[in] cl
     * @param[in] st
     */
    Device_3ImplWrap(PyObject *self, CppDeviceClass *cl, std::string &st);

    /**
     * Constructor
     *
     * @param[in] self
     * @param[in] cl
     * @param[in] name
     * @param[in] desc
     * @param[in] sta
     * @param[in] status
     */
    Device_3ImplWrap(PyObject *self, CppDeviceClass *cl, const char *name,
                     const char *desc = "A Tango device",
                     Tango::DevState sta = Tango::UNKNOWN,
                     const char *status = StatusNotSet);

    /**
     * A wrapper around the add_attribute in order to process some
     * internal information
     *
     * @param att the attribute reference containning information about
     *            the new attribute to be added
     */
    void _add_attribute(const Tango::Attr &att);

    /**
     * Wrapper around the remove_attribute in order to simplify
     * string & to const string & conversion and default parameters
     */
    void _remove_attribute(const char *att_name);

    /**
     * Destructor
     */
    virtual ~Device_3ImplWrap()
    {}

    /**
     * Necessary init_device implementation to call python
     */
    virtual void init_device();

    /**
     * Necessary delete_device implementation to call python
     */
    virtual void delete_device();

    /**
     * Executes default delete_device implementation
     */
    void default_delete_device();

    /**
     * called to ask Python to delete a device by decrementing the Python
     * reference count
     */
    virtual void delete_dev();

    /**
     * Necessary always_executed_hook implementation to call python
     */
    virtual void always_executed_hook();

    /**
     * Executes default always_executed_hook implementation
     */
    void default_always_executed_hook();

    /**
     * Necessary read_attr_hardware implementation to call python
     */
    virtual void read_attr_hardware(vector<long> &attr_list);

    /**
     * Executes default read_attr_hardware implementation
     */
    void default_read_attr_hardware(vector<long> &attr_list);

    /**
     * Necessary write_attr_hardware implementation to call python
     */
    virtual void write_attr_hardware(vector<long> &attr_list);

    /**
     * Executes default write_attr_hardware implementation
     */
    void default_write_attr_hardware(vector<long> &attr_list);

    /**
     * Necessary dev_state implementation to call python
     */
    virtual Tango::DevState dev_state();

    /**
     * Executes default dev_state implementation
     */
    Tango::DevState default_dev_state();

    /**
     * Necessary dev_status implementation to call python
     */
    virtual Tango::ConstDevString dev_status();

    /**
     * Executes default dev_status implementation
     */
    Tango::ConstDevString default_dev_status();

    /**
     * Necessary signal_handler implementation to call python
     */
    virtual void signal_handler(long signo);

    /**
     * Executes default signal_handler implementation
     */
    void default_signal_handler(long signo);

    virtual void py_delete_dev();

protected:
    /**
     * internal method used to initialize the class. Called by the constructors
     */
    void _init();
};

/**
 * Device_4ImplWrap is the class used to represent a Python Tango device.
 */
class Device_4ImplWrap : public Tango::Device_4Impl,
                         public PyDeviceImplBase,
                         public boost::python::wrapper<Tango::Device_4Impl>
{
public:
    /**
     * Constructor
     *
     * @param[in] self
     * @param[in] cl
     * @param[in] st
     */
    Device_4ImplWrap(PyObject *self, CppDeviceClass *cl, std::string &st);

    /**
     * Constructor
     *
     * @param[in] self
     * @param[in] cl
     * @param[in] name
     * @param[in] desc
     * @param[in] sta
     * @param[in] status
     */
    Device_4ImplWrap(PyObject *self, CppDeviceClass *cl, const char *name,
                     const char *desc = "A Tango device",
                     Tango::DevState sta = Tango::UNKNOWN,
                     const char *status = StatusNotSet);

    /**
     * Destructor
     */
    virtual ~Device_4ImplWrap()
    {}

    /**
     * A wrapper around the add_attribute in order to process some
     * internal information
     *
     * @param att the attribute reference containning information about
     *            the new attribute to be added
     */
    void _add_attribute(const Tango::Attr &att);

    /**
     * Wrapper around the remove_attribute in order to simplify
     * string & to const string & conversion and default parameters
     */
    void _remove_attribute(const char *att_name);

    /**
     * Necessary init_device implementation to call python
     */
    virtual void init_device();

    /**
     * Necessary delete_device implementation to call python
     */
    virtual void delete_device();

    /**
     * Executes default delete_device implementation
     */
    void default_delete_device();

    /**
     * called to ask Python to delete a device by decrementing the Python
     * reference count
     */
    virtual void delete_dev();

    /**
     * Necessary always_executed_hook implementation to call python
     */
    virtual void always_executed_hook();

    /**
     * Executes default always_executed_hook implementation
     */
    void default_always_executed_hook();

    /**
     * Necessary read_attr_hardware implementation to call python
     */
    virtual void read_attr_hardware(vector<long> &attr_list);

    /**
     * Executes default read_attr_hardware implementation
     */
    void default_read_attr_hardware(vector<long> &attr_list);

    /**
     * Necessary write_attr_hardware implementation to call python
     */
    virtual void write_attr_hardware(vector<long> &attr_list);

    /**
     * Executes default write_attr_hardware implementation
     */
    void default_write_attr_hardware(vector<long> &attr_list);

    /**
     * Necessary dev_state implementation to call python
     */
    virtual Tango::DevState dev_state();

    /**
     * Executes default dev_state implementation
     */
    Tango::DevState default_dev_state();

    /**
     * Necessary dev_status implementation to call python
     */
    virtual Tango::ConstDevString dev_status();

    /**
     * Executes default dev_status implementation
     */
    Tango::ConstDevString default_dev_status();

    /**
     * Necessary signal_handler implementation to call python
     */
    virtual void signal_handler(long signo);

    /**
     * Executes default signal_handler implementation
     */
    void default_signal_handler(long signo);

    virtual void py_delete_dev();

protected:
    /**
     * internal method used to initialize the class. Called by the constructors
     */
    void _init();
};

/**
 * Device_3Impl extension wrapper
 */
class Py_Device_3ImplExt:public Tango::Device_3ImplExt
{
public:
    /** pointer to pytango wrapper */
    PyDeviceImplBase *my_dev;

    /**
     * Constructor
     *
     * @param[in] ptr Device_4Impl wrapper pointer
     */
    Py_Device_3ImplExt(PyDeviceImplBase *ptr);

    /**
     * Destructor
     */
    virtual ~Py_Device_3ImplExt();

    /**
     * overwrite of delete_dev method
     */
    virtual void delete_dev();

};

#endif // _DEVICE_IMPL_H
