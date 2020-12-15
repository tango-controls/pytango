/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#ifndef _DEVICE_IMPL_H
#define _DEVICE_IMPL_H

#include <boost/python.hpp>
#include <tango.h>

#include "server/device_class.h"

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
                   const char *status = Tango::StatusNotSet);

    /**
     * Destructor
     */
    virtual ~DeviceImplWrap()
    {}

    /**
     * Invokes the actual init_device
     */
    void init_device();
    
    bool _is_attribute_polled(const std::string &att_name);
    bool _is_command_polled(const std::string &cmd_name);
    int _get_attribute_poll_period(const std::string &att_name);
    int _get_command_poll_period(const std::string &cmd_name);
    void _poll_attribute(const std::string &att_name, int period);
    void _poll_command(const std::string &cmd_name, int period);
    void _stop_poll_attribute(const std::string &att_name);
    void _stop_poll_command(const std::string &cmd_name);
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
                     const char *status = Tango::StatusNotSet);

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

    std::string the_status;

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
                     const char *status = Tango::StatusNotSet);

    /**
     * Destructor
     */
    virtual ~Device_3ImplWrap();
    
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
    virtual void read_attr_hardware(std::vector<long> &attr_list);

    /**
     * Executes default read_attr_hardware implementation
     */
    void default_read_attr_hardware(std::vector<long> &attr_list);

    /**
     * Necessary write_attr_hardware implementation to call python
     */
    virtual void write_attr_hardware(std::vector<long> &attr_list);

    /**
     * Executes default write_attr_hardware implementation
     */
    void default_write_attr_hardware(std::vector<long> &attr_list);

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
                     const char *status = Tango::StatusNotSet);

    /**
     * Destructor
     */
    virtual ~Device_4ImplWrap();

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
    virtual void read_attr_hardware(std::vector<long> &attr_list);

    /**
     * Executes default read_attr_hardware implementation
     */
    void default_read_attr_hardware(std::vector<long> &attr_list);

    /**
     * Necessary write_attr_hardware implementation to call python
     */
    virtual void write_attr_hardware(std::vector<long> &attr_list);

    /**
     * Executes default write_attr_hardware implementation
     */
    void default_write_attr_hardware(std::vector<long> &attr_list);

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
 * Device_5ImplWrap is the class used to represent a Python Tango device.
 */
class Device_5ImplWrap : public Tango::Device_5Impl,
                         public PyDeviceImplBase,
                         public boost::python::wrapper<Tango::Device_5Impl>
{
public:
    /**
     * Constructor
     *
     * @param[in] self
     * @param[in] cl
     * @param[in] st
     */
    Device_5ImplWrap(PyObject *self, CppDeviceClass *cl, std::string &st);

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
    Device_5ImplWrap(PyObject *self, CppDeviceClass *cl, const char *name,
                     const char *desc = "A Tango device",
                     Tango::DevState sta = Tango::UNKNOWN,
                     const char *status = Tango::StatusNotSet);

    /**
     * Destructor
     */
    virtual ~Device_5ImplWrap();

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
    virtual void read_attr_hardware(std::vector<long> &attr_list);

    /**
     * Executes default read_attr_hardware implementation
     */
    void default_read_attr_hardware(std::vector<long> &attr_list);

    /**
     * Necessary write_attr_hardware implementation to call python
     */
    virtual void write_attr_hardware(std::vector<long> &attr_list);

    /**
     * Executes default write_attr_hardware implementation
     */
    void default_write_attr_hardware(std::vector<long> &attr_list);

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
#endif // _DEVICE_IMPL_H
