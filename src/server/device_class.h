#ifndef _DEVICE_CLASS_H_
#define _DEVICE_CLASS_H_

#include <boost/python.hpp>
#include <tango.h>

class CppDeviceClass: public Tango::DeviceClass
{
public:
    CppDeviceClass(const string &name)
        :Tango::DeviceClass(const_cast<string&>(name))
    {}

    virtual ~CppDeviceClass()
    {}

    /**
     * Export a device.
     * Associate the servant to a CORBA object and send device network parameter
     * to TANGO database.
     * The main parameter sent to database is the CORBA object stringified device IOR.
     *
     * @param[in] dev The device to be exported (CORBA servant)
     * @param[in] corba_dev_name The name to be used in the CORBA object key.
     *                           This parameter does not need to be set in most of
     *                           cases and has a default value. It is used for special
     *                           device server like the database device server.
     */
    inline void export_device(Tango::DeviceImpl *dev, const char *corba_dev_nam = "Unused")
    {
        Tango::DeviceClass::export_device(dev, corba_dev_nam);
    }

    /**
     * Returns the python interpreter state
     *
     * @return python interpreter state pointer
     */
    inline PyInterpreterState *get_py_interp()
    {
        return interp;
    }

    /**
     * Sets the python interpreter state
     *
     * @param[in] in python interpreter state
     */
    inline void set_py_interp(PyInterpreterState *in)
    {
        interp = in;
    }

    /**
     * Creates an attribute and adds it to the att_list.
     * This method is intended to be called by python to register a new
     * attribute.
     */
    void create_attribute(vector<Tango::Attr *> &att_list,
                          const std::string &attr_name,
                          Tango::CmdArgType attr_type,
                          Tango::AttrDataFormat attr_format,
                          Tango::AttrWriteType attr_write,
                          long dim_x, long dim_y,
                          Tango::DispLevel display_level,
                          long polling_period,
                          bool memorized, bool hw_memorized,
                          const std::string &read_method_name,
                          const std::string &write_method_name,
                          const std::string &is_allowed_name,
                          Tango::UserDefaultAttrProp &att_prop);

    /**
     * Creates a command.
     * This method is intended to be called by python to register a new
     * command.
     */
    void create_command(const std::string &cmd_name,
                        Tango::CmdArgType param_type,
                        Tango::CmdArgType result_type,
                        const std::string &param_desc,
                        const std::string &result_desc,
                        Tango::DispLevel display_level,
                        bool default_command, long polling_period,
                        const std::string &is_allowed);

protected:
    PyInterpreterState *interp;
};

class CppDeviceClassWrap : public CppDeviceClass
{
public:

    /** a reference to itself */
    PyObject *m_self;

    /**
     * Constructor
     *
     * @param[in] self A reference to the python device class object
     * @param[in] name the class name
     */
    CppDeviceClassWrap(PyObject *self, const std::string &name)
        : CppDeviceClass(name), m_self(self)
    {
        init_class();
    }

    /**
     * Destructor
     */
    virtual ~CppDeviceClassWrap()
    {}

    /**
     * This method forward a C++ call to the device_factory method to the
     * Python method
     *
     * @param[in] dev_list The device name list
     */
    virtual void device_factory(const Tango::DevVarStringArray *dev_list);

    /**
     * This method forward a C++ call to the attribute_factory method to the
     * Python method
     *
     * @param[in] att_list attribute list
     */
    virtual void attribute_factory(std::vector<Tango::Attr *> &att_list);

    /**
     * This method forward a C++ call to the command_factory method to the
     * Python method
     */
    virtual void command_factory();

    /**
     * This method is called to ask Python to delete a class by decrementing
     * the Python ref count
     */
    virtual void delete_class();

    /**
     * This method forward a C++ call to the signal_handler method to the
     * Python method or executes default signal handler if no signal handler
     * is defined in python
     *
     * @param[in] signo signal identifier
     */
    virtual void signal_handler(long signo);

    /**
     * Default signal handler implementation
     *
     * @param[in] signo signal identifier
     */
    void default_signal_handler(long signo);

protected:

    /**
     * Initializes the class. Registers as a python DeviceClass to tango,
     * determines existence of a signal handler among other things
     */
    void init_class();

    /**
     * flag containing the information about the existence of a signal_handler
     * method in the python class
     */
    bool signal_handler_defined;
};

#endif // _DEVICE_CLASS_H_
