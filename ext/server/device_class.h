/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/
#ifndef _DEVICE_CLASS_H_
#define _DEVICE_CLASS_H_

#include <tango.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


class DeviceClass: public Tango::DeviceClass
{
public:
    DeviceClass(std::string& name, py::object py_self);

    virtual ~DeviceClass();

    /**
     * Export a device.
     * Associate the servant to a CORBA object and send device network parameter
     * to TANGO database.
     * The main parameter sent to database is the CORBA object stringified device IOR.
     *
     * @param[in] dev The device to be exported
     * @param[in] corba_dev_name The name to be used in the CORBA object key.
     *                           This parameter does not need to be set in most of
     *                           cases and has a default value. It is used for special
     *                           device server like the database device server.
     */
    inline void export_device(Tango::DeviceImpl *dev, const char *dev_nam = "Unused")
    {
        Tango::DeviceClass::export_device(dev, dev_nam);
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
    Tango::Attr* create_attribute(const std::string& attr_name,
                          Tango::CmdArgType attr_type,
                          Tango::AttrDataFormat attr_format,
                          Tango::AttrWriteType attr_write,
                          long dim_x, long dim_y,
                          Tango::DispLevel display_level,
                          long polling_period,
                          bool memorized, bool hw_memorized,
                          const std::string& read_method_name,
                          const std::string& write_method_name,
                          const std::string& is_allowed_name,
                          Tango::UserDefaultAttrProp *att_prop);

    Tango::Attr* create_fwd_attribute(const std::string& attr_name,
                              Tango::UserDefaultFwdAttrProp *att_prop);

    /**
     * Creates an pipe and adds it to the att_list.
     * This method is intended to be called by python to register a new
     * pipe.
     */
    Tango::Pipe* create_pipe(//std::vector<Tango::Pipe *>& pipe_list,
             const std::string& name,
             Tango::PipeWriteType access,
             Tango::DispLevel display_level,
             const std::string& read_method_name,
             const std::string& write_method_name,
             const std::string& is_allowed_name,
             Tango::UserDefaultPipeProp *prop);

    /**
     * Creates a command.
     * This method is intended to be called by python to register a new
     * command.
     */
    void create_command(std::string& cmd_name,
                        Tango::CmdArgType param_type,
                        Tango::CmdArgType result_type,
                        std::string& param_desc,
                        std::string& result_desc,
                        Tango::DispLevel display_level,
                        bool default_command, long polling_period,
                        const std::string& is_allowed);

    void command_factory();
    void device_factory(const Tango::DevVarStringArray *dev_list);
    void device_name_factory(std::vector<std::string> &dev_list);
    void attribute_factory(std::vector<Tango::Attr *> &att_list);
    void pipe_factory();
    void signal_handler(long signo);
//    void default_signal_handler(long signo);
    void delete_class();

    py::object m_self;


protected:
    PyInterpreterState *interp;
    bool signal_handler_defined;
};

#endif // _DEVICE_CLASS_H_
