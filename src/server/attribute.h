#ifndef _ATTRIBUTE_H_
#define _ATTRIBUTE_H_

#include <boost/python.hpp>
#include <tango.h>

namespace PyAttribute
{
    void set_value(Tango::Attribute &, boost::python::object &);

    void set_value(Tango::Attribute &, boost::python::str &,
                   boost::python::str &);

    void set_value(Tango::Attribute &, boost::python::object &, long);

    void set_value(Tango::Attribute &, boost::python::object &, long, long);

    void set_value_date_quality(Tango::Attribute &, boost::python::object &,
                                double, Tango::AttrQuality);

    void set_value_date_quality(Tango::Attribute &, boost::python::str &,
                                boost::python::str &, double,
                                Tango::AttrQuality);

    void set_value_date_quality(Tango::Attribute &, boost::python::object &,
                                double, Tango::AttrQuality , long);

    void set_value_date_quality(Tango::Attribute &, boost::python::object &,
                                double, Tango::AttrQuality , long, long);
                                
    boost::python::object get_properties(Tango::Attribute &,
                                         boost::python::object &);

    boost::python::object get_properties_2(Tango::Attribute &,
                                           boost::python::object &);

    boost::python::object get_properties_3(Tango::Attribute &,
                                           boost::python::object &);

    void set_properties(Tango::Attribute &, boost::python::object &,
                        boost::python::object &);
    
    void set_properties_3(Tango::Attribute &, boost::python::object &,
                          boost::python::object &);
};


#endif // _ATTRIBUTE_H_
