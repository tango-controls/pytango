/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#include <stdlib.h>
#include <tango.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

//#include "tango_numpy.h"

#include <memory.h>
#include <tgutils.h>
#include <pytgutils.h>

namespace py = pybind11;


# ifdef WIN32
#   define PYTG_TIME_FROM_DOUBLE(dbl, tv) \
            if (true) { \
                tv.time = (time_t)floor(dbl); \
                tv.millitm = (unsigned short)((dbl - tv.time)*1.0e3); \
            } else (void)0
#   define PYTG_NEW_TIME_FROM_DOUBLE(dbl, tv) \
            struct _timeb tv; PYTG_TIME_FROM_DOUBLE(dbl, tv)
# else
#   define PYTG_TIME_FROM_DOUBLE(dbl, tv) double2timeval(tv, dbl);
#   define PYTG_NEW_TIME_FROM_DOUBLE(dbl, tv) \
            struct timeval tv; PYTG_TIME_FROM_DOUBLE(dbl, tv)
#endif


inline static void throw_wrong_python_data_type(const std::string& att_name,
                                         const std::string& method)
{
    std::stringstream o;
    o << "Wrong Python type for attribute " << att_name << ends;
    Tango::Except::throw_exception(
            "PyDs_WrongPythonDataTypeForAttribute",
            o.str(), method);
}

inline static void throw_wrong_python_data_type_in_array(const std::string& att_name,
                                                  long idx,
                                                  const std::string& method)
{
    std::stringstream o;
    o << "Wrong Python type for attribute " << att_name
      << ".\nElement with index " << idx << " in sequence does not "
      << "have a correct type." << ends;

    Tango::Except::throw_exception(
            "PyDs_WrongPythonDataTypeForAttribute",
            o.str(), method);
}

extern long TANGO_VERSION_HEX;

namespace PyAttribute
{
    template<long tangoTypeConst>
    inline void __set_value_scalar(Tango::Attribute& att, py::object& value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType cpp_val = value.cast<TangoScalarType>();
        att.set_value(&cpp_val, 1, 0, false);
    }

    template<>
    inline void __set_value_scalar<Tango::DEV_STRING>(Tango::Attribute& att, py::object& value)
    {
        std::string cpp_val =  value.cast<std::string>();
        char* cptr = strdup(cpp_val.c_str());
        att.set_value(&cptr, 1, 0, false);
    }

    /**
     * Tango Attribute set_value wrapper for DevEncoded attributes
     *
     * @param att attribute reference
     * @param data_str new attribute data string
     * @param data new attribute data
     */
    inline void __set_value(Tango::Attribute& att, std::string& data_str,
                            std::string& data)
    {
        Tango::DevString val_str = strdup(data_str.c_str());
        Tango::DevString val = strdup(data.c_str());
        att.set_value(&val_str, (Tango::DevUChar*)val, (long)data.size());
    }

    /**
     * Tango Attribute set_value wrapper for DevEncoded attributes
     *
     * @param att attribute reference
     * @param data_str new attribute data string
     * @param data new attribute data
     */
    inline void __set_value(Tango::Attribute& att,
                            std::string& data_str,
                            py::object& data)
    {
        Tango::DevString val_str = strdup(data_str.c_str());
        long len = py::len(data.ptr());
        unsigned char* bptr = new unsigned char[len];
        for (auto& item : data) {
            *bptr++ = item.cast<unsigned char>();
        }
        att.set_value(&val_str, bptr-len, len);
    }

    /**
     * Tango Attribute set_value_date_quality wrapper for scalar attributes
     *
     * @param att attribute reference
     * @param value new attribute value
     * @param t timestamp
     * @param quality attribute quality
     */
    template<long tangoTypeConst>
    inline void __set_value_date_quality_scalar(Tango::Attribute& att,
                                                py::object& value,
                                                double t, Tango::AttrQuality quality)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        PYTG_NEW_TIME_FROM_DOUBLE(t, tv);
        /*
           I hate doing because tango inside is doing a new again when
           set_value_date_quality is invoked with the release flag to true
           the other option would be to use per thread tango data like it was
           done in v3.0.4
           I prefer this one since it decouples TangoC++ from PyTango and creating
           a scalar is not so expensive after all
        */
        std::unique_ptr<TangoScalarType> cpp_val(new TangoScalarType);
        *cpp_val =  value.cast<TangoScalarType>();
        att.set_value_date_quality(cpp_val.release(), tv, quality, 1, 0, true);
    }

    template<>
    inline void __set_value_date_quality_scalar<Tango::DEV_STRING>(Tango::Attribute& att,
                                                py::object& value,
                                                double t, Tango::AttrQuality quality)
    {
        PYTG_NEW_TIME_FROM_DOUBLE(t, tv);
        /*
           I hate doing because tango inside is doing a new again when
           set_value_date_quality is invoked with the release flag to true
           the other option would be to use per thread tango data like it was
           done in v3.0.4
           I prefer this one since it decouples TangoC++ from PyTango and creating
           a scalar is not so expensive after all
        */
        std::string cpp_val =  value.cast<std::string>();
        Tango::DevString val = const_cast<char*>(cpp_val.c_str());
         att.set_value_date_quality(&val, tv, quality, 1, 0, true);
    }

    /**
     * Tango Attribute set_value_date_quality wrapper for DevEncoded attributes
     *
     * @param att attribute reference
     * @param data_str new attribute data string
     * @param data new attribute data
     * @param t timestamp
     * @param quality attribute quality
     */
    inline void __set_value_date_quality(Tango::Attribute& att,
                                         std::string& data_str,
                                         std::string& data,
                                         double t, Tango::AttrQuality quality)
    {
        PYTG_NEW_TIME_FROM_DOUBLE(t, tv);
        Tango::DevString val_str = const_cast<char*>(data_str.c_str());
        Tango::DevString val = const_cast<char*>(data.c_str());
        att.set_value(&val_str, (Tango::DevUChar*)val, (long)data.size());
        att.set_value_date_quality(&val_str, (Tango::DevUChar*)val,
                                   (long)data.size(), tv, quality);
    }

    /**
     * Tango Attribute set_value_date_quality wrapper for DevEncoded attributes
     *
     * @param att attribute reference
     * @param data_str new attribute data string
     * @param data new attribute data
     * @param t timestamp
     * @param quality attribute quality
     */
    inline void __set_value_date_quality(Tango::Attribute& att,
                                         std::string& data_str,
                                         py::object& data,
                                         double t, Tango::AttrQuality quality)
    {
        PYTG_NEW_TIME_FROM_DOUBLE(t, tv);
        Py_buffer view;
        if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_FULL_RO) < 0)
        {
            throw_wrong_python_data_type(att.get_name(), "set_value()");
        }
        Tango::DevString val_str = const_cast<char*>(data_str.c_str());
        att.set_value(&val_str, (Tango::DevUChar*)view.buf, (long)view.len);
// why wont this line compile????????????????
//        att.set_value(&val_str, (Tango::DevUChar*)view.buf, (long)view.len, tv, quality);
        PyBuffer_Release(&view);
    }

    template<long tangoTypeConst>
    void __set_value_date_quality_array(
            Tango::Attribute& att,
            py::object& value,
            double& time,
            Tango::AttrQuality*& quality,
            long& x,
            long& y,
            const std::string& fname,
            const bool& isImage)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;

        py::list val_list(value);
        long len;
        if (isImage) {
            y = py::len(val_list);
            py::list sub_val_list = val_list[0];
            x = py::len(sub_val_list);
            len = x * y;
        } else {
            x = py::len(val_list);
            y = 0;
            len = x;
        }
        TangoScalarType* data_buffer = new TangoScalarType[len];
        if (isImage) {
            long k = 0;
            for (int j=0; j<y; j++) {
                py::list sub_val_list = val_list[j];
                for (int i=0; i<x; i++) {
                    data_buffer[k] = sub_val_list[i].cast<TangoScalarType>();
                    k +=1;
                }
            }
        } else {
            for (int i=0; i<x; i++) {
                data_buffer[i] = val_list[i].cast<TangoScalarType>();
            }
        }
        if (quality) {
            PYTG_NEW_TIME_FROM_DOUBLE(time, tv);
            att.set_value_date_quality(
                data_buffer, tv, *quality, x, y, false);
        } else {
            att.set_value(data_buffer, x, y, false);
        }
        delete data_buffer;
    }

    template<>
    void __set_value_date_quality_array<Tango::DEV_STRING>(
            Tango::Attribute& att,
            py::object& value,
            double& time,
            Tango::AttrQuality*& quality,
            long& x,
            long& y,
            const std::string& fname,
            const bool& isImage)
    {
        long len;
        py::list val_list(value);
        if (isImage) {
            y = py::len(val_list);
            py::list sub_val_list = val_list[0];
            x = py::len(sub_val_list);
            len = x * y;
        } else {
            x = py::len(val_list);
            y = 0;
            len = x;
        }
        char* data_buffer[len];
        if (isImage) {
            long k = 0;
            for (int j=0; j<y; j++) {
                py::list sub_val_list = val_list[j];
                for (int i=0; i<x; i++) {
                    std::string cpp_val =  sub_val_list[i].cast<std::string>();
                    data_buffer[k] = strdup(cpp_val.c_str());
                    k +=1;
                }
            }
        } else {
            for (int i=0; i<x; i++) {
                std::string cpp_val =  val_list[i].cast<std::string>();
                data_buffer[i] = strdup(cpp_val.c_str());
            }
        }
        if (quality) {
            PYTG_NEW_TIME_FROM_DOUBLE(time, tv);
            att.set_value_date_quality(
                data_buffer, tv, *quality, x, y, false);
        } else {
            att.set_value(data_buffer, x, y, false);
        }
    }

    inline void __set_value(const std::string& fname, Tango::Attribute &att, py::object& value,
            long x, long y, double t = 0.0, Tango::AttrQuality* quality = nullptr)
    {
        long type = att.get_data_type();
        Tango::AttrDataFormat format = att.get_data_format();

        const bool isScalar = (format == Tango::SCALAR);
        const bool isImage = (format == Tango::IMAGE);

        if (isScalar) {
            if (x > 1 || y > 0) {
                std::stringstream o;
                o << "Cannot call " << fname;
                if (y > 0) {
                    o << "(data, dim_x, dim_y) on scalar attribute";
                } else {
                    o << "(data, dim_x) on scalar attribute";
                }
                if (quality) {
                    o << att.get_name() << ". Use set_value_date_quality(data) instead" << ends;
                } else {
                    o << att.get_name() << ". Use set_value(data) instead" << ends;
                }
                Tango::Except::throw_exception(
                        "PyDs_WrongPythonDataTypeForAttribute",
                        o.str(),
                        fname + "()");
            } else {
                if (quality) {
                    TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, __set_value_date_quality_scalar, att, value, t, *quality);
                } else {
                    TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type, __set_value_scalar, att, value);
                }
            }
        } else {
            TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(type,
                __set_value_date_quality_array,
                    att, value, t, quality, x, y, fname, isImage);
        }
    }

    inline void __set_value(const std::string& fname, Tango::Attribute &att, std::string& data_str, std::string& data, double t = 0.0, Tango::AttrQuality* quality = 0)
    {
        if (quality)
            __set_value_date_quality(att, data_str, data, t, *quality);
        else
            __set_value(att, data_str, data);
    }

    inline void __set_value(const std::string& fname, Tango::Attribute &att, std::string &data_str, py::object& data, double t = 0.0, Tango::AttrQuality* quality = 0)
    {
        if (quality)
            __set_value_date_quality(att, data_str, data, t, *quality);
        else
            __set_value(att, data_str, data);
    }

    void set_value(Tango::Attribute &att, py::object& value)
    {
        __set_value("set_value", att, value, 0, 0);
    }

    void set_value(Tango::Attribute &att, Tango::EncodedAttribute *data)
    {
        att.set_value(data);
    }

    void set_value(Tango::Attribute &att, std::string& data_str, std::string& data)
    {
        __set_value("set_value", att, data_str, data);
    }

    void set_value(Tango::Attribute &att, std::string& data_str, py::object& data)
    {
        __set_value("set_value", att, data_str, data);
    }

    void set_value(Tango::Attribute &att, py::object& value, long x)
    {
        __set_value("set_value", att, value, x, 0);
    }

    void set_value(Tango::Attribute &att, py::object& value, long x, long y)
    {
        __set_value("set_value", att, value, x, y);
    }

    void set_value_date_quality(Tango::Attribute &att,
                                       py::object& value, double t,
                                       Tango::AttrQuality quality)
    {
        __set_value("set_value_date_quality", att, value, 0, 0, t, &quality);
    }

    void set_value_date_quality(Tango::Attribute &att,
                                       std::string& data_str,
                                       std::string& data,
                                       double t,
                                       Tango::AttrQuality quality)
    {
        __set_value("set_value_date_quality", att, data_str, data, t, &quality);
    }

    void set_value_date_quality(Tango::Attribute &att,
                                       std::string& data_str,
                                       py::object& data,
                                       double t,
                                       Tango::AttrQuality quality)
    {
        __set_value("set_value_date_quality", att, data_str, data, t, &quality);
    }

    void set_value_date_quality(Tango::Attribute &att,
                                       py::object& value,
                                       double t, Tango::AttrQuality quality,
                                       long x)
    {
        __set_value("set_value_date_quality", att, value, x, 0, t, &quality);
    }

    void set_value_date_quality(Tango::Attribute &att,
                                       py::object& value,
                                       double t, Tango::AttrQuality quality,
                                       long x, long y)
    {
        __set_value("set_value_date_quality", att, value, x, y, t, &quality);
    }

    void set_complex_value(Tango::Attribute& attr, py::object& value)
    {
        long dtype = attr.get_data_type();
        Tango::AttrDataFormat fmt = attr.get_data_format();
        if (dtype == Tango::DEV_ENCODED) {
            py::tuple tup = py::cast<py::tuple>(value);
            std::string ss0 = tup[0].cast<std::string>();
            try {
                std::string ss1 = tup[1].cast<std::string>();
                set_value(attr, ss0, ss1);
            } catch (...) {
                py::object obj = tup[1];
                set_value(attr, ss0, obj);
            }

//            if is_tuple and len(value) == 2:
//                if is_pure_str(value[1]):
//                    print("this one?")
//                    set_value(attr, value[0], str(value[1]))
//                else:
//                    set_value(attr, value[0], value[1])
//            elif is_tuple and len(value) == 4:
//                set_value_date_quality(attr, *value)
//            elif is_tuple and len(value) == 3 and is_non_str_seq(value[0]):
//                set_value_date_quality(attr, value[0][0], value[0][1], *value[1:])
//            else:
//                set_value(*value);
        } else {
//           if (is_tuple) {
//                if len(value) == 3:
//                    if fmt == AttrDataFormat.SCALAR:
//                        set_value_date_quality(attr, *value)
//                    elif fmt == AttrDataFormat.SPECTRUM:
//                        if is_seq(value[0]):
//                            set_value_date_quality(attr, *value)
//                        else:
//                            set_value(attr, value)
//                    else:
//                        if is_seq(value[0]) and is_seq(value[0][0]):
//                            set_value_date_quality(attr, *value)
//                        else:
//                            set_value(attr, value)
//                else:
//                    set_value(attr, value);
//            } else {
//            }
            if (fmt == Tango::SCALAR) {
                set_value(attr, value);
            } else if (fmt == Tango::SPECTRUM) {
                set_value(attr, value, 0);
            } else {
                set_value(attr, value, 0, 0);
            }
        }
}

    template<typename TangoScalarType>
    inline void _get_properties_multi_attr_prop(Tango::Attribute &att, py::object& multi_attr_prop)
    {
        Tango::MultiAttrProp<TangoScalarType> tg_multi_attr_prop;
        att.get_properties(tg_multi_attr_prop);
//        to_py(tg_multi_attr_prop,multi_attr_prop);
    }

    inline py::object get_properties_multi_attr_prop(Tango::Attribute &att,
                                                py::object& multi_attr_prop)
    {
        long tangoTypeConst = att.get_data_type();
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_NAME(tangoTypeConst, _get_properties_multi_attr_prop, att, multi_attr_prop);
        return multi_attr_prop;
    }

    template<typename TangoScalarType>
    inline void _set_properties_multi_attr_prop(Tango::Attribute &att, py::object& multi_attr_prop)
    {
//        Tango::MultiAttrProp<TangoScalarType> tg_multi_attr_prop;
//        from_py_object(multi_attr_prop,tg_multi_attr_prop);
//        att.set_properties(tg_multi_attr_prop);
    }

    void set_properties_multi_attr_prop(Tango::Attribute& self, py::object& multi_attr_prop)
    {
        long tangoTypeConst = self.get_data_type();
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_NAME(tangoTypeConst, _set_properties_multi_attr_prop, self, multi_attr_prop);
    }

    void set_upd_properties(Tango::Attribute& self, py::object& attr_cfg)
    {
        std::cerr << "set_upd_properties required" << std::endl;
//        Tango::AttributeConfig_3 tg_attr_cfg;
//        from_py_object(attr_cfg, tg_attr_cfg);
//        self.set_upd_properties(tg_attr_cfg);
    }

    void set_upd_properties(Tango::Attribute& self, py::object& attr_cfg, std::string& dev_name)
    {
        std::cerr << "set_upd_properties required2" << std::endl;
//        Tango::AttributeConfig_3 tg_attr_cfg;
//        from_py_object(attr_cfg, tg_attr_cfg);
//        self.set_upd_properties(tg_attr_cfg, dev_name);
    }

    inline void fire_change_event(Tango::Attribute& self)
    {
        self.fire_change_event();
    }

    inline void fire_change_event(Tango::Attribute& self, py::object& data)
    {
//        py::extract<Tango::DevFailed> except_convert(data);
//        if (except_convert.check()) {
//            self.fire_change_event(
//                           const_cast<Tango::DevFailed*>( &except_convert() ));
//            return;
//        }
//        std::stringstream o;
//        o << "Wrong Python argument type for attribute " << self.get_name()
//            << ". Expected a DevFailed." << ends;
//        Tango::Except::throw_exception(
//                "PyDs_WrongPythonDataTypeForAttribute",
//                o.str(),
//                "fire_change_event()");
    }

    template<typename TangoScalarType>
    inline void _set_min_alarm(Tango::Attribute& self, py::object value)
    {
        TangoScalarType c_value = value.cast<TangoScalarType>();
        self.set_min_alarm(c_value);
    }

    template<>
    inline void _set_min_alarm<Tango::DevString>(Tango::Attribute& self, py::object value)
    {
        std::string value_str = value.cast<std::string>();
        self.set_max_warning(value_str.c_str());
    }

    inline void set_min_alarm(Tango::Attribute& self, py::object& value)
    {
        long tangoTypeConst = self.get_data_type();
        // TODO: the below line is a neat trick to properly raise a Tango exception if a property is set
        // for one of the forbidden attribute data types; code dependent on Tango C++ implementation
        if(tangoTypeConst == Tango::DEV_BOOLEAN || tangoTypeConst == Tango::DEV_STATE)
            tangoTypeConst = Tango::DEV_DOUBLE;
        else if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_NAME(tangoTypeConst, _set_min_alarm, self, value);
    }

    template<typename TangoScalarType>
    inline void _set_max_alarm(Tango::Attribute& self, py::object value)
    {
        TangoScalarType c_value = value.cast<TangoScalarType>();
        self.set_max_alarm(c_value);
    }

    template<>
    inline void _set_max_alarm<Tango::DevString>(Tango::Attribute& self, py::object value)
    {
        std::string value_str = value.cast<std::string>();
        self.set_max_warning(value_str.c_str());
    }

    inline void set_max_alarm(Tango::Attribute& self, py::object& value)
    {
        long tangoTypeConst = self.get_data_type();
        // TODO: the below line is a neat trick to properly raise a Tango exception if a property is set
        // for one of the forbidden attribute data types; code dependent on Tango C++ implementation
        if(tangoTypeConst == Tango::DEV_BOOLEAN || tangoTypeConst == Tango::DEV_STATE)
            tangoTypeConst = Tango::DEV_DOUBLE;
        else if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_NAME(tangoTypeConst, _set_max_alarm, self, value);
    }

    template<typename TangoScalarType>
    inline void _set_min_warning(Tango::Attribute& self, py::object value)
    {
        TangoScalarType c_value = value.cast<TangoScalarType>();
        self.set_min_warning(c_value);
    }

    template<>
    inline void _set_min_warning<Tango::DevString>(Tango::Attribute& self, py::object value)
    {
        std::string value_str = value.cast<std::string>();
        self.set_max_warning(value_str.c_str());
    }

    inline void set_min_warning(Tango::Attribute& self, py::object& value)
    {
        long tangoTypeConst = self.get_data_type();
        // TODO: the below line is a neat trick to properly raise a Tango exception if a property is set
        // for one of the forbidden attribute data types; code dependent on Tango C++ implementation
        if(tangoTypeConst == Tango::DEV_BOOLEAN || tangoTypeConst == Tango::DEV_STATE)
            tangoTypeConst = Tango::DEV_DOUBLE;
        else if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_NAME(tangoTypeConst, _set_min_warning, self, value);
    }

    template<typename TangoScalarType>
    inline void _set_max_warning(Tango::Attribute& self, py::object value)
    {
        TangoScalarType c_value = value.cast<TangoScalarType>();
        self.set_max_warning(c_value);
    }

    template<>
    inline void _set_max_warning<Tango::DevString>(Tango::Attribute& self, py::object value)
    {
        std::string value_str = value.cast<std::string>();
        self.set_max_warning(value_str.c_str());
    }

    inline void set_max_warning(Tango::Attribute& self, py::object& value)
    {
        long tangoTypeConst = self.get_data_type();
        // TODO: the below line is a neat trick to properly raise a Tango exception if a property is set
        // for one of the forbidden attribute data types; code dependent on Tango C++ implementation
        if(tangoTypeConst == Tango::DEV_BOOLEAN || tangoTypeConst == Tango::DEV_STATE)
            tangoTypeConst = Tango::DEV_DOUBLE;
        else if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_NAME(tangoTypeConst, _set_max_warning, self, value);
    }

    template<long tangoTypeConst>
    py::object __get_min_alarm(Tango::Attribute &att)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType tg_val;
        att.get_min_alarm(tg_val);
        py::object py_value = py::cast(tg_val);
        return py_value;
    }

    py::object get_min_alarm(Tango::Attribute& att)
    {
        long tangoTypeConst = att.get_data_type();
        if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        py::object min_alarm;
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(tangoTypeConst, min_alarm = __get_min_alarm, att);
        return min_alarm;
    }

    template<long tangoTypeConst>
    py::object __get_max_alarm(Tango::Attribute &att)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType tg_val;
        att.get_max_alarm(tg_val);
        py::object py_value = py::cast(tg_val);
        return py_value;
    }

    py::object get_max_alarm(Tango::Attribute& att)
    {
        long tangoTypeConst = att.get_data_type();
        if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        py::object max_alarm;
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(tangoTypeConst, max_alarm = __get_max_alarm, att);
        return max_alarm;
    }

    template<long tangoTypeConst>
    py::object __get_min_warning(Tango::Attribute &att)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType tg_val;
        att.get_min_warning(tg_val);
        py::object py_value = py::cast(tg_val);
        return py_value;
    }

    py::object get_min_warning(Tango::Attribute& att)
    {
        long tangoTypeConst = att.get_data_type();
        if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        py::object min_warn;
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(tangoTypeConst, min_warn = __get_min_warning, att);
        return min_warn;
    }

    template<long tangoTypeConst>
    py::object __get_max_warning(Tango::Attribute &att)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        TangoScalarType tg_val;
        att.get_max_warning(tg_val);
        py::object py_value = py::cast(tg_val);
        return py_value;
    }

    py::object get_max_warning(Tango::Attribute& att)
    {
        long tangoTypeConst = att.get_data_type();
        if(tangoTypeConst == Tango::DEV_ENCODED)
            tangoTypeConst = Tango::DEV_UCHAR;

        py::object max_warn;
        TANGO_CALL_ON_ATTRIBUTE_DATA_TYPE_ID(tangoTypeConst, max_warn = __get_max_warning, att);
        return max_warn;
    }
};

void export_attribute(py::module &m)
{
    py::enum_<Tango::Attribute::alarm_flags>(m, "alarm_flags")
        .value("min_level", Tango::Attribute::min_level)
        .value("max_level", Tango::Attribute::max_level)
        .value("rds", Tango::Attribute::rds)
        .value("min_warn", Tango::Attribute::min_warn)
        .value("max_warn", Tango::Attribute::max_warn)
        .value("numFlags", Tango::Attribute::numFlags)
    ;
    
    py::class_<Tango::Attribute>(m, "Attribute")
        .def("is_write_associated", [](Tango::Attribute& self) -> bool {
            return self.is_writ_associated();
        })
        .def("is_min_alarm", [](Tango::Attribute& self) -> bool {
            return self.is_min_alarm();
        })
        .def("is_max_alarm", [](Tango::Attribute& self) -> bool {
            return self.is_max_alarm();
        })
        .def("is_min_warning", [](Tango::Attribute& self) -> bool {
            return self.is_min_warning();
        })
        .def("is_max_warning", [](Tango::Attribute& self) -> bool {
            return self.is_max_warning();
        })
        .def("is_rds_alarm", [](Tango::Attribute& self) -> bool {
            return self.is_rds_alarm();
        })
        .def("is_polled", [](Tango::Attribute& self) -> bool {
            return self.is_polled();
        })
        .def("check_alarm", [](Tango::Attribute& self) -> bool {
            return self.check_alarm();
        })
        .def("get_writable", [](Tango::Attribute& self) -> Tango::AttrWriteType {
            return self.get_writable();
        })
        .def("get_name", [](Tango::Attribute& self) -> std::string& {
            return self.get_name();
        })
        .def("get_data_type", [](Tango::Attribute& self) -> long {
            return self.get_data_type();
        })
        .def("get_data_format", [](Tango::Attribute& self) -> Tango::AttrDataFormat {
            return self.get_data_format();
        })
        .def("get_assoc_name", [](Tango::Attribute& self) -> std::string& {
            return self.get_assoc_name();
        })
        .def("get_assoc_ind", [](Tango::Attribute& self) -> long {
            return self.get_assoc_ind();
        })
        .def("set_assoc_ind", [](Tango::Attribute& self, long val) -> void {
            self.set_assoc_ind(val);
        })
        .def("get_date", [](Tango::Attribute& self) -> Tango::TimeVal& {
            return self.get_date();
        })
        .def("set_date",[](Tango::Attribute& self, Tango::TimeVal& new_date) -> void {
            self.set_date(new_date);
        })
        .def("get_label", [](Tango::Attribute& self) -> std::string& {
            return self.get_label();
        })
        .def("get_quality",[](Tango::Attribute& self) -> Tango::AttrQuality& {
            return self.get_quality();
        })
        .def("set_quality", [](Tango::Attribute& self, Tango::AttrQuality& qual, bool send_event) -> void {
            self.set_quality(qual, send_event);
        }, py::arg("qual"), py::arg("send_event")=false)
        .def("get_data_size", [](Tango::Attribute& self) -> long {
            return self.get_data_size();
        })
        .def("get_x", [](Tango::Attribute& self) -> long {
            return self.get_x();
        })
        .def("get_max_dim_x", [](Tango::Attribute& self) -> long {
            return self.get_max_dim_x();
        })
        .def("get_y", [](Tango::Attribute& self) -> long {
            return self.get_y();
        })
        .def("get_max_dim_y", [](Tango::Attribute& self) -> long {
            return self.get_max_dim_y();
        })
        .def("get_polling_period", [](Tango::Attribute& self) -> long {
            return self.get_polling_period();
        })
        .def("set_attr_serial_model", [](Tango::Attribute& self, Tango::AttrSerialModel ser_model) -> void {
            self.set_attr_serial_model(ser_model);
        })
        .def("get_attr_serial_model", [](Tango::Attribute& self) -> Tango::AttrSerialModel {
            return self.get_attr_serial_model();
        })
        .def("set_min_alarm", [](Tango::Attribute& self, py::object& min) -> void {
            PyAttribute::set_min_alarm(self, min);
        })
        .def("set_max_alarm", [](Tango::Attribute& self, py::object& max) -> void {
            PyAttribute::set_max_alarm(self, max);
        })
        .def("set_min_warning", [](Tango::Attribute& self, py::object& min) -> void {
            PyAttribute::set_min_warning(self, min);
        })
        .def("set_max_warning", [](Tango::Attribute& self, py::object& max) -> void {
            PyAttribute::set_max_warning(self, max);
        })
        .def("get_value_flag", [](Tango::Attribute& self) -> bool {
            return self.get_value_flag();
        })
        .def("set_value_flag", [](Tango::Attribute& self, bool flag) -> void {
            self.set_value_flag(flag);
        })
        .def("get_disp_level", [](Tango::Attribute& self) -> Tango::DispLevel {
            return self.get_disp_level();
        })
        .def("change_event_subscribed", [](Tango::Attribute& self) -> bool {
            return self.change_event_subscribed();
        })
        .def("periodic_event_subscribed", [](Tango::Attribute& self) -> bool {
            return self.periodic_event_subscribed();
        })
        .def("archive_event_subscribed", [](Tango::Attribute& self) -> bool {
            return self.archive_event_subscribed();
        })
        .def("quality_event_subscribed", [](Tango::Attribute& self) -> bool {
            return self.quality_event_subscribed();
        })
        .def("user_event_subscribed", [](Tango::Attribute& self) -> bool {
            return self.user_event_subscribed();
        })
        .def("use_notifd_event", [](Tango::Attribute& self) -> bool {
            return self.use_notifd_event();
        })
        .def("use_zmq_event", [](Tango::Attribute& self) -> bool {
            return self.use_zmq_event();
        })
        .def("get_min_alarm", [](Tango::Attribute& self) -> py::object {
            return PyAttribute::get_min_alarm(self);
        })
        .def("get_max_alarm", [](Tango::Attribute& self) -> py::object {
            return PyAttribute::get_max_alarm(self);
        })
        .def("get_min_warning", [](Tango::Attribute& self) -> py::object {
            return PyAttribute::get_min_warning(self);
        })
        .def("get_max_warning", [](Tango::Attribute& self) -> py::object {
            return PyAttribute::get_max_warning(self);
        })
        .def("set_value", [](Tango::Attribute& self, py::object& value) -> void {
            PyAttribute::set_value(self, value);
        })
        .def("set_value", [](Tango::Attribute& self, std::string& str1, std::string& str2) -> void {
            PyAttribute::set_value(self, str1, str2);
        })
        .def("set_value", [](Tango::Attribute& self, std::string& str1, py::object& value) -> void {
            PyAttribute::set_value(self, str1, value);
        })
        .def("set_value", [](Tango::Attribute& self, Tango::EncodedAttribute* value) -> void {
            PyAttribute::set_value(self, value);
        })
        .def("set_value", [](Tango::Attribute& self, py::object& obj, long value) -> void {
            PyAttribute::set_value(self, obj, value);
        })
        .def("set_value", [](Tango::Attribute& self, py::object& obj, long val1, long val2) -> void {
            PyAttribute::set_value(self, obj, val1, val2);
        })
        .def("set_value_date_quality", [](Tango::Attribute& self, py::object& obj, double t, Tango::AttrQuality quality) -> void {
            PyAttribute::set_value_date_quality(self, obj, t, quality);
        })
        .def("set_value_date_quality", [](Tango::Attribute& self, std::string& str1, std::string& str2, double t, Tango::AttrQuality quality) -> void {
            PyAttribute::set_value_date_quality(self, str1, str2, t, quality);
        })
        .def("set_value_date_quality", [](Tango::Attribute& self, std::string& str1, py::object& obj, double t, Tango::AttrQuality quality) -> void {
            PyAttribute::set_value_date_quality(self, str1, obj, t, quality);
        })
        .def("set_value_date_quality", [](Tango::Attribute& self, py::object& obj, double t, Tango::AttrQuality quality, long val) -> void {
            PyAttribute::set_value_date_quality(self, obj, t, quality, val);
        })
        .def("set_value_date_quality", [](Tango::Attribute& self, py::object& obj, double t, Tango::AttrQuality quality, long val1, long val2) -> void {
            PyAttribute::set_value_date_quality(self, obj, t, quality, val1, val2);
        })
        .def("set_change_event", [](Tango::Attribute& self, bool implemented, bool detect) -> void {
            self.set_change_event(implemented, detect);
        }, py::arg("implemented"), py::arg("detect")=true)
        .def("set_archive_event", [](Tango::Attribute& self, bool implemented, bool detect) -> void {
            self.set_archive_event(implemented, detect);
        }, py::arg("implemented"), py::arg("detect")=true)
        .def("is_change_event", [](Tango::Attribute& self) -> bool {
            return self.is_change_event();
        })
        .def("is_check_change_criteria", [](Tango::Attribute& self) -> bool {
            return self.is_check_change_criteria();
        })
        .def("is_archive_event", [](Tango::Attribute& self) -> bool {
            return self.is_archive_event();
        })
        .def("is_check_archive_criteria", [](Tango::Attribute& self) -> bool {
            return self.is_check_archive_criteria();
        })
        .def("set_data_ready_event", [](Tango::Attribute& self, bool implemented) -> void {
            self.set_data_ready_event(implemented);
        })
        .def("is_data_ready_event", [](Tango::Attribute& self) -> bool {
            return self.is_data_ready_event();
        })
        .def("remove_configuration", [](Tango::Attribute& self) -> void {
            self.remove_configuration();
        })
        .def("_get_properties_multi_attr_prop", [](Tango::Attribute& self, py::object& multi_attr_prop) -> py::object {
            return PyAttribute::get_properties_multi_attr_prop(self, multi_attr_prop);
        })
        .def("_set_properties_multi_attr_prop", [](Tango::Attribute& self, py::object& multi_attr_prop) -> void {
            PyAttribute::set_properties_multi_attr_prop(self, multi_attr_prop);
        })
        .def("set_upd_properties", [](Tango::Attribute& self, py::object& attr_cfg) -> void {
            PyAttribute::set_upd_properties(self, attr_cfg);
        })
        .def("set_upd_properties", [](Tango::Attribute& self, py::object& attr_cfg, std::string& dev_name) -> void {
            PyAttribute::set_upd_properties(self, attr_cfg, dev_name);
        })
        .def("fire_change_event", [](Tango::Attribute& self) -> void {
            PyAttribute::fire_change_event(self);
        })
        .def("fire_change_event", [](Tango::Attribute& self, py::object& obj) -> void {
            PyAttribute::fire_change_event(self, obj);
        })
        ;
}
