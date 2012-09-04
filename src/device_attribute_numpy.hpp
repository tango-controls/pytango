/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
   
   PyTango is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   PyTango is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/


// This header file is just some template functions moved apart from
// device_attribute.cpp, and should only be included there.

#pragma once

namespace PyDeviceAttribute {

    /// This callback is run to delete Tango::DevVarXArray* objects.
    /// It is called by python. The array was associated with an attribute
    /// value object that is not being used anymore.
    /// @param ptr_ The array object.
    /// @param type_ The type of the array objects. We need it to convert ptr_
    ///              to the proper type before deleting it. ex: Tango::DEV_SHORT.
#ifdef PYCAPSULE_OLD
    template<long type>
    static void _dev_var_x_array_deleter(void * ptr_)
    {
        TANGO_DO_ON_ATTRIBUTE_DATA_TYPE(type,
            delete static_cast<TANGO_const2arraytype(tangoTypeConst)*>(ptr_);
        );
    }
#else
    template<long type>
    static void _dev_var_x_array_deleter(PyObject* obj)
    {
        void * ptr_ = PyCapsule_GetPointer(obj, NULL);
        TANGO_DO_ON_ATTRIBUTE_DATA_TYPE(type,
            delete static_cast<TANGO_const2arraytype(tangoTypeConst)*>(ptr_);
        );
    }
#endif

    template<long tangoTypeConst>
    static inline void _update_array_values(Tango::DeviceAttribute &self, bool isImage, object py_value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        typedef typename TANGO_const2arraytype(tangoTypeConst) TangoArrayType;

        // Extract the actual data from Tango::DeviceAttribute (self)
        TangoArrayType* value_ptr = 0;
        try {
            self >> value_ptr;
        } catch (Tango::DevFailed &e ) {
            if (strcmp(e.errors[0].reason.in(),"API_EmptyDeviceAttribute") != 0)
                throw;
        }

        static const int typenum = TANGO_const2numpy(tangoTypeConst);

        if (value_ptr == 0) {
            // Empty device attribute
            PyObject* value = PyArray_SimpleNew(0, 0, typenum);
            if (!value)
                throw_error_already_set();
            py_value.attr(value_attr_name) = object(handle<>(value));
            py_value.attr(w_value_attr_name) = object();
            return;
        }

        TangoScalarType* buffer = value_ptr->get_buffer();

        npy_intp dims[2];
        int nd = 1;
        size_t write_part_offset = 0;
        if (isImage) {
            nd = 2;
            dims[1] = self.get_dim_x();
            dims[0] = self.get_dim_y();
            write_part_offset = dims[1] * dims[0];
        } else {
            nd = 1;
            dims[0] = self.get_dim_x();
            write_part_offset = dims[0];
        }

        // Create a new numpy.ndarray() object. It uses ch_ptr as the data,
        // so no costy memory copies when handling big images.
        char *ch_ptr = reinterpret_cast<char *>(buffer);

        PyObject* array = PyArray_SimpleNewFromData(nd, dims, typenum, ch_ptr);
        if (!array) {
            delete value_ptr;
            throw_error_already_set();
        }

        // Create the numpy array for the write part. It will be stored in
        // another place.
        PyObject* warray = 0;
        if (self.get_written_dim_x() != 0) {
            if (isImage) {
                nd = 2;
                dims[1] = self.get_written_dim_x();
                dims[0] = self.get_written_dim_y();
            } else {
                nd = 1;
                dims[0] = self.get_written_dim_x();
            }
            char* w_ch_ptr = reinterpret_cast<char *>( buffer + write_part_offset );
            warray = PyArray_SimpleNewFromData(nd, dims, typenum, w_ch_ptr);
            if (!warray) {
                Py_XDECREF(array);
                delete value_ptr;
                throw_error_already_set();
            }
        }

        // numpy.ndarray() does not own it's memory, so we need to manage it.
        // We can assign a 'base' object that will be informed (decref'd) when
        // the last copy of numpy.ndarray() disappears.
        // PyCObject is intended for that kind of things. It's seen as a
        // black box object from python. We assign him a function to be called
        // when it is deleted -> the function deletes the data.
        PyObject* guard = PyCapsule_New(
                static_cast<void*>(value_ptr),
                NULL,
                _dev_var_x_array_deleter<tangoTypeConst>);
        if (!guard ) {
            Py_XDECREF(array);
            Py_XDECREF(warray);
            delete value_ptr;
            throw_error_already_set();
        }
        
        PyArray_BASE(array) = guard;
        py_value.attr(value_attr_name) = boost::python::object(boost::python::handle<>(array));

        // The original C api object storing the data is the same for the
        // read data and the write data. so, both array and warray share
        // the same 'base' (guard). Thus, the data will not be deleted until
        // neither is accessed anymore.
        if (warray) {
            Py_INCREF(guard);
            PyArray_BASE(warray) = guard;
            py_value.attr(w_value_attr_name) = boost::python::object(boost::python::handle<>(warray));
        } else
            py_value.attr(w_value_attr_name) = object();
        // py_value.attr("__internal_data") = object(handle<>(borrowed(guard)));
    }

    template<>
    inline void _update_array_values<Tango::DEV_STRING>(Tango::DeviceAttribute &self, bool isImage, object py_value)
    {
        _update_array_values_as_tuples<Tango::DEV_STRING>(self, isImage, py_value);
    }

    template<>
    inline void _update_array_values<Tango::DEV_ENCODED>(Tango::DeviceAttribute &self, bool isImage, object py_value)
    {
        /// @todo Sure, it is not necessary?
        assert(false);
    }

    // template<long tangoTypeConst>
    // static inline void _update_array_values(PythonDeviceAttribute &self, bool isImage)
    // {
    //     return _update_array_values_numpy<tangoTypeConst>(self, isImage);
    // }

    template<long tangoTypeConst>
    static inline void _fill_numpy_attribute(Tango::DeviceAttribute & dev_attr, const bool isImage, const object & py_value)
    {
        typedef typename TANGO_const2type(tangoTypeConst) TangoScalarType;
        typedef typename TANGO_const2arraytype(tangoTypeConst) TangoArrayType;

        // -- Check dimensions
        unsigned long dim_x=0, dim_y=0, nelems=0;
        bool ok;
        switch (PyArray_NDIM(py_value.ptr())) {
            case 2: // -- Image
                ok = isImage;
                dim_x = PyArray_DIM(py_value.ptr(), 1);
                dim_y = PyArray_DIM(py_value.ptr(), 0);
                nelems = dim_x*dim_y;
                break;
            case 1: // -- Spectrum
                ok = !isImage;
                dim_x = PyArray_DIM(py_value.ptr(), 0);
                dim_y = 0;
                nelems = dim_x;
                break;
            default: // -- WTF?!!?
                ok = false;
                break;
        }
        if (!ok)
            raise_(PyExc_TypeError, isImage? non_valid_image : non_valid_spectrum);

        // -- Allocate memory for the new data object
        unique_pointer<TangoArrayType> value;
        TangoScalarType* buffer = TangoArrayType::allocbuf(nelems);
        try {
            value.reset(new TangoArrayType(nelems, nelems, buffer, true));
        } catch(...) {
            TangoArrayType::freebuf(buffer);
            throw;
        }

        // -- Copy from numpy.array to TangoArrayType...
        PyArrayIterObject *iter;
        PyObject *array = py_value.ptr();
        iter = (PyArrayIterObject *)PyArray_IterNew(array);
        if (!iter)
            throw_error_already_set();
        object iter_guard(handle<>(iter));

        if (isImage) {
            // Why not use PyArray_ITER_NEXT() instead of PyArray_ITER_GOTO()?
            // We could do a single while(iter->index < iter->size) instead
            // of the double "for".
            // I did this and it worked in the sense that it went across
            // the correct number of elements but... I did not know the
            // x and y position it corresponded! Yes, 'iter' has a coordinates
            // field, but it was always [0,0], never updated!!
            unsigned long coordinates[2];
            unsigned long &x = coordinates[1];
            unsigned long &y = coordinates[0];
            for (y=0; y < dim_y; ++y) {
                for (x=0; x < dim_x; ++x) {
                    PyArray_ITER_GOTO(iter, coordinates);

                    PyObject* dataObj = PyArray_GETITEM(array, iter->dataptr);
                    const object py_data = object( handle<>( dataObj ) );

                    buffer[y*dim_x + x] = extract<TangoScalarType>(py_data);
                }
            }
        } else {
            for (unsigned long x=0; x < dim_x; ++x) {
                PyObject* dataObj = PyArray_GETITEM(array, iter->dataptr);
                const object py_data = object( handle<>( dataObj ) );

                buffer[x] = extract<TangoScalarType>(py_data);

                PyArray_ITER_NEXT(iter);
            }
        }

        // -- Insert into device attribute
        dev_attr.insert( value.get(), dim_x, dim_y );

        // -- Final cleaning...
        value.release(); // Do not delete value, it is handled by dev_attr now!
    }

    template<>
    inline void _fill_numpy_attribute<Tango::DEV_ENCODED>(Tango::DeviceAttribute & dev_attr, const bool isImage, const object & py_value)
    {
        // Unsupported
        assert(false);
    }
}
