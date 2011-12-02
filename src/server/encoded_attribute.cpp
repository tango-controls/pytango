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

#include "precompiled_header.hpp"
#include <tango.h>
#include "tango_numpy.h"
#include "device_attribute.h"

using namespace boost::python;

const int i = 1;
#define IS_BIGENDIAN() ( (*(char*)&i) == 0 )

namespace PyEncodedAttribute
{

	/// This callback is run to delete char* objects.
    /// It is called by python. The array was associated with an attribute
    /// value object that is not being used anymore.
    /// @param ptr_ The array object.
    /// @param type_ The type of data. We don't need it for now

    static void __ptr_deleter(void * ptr_, void *type_)
    {
        long t = reinterpret_cast<long>(type_);
        if (1 == t)
            delete [] (static_cast<unsigned char*>(ptr_));
        else if (2 == t)
            delete [] (static_cast<unsigned short*>(ptr_));
        else if (4 == t)
			delete [] (static_cast<Tango::DevULong*>(ptr_));
    }
    
    void encode_gray8(Tango::EncodedAttribute &self, object py_value, int w, int h)
    {
        PyObject *py_value_ptr = py_value.ptr();
        unsigned char *buffer = NULL;
        if (PyString_Check(py_value_ptr))
        {
            buffer = reinterpret_cast<unsigned char*>(PyString_AsString(py_value_ptr));
            self.encode_gray8(buffer, w, h);
            return;
        }
#ifndef DISABLE_PYTANGO_NUMPY
        else if (PyArray_Check(py_value_ptr))
        {
            w = PyArray_DIM(py_value_ptr, 1);
            h = PyArray_DIM(py_value_ptr, 0);
            
            buffer = static_cast<unsigned char*>(PyArray_DATA(py_value_ptr));
            self.encode_gray8(buffer, w, h);
            return;
        }
#endif
        // It must be a py sequence
        // we are sure that w and h are given by python (see encoded_attribute.py)
		const int length = w*h;
	    unsigned char *raw_b = new unsigned char[length];
		auto_ptr<unsigned char> b(raw_b);
        buffer = raw_b;
        unsigned char *p = raw_b;
        int w_bytes = w;
        for (long y=0; y<h; ++y)
        {
            PyObject *row = PySequence_GetItem(py_value_ptr, y);
            if (!row) boost::python::throw_error_already_set();
            if (!PySequence_Check(row))
            {
                Py_DECREF(row);
                PyErr_SetString(PyExc_TypeError,
                    "Expected sequence (str, numpy.ndarray, list, tuple or "
                    "bytearray) inside a sequence");
                boost::python::throw_error_already_set();
            }
            // The given object is a sequence of strings were each string is the entire row
            if (PyString_Check(row))
            {
                if (PyString_Size(row) != w_bytes)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                memcpy(p, PyString_AsString(row), w_bytes);
                p += w;
            }
            else
            {
                if (PySequence_Size(row) != w)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                
                for (long x=0; x<w; ++x)
                {
                    PyObject *cell = PySequence_GetItem(row, x);
                    if (!cell)
                    {
                        Py_DECREF(row);
                        boost::python::throw_error_already_set();
                    }
                    if (PyString_Check(cell))
                    {
                        if (PyString_Size(cell) != 1)
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            PyErr_SetString(PyExc_TypeError,
                                "All string items must have length one");
                            boost::python::throw_error_already_set();
                        }
                        char byte = PyString_AsString(cell)[0];
                        *p = byte;
                    }
                    else if (PyInt_Check(cell) || PyLong_Check(cell))
                    {
                        long byte = PyLong_AsLong(cell);
                        if (byte==-1 && PyErr_Occurred())
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            boost::python::throw_error_already_set();
                        }
                        if (byte < 0 || byte > 255)
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            PyErr_SetString(PyExc_TypeError,
                                "int item not in range(256)");
                            boost::python::throw_error_already_set();
                        }
                        *p = (unsigned char)byte;
                        
                    }
                    Py_DECREF(cell);
                    p++;
                }
            }
            Py_DECREF(row);
        }
        self.encode_gray8(buffer, w, h);
    }

    void encode_jpeg_gray8(Tango::EncodedAttribute &self, object py_value, int w, int h, double quality)
    {
        PyObject *py_value_ptr = py_value.ptr();
        unsigned char *buffer = NULL;
        if (PyString_Check(py_value_ptr))
        {
            buffer = reinterpret_cast<unsigned char*>(PyString_AsString(py_value_ptr));
            self.encode_jpeg_gray8(buffer, w, h, quality);
            return;
        }
#ifndef DISABLE_PYTANGO_NUMPY
        else if (PyArray_Check(py_value_ptr))
        {
            w = PyArray_DIM(py_value_ptr, 1);
            h = PyArray_DIM(py_value_ptr, 0);
            
            buffer = static_cast<unsigned char*>(PyArray_DATA(py_value_ptr));
            self.encode_jpeg_gray8(buffer, w, h, quality);
            return;
        }
#endif
        // It must be a py sequence
        // we are sure that w and h are given by python (see encoded_attribute.py)
		const int length = w*h;
        unsigned char *raw_b = new unsigned char[length];
        auto_ptr<unsigned char> b(raw_b);
		buffer = raw_b;
        unsigned char *p = raw_b;
        int w_bytes = w;
        for (long y=0; y<h; ++y)
        {
            PyObject *row = PySequence_GetItem(py_value_ptr, y);
            if (!row) boost::python::throw_error_already_set();
            if (!PySequence_Check(row))
            {
                Py_DECREF(row);
                PyErr_SetString(PyExc_TypeError,
                    "Expected sequence (str, numpy.ndarray, list, tuple or "
                    "bytearray) inside a sequence");
                boost::python::throw_error_already_set();
            }
            // The given object is a sequence of strings were each string is the entire row
            if (PyString_Check(row))
            {
                if (PyString_Size(row) != w_bytes)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                memcpy(p, PyString_AsString(row), w_bytes);
                p += w;
            }
            else
            {
                if (PySequence_Size(row) != w)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                
                for (long x=0; x<w; ++x)
                {
                    PyObject *cell = PySequence_GetItem(row, x);
                    if (!cell)
                    {
                        Py_DECREF(row);
                        boost::python::throw_error_already_set();
                    }
                    if (PyString_Check(cell))
                    {
                        if (PyString_Size(cell) != 1)
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            PyErr_SetString(PyExc_TypeError,
                                "All string items must have length one");
                            boost::python::throw_error_already_set();
                        }
                        char byte = PyString_AsString(cell)[0];
                        *p = byte;
                    }
                    else if (PyInt_Check(cell) || PyLong_Check(cell))
                    {
                        long byte = PyLong_AsLong(cell);
                        if (byte==-1 && PyErr_Occurred())
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            boost::python::throw_error_already_set();
                        }
                        if (byte < 0 || byte > 255)
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            PyErr_SetString(PyExc_TypeError,
                                "int item not in range(256)");
                            boost::python::throw_error_already_set();
                        }
                        *p = (unsigned char)byte;
                        
                    }
                    Py_DECREF(cell);
                    p++;
                }
            }
            Py_DECREF(row);
        }
        self.encode_jpeg_gray8(buffer, w, h, quality);
    }
    
    void encode_gray16(Tango::EncodedAttribute &self, object py_value, int w, int h)
    {
        PyObject *py_value_ptr = py_value.ptr();
        unsigned short *buffer = NULL;
        if (PyString_Check(py_value_ptr))
        {
            buffer = reinterpret_cast<unsigned short*>(PyString_AsString(py_value_ptr));
            self.encode_gray16(buffer, w, h);
            return;
        }
#ifndef DISABLE_PYTANGO_NUMPY
        else if (PyArray_Check(py_value_ptr))
        {
            w = PyArray_DIM(py_value_ptr, 1);
            h = PyArray_DIM(py_value_ptr, 0);
            
            buffer = static_cast<unsigned short*>(PyArray_DATA(py_value_ptr));
            self.encode_gray16(buffer, w, h);
            return;
        }
#endif
        // It must be a py sequence
        // we are sure that w and h are given by python (see encoded_attribute.py)
		const int length = w*h;
        unsigned short *raw_b = new unsigned short[length];
        auto_ptr<unsigned short> b(raw_b);
		buffer = raw_b;
        unsigned short *p = raw_b;
        int w_bytes = 2*w;
        for (long y=0; y<h; ++y)
        {
            PyObject *row = PySequence_GetItem(py_value_ptr, y);
            if (!row) boost::python::throw_error_already_set();
            if (!PySequence_Check(row))
            {
                Py_DECREF(row);
                PyErr_SetString(PyExc_TypeError,
                    "Expected sequence (str, numpy.ndarray, list, tuple or "
                    "bytearray) inside a sequence");
                boost::python::throw_error_already_set();
            }
            // The given object is a sequence of strings were each string is the entire row
            if (PyString_Check(row))
            {
                if (PyString_Size(row) != w_bytes)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                memcpy(p, PyString_AsString(row), w_bytes);
                p += w;
            }
            else
            {
                if (PySequence_Size(row) != w)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                
                for (long x=0; x<w; ++x)
                {
                    PyObject *cell = PySequence_GetItem(row, x);
                    if (!cell)
                    {
                        Py_DECREF(row);
                        boost::python::throw_error_already_set();
                    }
                    if (PyString_Check(cell))
                    {
                        if (PyString_Size(cell) != 2)
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            PyErr_SetString(PyExc_TypeError,
                                "All string items must have length two");
                            boost::python::throw_error_already_set();
                        }
                        unsigned short *word = reinterpret_cast<unsigned short *>(PyString_AsString(cell));
                        *p = *word;
                    }
                    else if (PyInt_Check(cell) || PyLong_Check(cell))
                    {
                        unsigned short word = (unsigned short)PyLong_AsUnsignedLong(cell);
                        if (PyErr_Occurred())
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            boost::python::throw_error_already_set();
                        }
                        *p = word;
                    }
                    else
                    {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            PyErr_SetString(PyExc_TypeError,
                                "Unsupported data type in array element");
                            boost::python::throw_error_already_set();
                    }
                    Py_DECREF(cell);
                    p++;
                }
            }
            Py_DECREF(row);
        }
        self.encode_gray16(buffer, w, h);
    }

    void encode_rgb24(Tango::EncodedAttribute &self, object py_value, int w, int h)
    {
        PyObject *py_value_ptr = py_value.ptr();
        unsigned char *buffer = NULL;
        if (PyString_Check(py_value_ptr))
        {
            buffer = reinterpret_cast<unsigned char*>(PyString_AsString(py_value_ptr));
            self.encode_rgb24(buffer, w, h);
            return;
        }
#ifndef DISABLE_PYTANGO_NUMPY
        else if (PyArray_Check(py_value_ptr))
        {
            buffer = static_cast<unsigned char*>(PyArray_DATA(py_value_ptr));
            self.encode_rgb24(buffer, w, h);
            return;
        }
#endif
        // It must be a py sequence
        // we are sure that w and h are given by python (see encoded_attribute.py)
		const int length = w*h;
        unsigned char *raw_b = new unsigned char[length];
        auto_ptr<unsigned char> b(raw_b);
		buffer = raw_b;
        unsigned char *p = raw_b;
        int w_bytes = 3*w;
        for (long y=0; y<h; ++y)
        {
            PyObject *row = PySequence_GetItem(py_value_ptr, y);
            if (!row) boost::python::throw_error_already_set();
            if (!PySequence_Check(row))
            {
                Py_DECREF(row);
                PyErr_SetString(PyExc_TypeError,
                    "Expected sequence (str, numpy.ndarray, list, tuple or "
                    "bytearray) inside a sequence");
                boost::python::throw_error_already_set();
            }
            // The given object is a sequence of strings were each string is the entire row
            if (PyString_Check(row))
            {
                if (PyString_Size(row) != w_bytes)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                memcpy(p, PyString_AsString(row), w_bytes);
                p += w;
            }
            else
            {
                if (PySequence_Size(row) != w)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                
                for (long x=0; x<w; ++x)
                {
                    PyObject *cell = PySequence_GetItem(row, x);
                    if (!cell)
                    {
                        Py_DECREF(row);
                        boost::python::throw_error_already_set();
                    }
                    if (PyString_Check(cell))
                    {
                        if (PyString_Size(cell) != 3)
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            PyErr_SetString(PyExc_TypeError,
                                "All string items must have length one");
                            boost::python::throw_error_already_set();
                        }
                        char *byte = PyString_AsString(cell);
                        *p = *byte; p++; byte++;
                        *p = *byte; p++; byte++;
                        *p = *byte; p++;
                    }
                    else if (PyInt_Check(cell) || PyLong_Check(cell))
                    {
                        long byte = PyLong_AsLong(cell);
                        if (byte==-1 && PyErr_Occurred())
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            boost::python::throw_error_already_set();
                        }
                        if (IS_BIGENDIAN())
                        {
                            *p = (unsigned char)(byte >> 16) & 0xFF; p++;
                            *p = (unsigned char)(byte >>  8) & 0xFF; p++;
                            *p = (unsigned char)(byte) & 0xFF; p++;
                        }
                        else
                        {
                            *p = (unsigned char)(byte) & 0xFF; p++;
                            *p = (unsigned char)(byte >>  8) & 0xFF; p++;
                            *p = (unsigned char)(byte >> 16) & 0xFF; p++;
                        }
                    }
                    Py_DECREF(cell);
                }
            }
            Py_DECREF(row);
        }
        self.encode_rgb24(buffer, w, h);
    }

    void encode_jpeg_rgb24(Tango::EncodedAttribute &self, object py_value, int w, int h, double quality)
    {
        PyObject *py_value_ptr = py_value.ptr();
        unsigned char *buffer = NULL;
        if (PyString_Check(py_value_ptr))
        {
            buffer = reinterpret_cast<unsigned char*>(PyString_AsString(py_value_ptr));
            self.encode_jpeg_rgb24(buffer, w, h, quality);
            return;
        }
#ifndef DISABLE_PYTANGO_NUMPY
        else if (PyArray_Check(py_value_ptr))
        {
            buffer = static_cast<unsigned char*>(PyArray_DATA(py_value_ptr));
            self.encode_jpeg_rgb24(buffer, w, h, quality);
            return;
        }
#endif
        // It must be a py sequence
        // we are sure that w and h are given by python (see encoded_attribute.py)
		const int length = w*h;
        unsigned char *raw_b = new unsigned char[length];
        auto_ptr<unsigned char> b(raw_b);
		buffer = raw_b;
        unsigned char *p = raw_b;
        int w_bytes = 3*w;
        for (long y=0; y<h; ++y)
        {
            PyObject *row = PySequence_GetItem(py_value_ptr, y);
            if (!row) boost::python::throw_error_already_set();
            if (!PySequence_Check(row))
            {
                Py_DECREF(row);
                PyErr_SetString(PyExc_TypeError,
                    "Expected sequence (str, numpy.ndarray, list, tuple or "
                    "bytearray) inside a sequence");
                boost::python::throw_error_already_set();
            }
            // The given object is a sequence of strings were each string is the entire row
            if (PyString_Check(row))
            {
                if (PyString_Size(row) != w_bytes)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                memcpy(p, PyString_AsString(row), w_bytes);
                p += w;
            }
            else
            {
                if (PySequence_Size(row) != w)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                
                for (long x=0; x<w; ++x)
                {
                    PyObject *cell = PySequence_GetItem(row, x);
                    if (!cell)
                    {
                        Py_DECREF(row);
                        boost::python::throw_error_already_set();
                    }
                    if (PyString_Check(cell))
                    {
                        if (PyString_Size(cell) != 3)
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            PyErr_SetString(PyExc_TypeError,
                                "All string items must have length one");
                            boost::python::throw_error_already_set();
                        }
                        char *byte = PyString_AsString(cell);
                        *p = *byte; p++; byte++;
                        *p = *byte; p++; byte++;
                        *p = *byte; p++;
                    }
                    else if (PyInt_Check(cell) || PyLong_Check(cell))
                    {
                        long byte = PyLong_AsLong(cell);
                        if (byte==-1 && PyErr_Occurred())
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            boost::python::throw_error_already_set();
                        }
                        if (IS_BIGENDIAN())
                        {
                            *p = (unsigned char)(byte >> 16) & 0xFF; p++;
                            *p = (unsigned char)(byte >>  8) & 0xFF; p++;
                            *p = (unsigned char)(byte) & 0xFF; p++;
                        }
                        else
                        {
                            *p = (unsigned char)(byte) & 0xFF; p++;
                            *p = (unsigned char)(byte >>  8) & 0xFF; p++;
                            *p = (unsigned char)(byte >> 16) & 0xFF; p++;
                        }
                    }
                    Py_DECREF(cell);
                }
            }
            Py_DECREF(row);
        }
        self.encode_jpeg_rgb24(buffer, w, h, quality);
    }

    void encode_jpeg_rgb32(Tango::EncodedAttribute &self, object py_value, int w, int h, double quality)
    {
        PyObject *py_value_ptr = py_value.ptr();
        unsigned char *buffer = NULL;
        if (PyString_Check(py_value_ptr))
        {
            buffer = reinterpret_cast<unsigned char*>(PyString_AsString(py_value_ptr));
            self.encode_jpeg_rgb32(buffer, w, h, quality);
            return;
        }
#ifndef DISABLE_PYTANGO_NUMPY
        else if (PyArray_Check(py_value_ptr))
        {
            buffer = static_cast<unsigned char*>(PyArray_DATA(py_value_ptr));
            self.encode_jpeg_rgb32(buffer, w, h, quality);
            return;
        }
#endif
        // It must be a py sequence
        // we are sure that w and h are given by python (see encoded_attribute.py)
		const int length = w*h;
        unsigned char *raw_b = new unsigned char[length];
        auto_ptr<unsigned char> b(raw_b);
		buffer = raw_b;
        unsigned char *p = raw_b;
        int w_bytes = 4*w;
        for (long y=0; y<h; ++y)
        {
            PyObject *row = PySequence_GetItem(py_value_ptr, y);
            if (!row) boost::python::throw_error_already_set();
            if (!PySequence_Check(row))
            {
                Py_DECREF(row);
                PyErr_SetString(PyExc_TypeError,
                    "Expected sequence (str, numpy.ndarray, list, tuple or "
                    "bytearray) inside a sequence");
                boost::python::throw_error_already_set();
            }
            // The given object is a sequence of strings were each string is the entire row
            if (PyString_Check(row))
            {
                if (PyString_Size(row) != w_bytes)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                memcpy(p, PyString_AsString(row), w_bytes);
                p += w;
            }
            else
            {
                if (PySequence_Size(row) != w)
                {
                    Py_DECREF(row);
                    PyErr_SetString(PyExc_TypeError,
                        "All sequences inside a sequence must have same size");
                    boost::python::throw_error_already_set();
                }
                
                for (long x=0; x<w; ++x)
                {
                    PyObject *cell = PySequence_GetItem(row, x);
                    if (!cell)
                    {
                        Py_DECREF(row);
                        boost::python::throw_error_already_set();
                    }
                    if (PyString_Check(cell))
                    {
                        if (PyString_Size(cell) != 3)
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            PyErr_SetString(PyExc_TypeError,
                                "All string items must have length one");
                            boost::python::throw_error_already_set();
                        }
                        char *byte = PyString_AsString(cell);
                        *p = *byte; p++; byte++;
                        *p = *byte; p++; byte++;
                        *p = *byte; p++; byte++;
                        *p = *byte; p++;
                    }
                    else if (PyInt_Check(cell) || PyLong_Check(cell))
                    {
                        long byte = PyLong_AsLong(cell);
                        if (byte==-1 && PyErr_Occurred())
                        {
                            Py_DECREF(row);
                            Py_DECREF(cell);
                            boost::python::throw_error_already_set();
                        }
                        if (IS_BIGENDIAN())
                        {
                            *p = (unsigned char)(byte >> 24) & 0xFF; p++;
                            *p = (unsigned char)(byte >> 16) & 0xFF; p++;
                            *p = (unsigned char)(byte >>  8) & 0xFF; p++;
                            *p = (unsigned char)(byte) & 0xFF; p++;
                        }
                        else
                        {
                            *p = (unsigned char)(byte) & 0xFF; p++;
                            *p = (unsigned char)(byte >>  8) & 0xFF; p++;
                            *p = (unsigned char)(byte >> 16) & 0xFF; p++;
                            *p = (unsigned char)(byte >> 24) & 0xFF; p++;
                        }
                    }
                    Py_DECREF(cell);
                }
            }
            Py_DECREF(row);
        }
        self.encode_jpeg_rgb32(buffer, w, h, quality);
    }

    PyObject *decode_gray8(Tango::EncodedAttribute &self, Tango::DeviceAttribute *attr, PyTango::ExtractAs extract_as)
    {
        unsigned char *buffer;
        int width, height;

        self.decode_gray8(attr, &width, &height, &buffer);
        
        char *ch_ptr = reinterpret_cast<char *>(buffer);
        PyObject *ret = NULL;
        switch (extract_as)
        {
            case PyTango::ExtractAsNumpy:
#ifndef DISABLE_PYTANGO_NUMPY
            {
                npy_intp dims[2] = { height, width };
                ret = PyArray_SimpleNewFromData(2, dims, NPY_UBYTE, ch_ptr);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                // numpy.ndarray() does not own it's memory, so we need to manage it.
                // We can assign a 'base' object that will be informed (decref'd) when
                // the last copy of numpy.ndarray() disappears.
                // PyCObject is intended for that kind of things. It's seen as a
                // black box object from python. We assign him a function to be called
                // when it is deleted -> the function deletes de data.
                PyObject* guard = PyCObject_FromVoidPtrAndDesc(
                    static_cast<void*>(ch_ptr),
                    reinterpret_cast<void*>(1),
                    __ptr_deleter);
                    
                if (!guard)
                {
                    Py_XDECREF(ret);
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                PyArray_BASE(ret) = guard;
                break;
            }
#endif
            case PyTango::ExtractAsString:
            {
                ret = PyTuple_New(3);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                size_t nb_bytes = width*height*sizeof(char);
                PyObject *buffer_str = PyString_FromStringAndSize(ch_ptr, nb_bytes);
                if (!buffer_str)
                {
                    Py_XDECREF(ret);
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                PyTuple_SetItem(ret, 0, PyLong_FromLong(width));
                PyTuple_SetItem(ret, 1, PyLong_FromLong(height));
                PyTuple_SetItem(ret, 2, buffer_str);
                delete [] buffer;
                break;
            }
            case PyTango::ExtractAsTuple:
            {
                ret = PyTuple_New(height);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                for (long y=0; y < height; ++y) 
                {
                    PyObject *row = PyTuple_New(width);
                    if (!row)
                    {
                        Py_XDECREF(ret);
                        delete [] buffer;
                        throw_error_already_set();
                    }
                    for (long x=0; x < width; ++x)
                    {
                        PyTuple_SetItem(row, x, PyString_FromStringAndSize(ch_ptr + y*width+x, 1));
                    }
                    PyTuple_SetItem(ret, y, row);
                }
                
                delete [] buffer;
                break;
            }
            case PyTango::ExtractAsPyTango3:
            case PyTango::ExtractAsList:
            {
                ret = PyList_New(height);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                for (long y=0; y < height; ++y) 
                {
                    PyObject *row = PyList_New(width);
                    if (!row)
                    {
                        Py_XDECREF(ret);
                        delete [] buffer;
                        throw_error_already_set();
                    }
                    for (long x=0; x < width; ++x)
                    {
                        PyList_SetItem(row, x, PyString_FromStringAndSize(ch_ptr + y*width+x, 1));
                    }
                    PyList_SetItem(ret, y, row);
                }
                
                delete [] buffer;
                break;
            }
            default:
            {
                delete [] buffer;
                PyErr_SetString(PyExc_TypeError, "decode only supports "
                    "ExtractAs Numpy, String, Tuple and List");
                boost::python::throw_error_already_set();
                break;
            }
        }
        return ret;
    }
    
    PyObject *decode_gray16(Tango::EncodedAttribute &self, Tango::DeviceAttribute *attr, PyTango::ExtractAs extract_as)
    {
        unsigned short *buffer;
        int width, height;

        self.decode_gray16(attr, &width, &height, &buffer);
        
        unsigned short *ch_ptr = buffer;
        PyObject *ret = NULL;
        switch (extract_as)
        {
            case PyTango::ExtractAsNumpy:
#ifndef DISABLE_PYTANGO_NUMPY
            {
                npy_intp dims[2] = { height, width };
                ret = PyArray_SimpleNewFromData(2, dims, NPY_USHORT, ch_ptr);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                // numpy.ndarray() does not own it's memory, so we need to manage it.
                // We can assign a 'base' object that will be informed (decref'd) when
                // the last copy of numpy.ndarray() disappears.
                // PyCObject is intended for that kind of things. It's seen as a
                // black box object from python. We assign him a function to be called
                // when it is deleted -> the function deletes de data.
                PyObject* guard = PyCObject_FromVoidPtrAndDesc(
                    static_cast<void*>(ch_ptr),
                    reinterpret_cast<void*>(2),
                    __ptr_deleter);
                    
                if (!guard)
                {
                    Py_XDECREF(ret);
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                PyArray_BASE(ret) = guard;
                break;
            }
#endif
            case PyTango::ExtractAsString:
            {
                ret = PyTuple_New(3);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                size_t nb_bytes = width*height*sizeof(unsigned short);
                
                PyObject *buffer_str = PyString_FromStringAndSize(
                    reinterpret_cast<char *>(ch_ptr), nb_bytes);
                delete [] buffer;
                
                if (!buffer_str)
                {
                    Py_XDECREF(ret);
                    throw_error_already_set();
                }
                
                PyTuple_SetItem(ret, 0, PyLong_FromLong(width));
                PyTuple_SetItem(ret, 1, PyLong_FromLong(height));
                PyTuple_SetItem(ret, 2, buffer_str);
                
                break;
            }
            case PyTango::ExtractAsTuple:
            {
                ret = PyTuple_New(height);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                for (long y=0; y < height; ++y) 
                {
                    PyObject *row = PyTuple_New(width);
                    if (!row)
                    {
                        Py_XDECREF(ret);
                        delete [] buffer;
                        throw_error_already_set();
                    }
                    for (long x=0; x < width; ++x)
                    {
                        PyTuple_SetItem(row, x, PyLong_FromUnsignedLong(ch_ptr[y*width+x]));
                    }
                    PyTuple_SetItem(ret, y, row);
                }
                
                delete [] buffer;
                break;
            }
            case PyTango::ExtractAsPyTango3:
            case PyTango::ExtractAsList:
            {
                ret = PyList_New(height);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                for (long y=0; y < height; ++y) 
                {
                    PyObject *row = PyList_New(width);
                    if (!row)
                    {
                        Py_XDECREF(ret);
                        delete [] buffer;
                        throw_error_already_set();
                    }
                    for (long x=0; x < width; ++x)
                    {
                        PyList_SetItem(row, x, PyLong_FromUnsignedLong(ch_ptr[y*width+x]));
                    }
                    PyList_SetItem(ret, y, row);
                }
                
                delete [] buffer;
                break;
            }
            default:
            {
                delete [] buffer;
                PyErr_SetString(PyExc_TypeError, "decode only supports "
                    "ExtractAs Numpy, String, Tuple and List");
                boost::python::throw_error_already_set();
                break;
            }
        }
        return ret;
    }

    PyObject *decode_rgb32(Tango::EncodedAttribute &self, Tango::DeviceAttribute *attr, PyTango::ExtractAs extract_as)
    {
        unsigned char *buffer;
        int width, height;

        self.decode_rgb32(attr, &width, &height, &buffer);

        unsigned char *ch_ptr = buffer;
        PyObject *ret = NULL;
        switch (extract_as)
        {
            case PyTango::ExtractAsNumpy:
#ifndef DISABLE_PYTANGO_NUMPY
            {
                npy_intp dims[2] = { height, width };
                ret = PyArray_SimpleNewFromData(2, dims, NPY_UINT32, ch_ptr);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                // numpy.ndarray() does not own it's memory, so we need to manage it.
                // We can assign a 'base' object that will be informed (decref'd) when
                // the last copy of numpy.ndarray() disappears.
                // PyCObject is intended for that kind of things. It's seen as a
                // black box object from python. We assign him a function to be called
                PyObject* guard = PyCObject_FromVoidPtrAndDesc(
                // when it is deleted -> the function deletes de data.
                    static_cast<void*>(ch_ptr),
                    reinterpret_cast<void*>(2),
                    __ptr_deleter);
                    
                if (!guard)
                {
                    Py_XDECREF(ret);
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                PyArray_BASE(ret) = guard;
                break;
            }
#endif
            case PyTango::ExtractAsString:
            {
                ret = PyTuple_New(3);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                size_t nb_bytes = width*height*4;
                
                PyObject *buffer_str = PyString_FromStringAndSize(
                    reinterpret_cast<char *>(ch_ptr), nb_bytes);
                delete [] buffer;
                
                if (!buffer_str)
                {
                    Py_XDECREF(ret);
                    throw_error_already_set();
                }
                
                PyTuple_SetItem(ret, 0, PyLong_FromLong(width));
                PyTuple_SetItem(ret, 1, PyLong_FromLong(height));
                PyTuple_SetItem(ret, 2, buffer_str);
                
                break;
            }
            case PyTango::ExtractAsTuple:
            {
                ret = PyTuple_New(height);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                for (long y=0; y < height; ++y) 
                {
                    PyObject *row = PyTuple_New(width);
                    if (!row)
                    {
                        Py_XDECREF(ret);
                        delete [] buffer;
                        throw_error_already_set();
                    }
                    for (long x=0; x < width; ++x)
                    {
                        long idx = 4*(y*width+x);
                        // data comes in in big endian format
						Tango::DevULong data;
                        if (IS_BIGENDIAN())
                        {
                            char *p = reinterpret_cast<char *>(&data);
                            *p = ch_ptr[idx++]; ++p;
                            *p = ch_ptr[idx++]; ++p;
                            *p = ch_ptr[idx++]; ++p;
                            *p = ch_ptr[idx];
                        }
                        else
                        {
                            idx +=3;
                            char *p = reinterpret_cast<char *>(&data);
                            *p = ch_ptr[idx--]; ++p;
                            *p = ch_ptr[idx--]; ++p;
                            *p = ch_ptr[idx--]; ++p;
                            *p = ch_ptr[idx];
                        }
                        PyTuple_SetItem(row, x, PyLong_FromUnsignedLong(data));
                    }
                    PyTuple_SetItem(ret, y, row);
                }
                
                delete [] buffer;
                break;
            }
            case PyTango::ExtractAsPyTango3:
            case PyTango::ExtractAsList:
            {
                ret = PyList_New(height);
                if (!ret)
                {
                    delete [] buffer;
                    throw_error_already_set();
                }
                
                for (long y=0; y < height; ++y) 
                {
                    PyObject *row = PyList_New(width);
                    if (!row)
                    {
                        Py_XDECREF(ret);
                        delete [] buffer;
                        throw_error_already_set();
                    }
                    for (long x=0; x < width; ++x)
                    {
                        long idx = 4*(y*width+x);
                        // data comes in in big endian format
						Tango::DevULong data;
                        if (IS_BIGENDIAN())
                        {
                            char *p = reinterpret_cast<char *>(&data);
                            *p = ch_ptr[idx++]; ++p;
                            *p = ch_ptr[idx++]; ++p;
                            *p = ch_ptr[idx++]; ++p;
                            *p = ch_ptr[idx];
                        }
                        else
                        {
                            idx +=3;
                            char *p = reinterpret_cast<char *>(&data);
                            *p = ch_ptr[idx--]; ++p;
                            *p = ch_ptr[idx--]; ++p;
                            *p = ch_ptr[idx--]; ++p;
                            *p = ch_ptr[idx];
                        }
                        PyList_SetItem(row, x, PyLong_FromUnsignedLong(data));
                    }
                    PyList_SetItem(ret, y, row);
                }
                
                delete [] buffer;
                break;
            }
            default:
            {
                delete [] buffer;
                PyErr_SetString(PyExc_TypeError, "decode only supports "
                    "ExtractAs Numpy, String, Tuple and List");
                boost::python::throw_error_already_set();
                break;
            }
        }
        return ret;
    }
    
}

void export_encoded_attribute()
{
    
    class_<Tango::EncodedAttribute, boost::noncopyable> EncodedAttribute(
        "EncodedAttribute",
        init<>());
    
    EncodedAttribute
        .def(init<int, optional<bool> >())
        .def("_encode_gray8", &PyEncodedAttribute::encode_gray8)
        .def("_encode_gray16", &PyEncodedAttribute::encode_gray16)
        .def("_encode_rgb24", &PyEncodedAttribute::encode_rgb24)
        .def("_encode_jpeg_gray8", &PyEncodedAttribute::encode_jpeg_gray8)
        .def("_encode_jpeg_rgb24", &PyEncodedAttribute::encode_jpeg_rgb24)
        .def("_encode_jpeg_rgb32", &PyEncodedAttribute::encode_jpeg_rgb32)
        .def("_decode_gray8", &PyEncodedAttribute::decode_gray8)
        .def("_decode_gray16", &PyEncodedAttribute::decode_gray16)
        .def("_decode_rgb32", &PyEncodedAttribute::decode_rgb32)
    ;
}