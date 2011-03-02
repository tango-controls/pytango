/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   (copyleft) CELLS / ALBA Synchrotron, Bellaterra, Spain
  
   This is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.
  
   This software is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with this program; if not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

#pragma once

#ifndef DISABLE_PYTANGO_NUMPY

#include <Python.h>

// See "Importing the API" for the why of these weird defines before
// the inclusion of numpy. They are needed so that you can do import_array
// in just one file while using numpy in all the project files.
// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#miscellaneous
// - {
#      define PY_ARRAY_UNIQUE_SYMBOL pytango_ARRAY_API
#      define NO_IMPORT_ARRAY
#      include <numpy/arrayobject.h>
// - }

#include "tgutils.h"

/// @name Conversion from a Tango scalar type name to the numpy equivalent name
/// @{

#   define TANGO_const2numpy(tangoid) tango_name2numpy<tangoid>::value

    template<int N>
    struct tango_name2numpy
    {
        enum { };
    };

#   define DEF_TANGO2NUMPY(tangoid, numpyid) \
        template<> \
        struct tango_name2numpy<tangoid> \
        { \
            enum {value = numpyid}; \
        }

    DEF_TANGO2NUMPY(Tango::DEV_STATE, NPY_UINT32 );
    DEF_TANGO2NUMPY(Tango::DEV_SHORT, NPY_INT16 );
    DEF_TANGO2NUMPY(Tango::DEV_LONG, NPY_INT32 );
    DEF_TANGO2NUMPY(Tango::DEV_DOUBLE, NPY_FLOAT64 );
    DEF_TANGO2NUMPY(Tango::DEV_FLOAT, NPY_FLOAT32 );
    DEF_TANGO2NUMPY(Tango::DEV_BOOLEAN, NPY_BOOL );
    DEF_TANGO2NUMPY(Tango::DEV_USHORT, NPY_UINT16 );
    DEF_TANGO2NUMPY(Tango::DEV_ULONG, NPY_UINT32 );
    // Unassigned Tango::DEV_STRING, mapping to NPY_STRING is not copy-free
    DEF_TANGO2NUMPY(Tango::DEV_UCHAR, NPY_UBYTE );
    // DEF_TANGO2NUMPY(Tango::DEV_CHAR, NPY_BYTE );
    //Unassigned: Tango::DEV_ENCODED
    DEF_TANGO2NUMPY(Tango::DEV_LONG64, NPY_INT64 );
    DEF_TANGO2NUMPY(Tango::DEV_ULONG64, NPY_UINT64 );

/// @name Conversion from a Tango array type name to the scalar numpy name
/// For types like DEVVAR_DOUBLEARRAY. This is ended with ARRAY, except
/// DEVVAR_LONGSTRINGARRAY, DEVVAR_DOUBLESTRINGARRAY and DEVVAR_STRINGARRAY
/// @{

#   define TANGO_const2scalarnumpy(tangoid) tango_name2scalarnumpy<tangoid>::value

    // We can use TANGO_const2scalarconst which gives us the equivalence
    // except for DEVVAR_CHARARRAY, that does not have any Scalar type
    // equivalent.
    template<int N>
    struct tango_name2scalarnumpy
    {
        enum {value = TANGO_const2numpy(TANGO_const2scalarconst(N)) };
    };
    template<>
    struct tango_name2scalarnumpy<Tango::DEVVAR_CHARARRAY>
    {
        enum {value = NPY_UBYTE };
    };
/// @}

#endif // #ifndef DISABLE_PYTANGO_NUMPY
