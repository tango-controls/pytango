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

#pragma once

#include <string>
#include <vector>

typedef std::vector<std::string> StdStringVector;
typedef std::vector<long> StdLongVector;
typedef std::vector<double> StdDoubleVector;

/* HAS_UNIQUE_PTR definition comes from tango. It tells PyTango if Tango itself
   is using the new C++0x unique_ptr or not. In PyTango we try to use the same
   as TangoC++ whenever possible. */
#ifdef PYTANGO_HAS_UNIQUE_PTR
#define unique_pointer std::unique_ptr
#else
#define unique_pointer std::auto_ptr
#endif

/*
#ifdef HAS_UNIQUE_PTR
typedef std::unique_ptr unique_pointer;
#else
typedef std::auto_ptr unique_pointer;
#endif
*/

namespace PyTango
{
    enum ExtractAs {
        ExtractAsNumpy,
        ExtractAsByteArray,
        ExtractAsBytes,
        ExtractAsTuple,
        ExtractAsList,
        ExtractAsString,
        ExtractAsNothing
    };
    
    enum ImageFormat {
        RawImage,
        JpegImage
    };
}
