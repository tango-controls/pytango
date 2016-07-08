/******************************************************************************
  This file is part of PyTango (http://www.tinyurl.com/PyTango)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

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
        ExtractAsPyTango3,
        ExtractAsNothing
    };
    
    enum ImageFormat {
        RawImage,
        JpegImage
    };

    enum GreenMode {
        GreenModeSynchronous,
        GreenModeFutures,
        GreenModeGevent
    };
}
