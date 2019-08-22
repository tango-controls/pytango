/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

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
        GreenModeSynchronous = 0,
        GreenModeFutures,
        GreenModeGevent,
        GreenModeAsyncio
    };
}
