/******************************************************************************
  This file is part of PyTango (http://pytango.rtfd.io)

  Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
  Copyright 2013-2019 European Synchrotron Radiation Facility, Grenoble, France

  Distributed under the terms of the GNU Lesser General Public License,
  either version 3 of the License, or (at your option) any later version.
  See LICENSE.txt for more info.
******************************************************************************/

#ifndef _PYTGUTILS_H_
#define _PYTGUTILS_H_

#include <tango.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/// Get the python Global Interpret Lock
class AutoPythonGIL
{
    /**
     * Check python. Before acquiring python GIL check if python as not been
     * shutdown. If this is the case then the best we can do here is throw an
     * exception to try to prevent the PyTango from calling python code
     **/
    inline void check_python()
    {
        if (!Py_IsInitialized())
        {
            Tango::Except::throw_exception(
                "AutoPythonGIL_PythonShutdown",
                "Trying to execute python code when python interpreter as shutdown.",
                "AutoPythonGIL::check_python");
        }
    }

public:
    inline AutoPythonGIL(bool safe=true) 
    { 
        if (safe) check_python();
            // Acquire GIL before calling Python code
            py::gil_scoped_acquire acquire;
            std::cout << "Acquired GIL before calling Python code" << std::endl;
    }
    
    inline ~AutoPythonGIL() {
        py::gil_scoped_release release;
    }
    
};

class AutoPythonGILEnsure
{
    PyGILState_STATE m_gstate;

public:
    inline AutoPythonGILEnsure()
    {
        // When threads are created from tango c++ they don’t hold the GIL,
        // nor is there a thread state structure for them.
        // So ensure that the current thread is ready to call the Python C API
        // regardless of the current state of Python, or of the global interpreter lock.
        m_gstate = PyGILState_Ensure();
        std::cerr << "Ensured GIL before calling Python code" << std::endl;
    }


    inline ~AutoPythonGILEnsure() {
        PyGILState_Release(m_gstate);
        std::cerr << "Released GIL after calling Python code" << std::endl;
    }
};

/**
 * Translate a double into a timeval structure
 *
 * @param[out] tv timeval structure to be filled with the time
 * @param[in] t a double representing the time
 */
inline void double2timeval(struct timeval &tv, double t)
{
    double sec = floor(t);
#ifdef WIN32
    tv.tv_usec = (long)((t-sec)*1.0E6);
    tv.tv_sec = (long)(sec);
#else
    tv.tv_usec = (time_t)((t-sec)*1.0E6);
    tv.tv_sec = (suseconds_t)(sec);
#endif
}
#endif //_PYTGUTILS_H_
