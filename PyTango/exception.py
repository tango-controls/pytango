################################################################################
##
## This file is part of PyTango, a python binding for Tango
## 
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
## 
## PyTango is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## PyTango is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
## 
## You should have received a copy of the GNU Lesser General Public License
## along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
##
################################################################################

"""
This is an internal PyTango module.
"""

__all__ = ["exception_init"]

__docformat__ = "restructuredtext"

from .utils import document_static_method as __document_static_method
from ._PyTango import Except, DevError

def __to_dev_failed(exc_type=None, exc_value=None, traceback=None):
    """to_dev_failed(exc_type, exc_value, traceback) -> PyTango.DevFailed

            Generate a TANGO DevFailed exception.
            The exception is created with a single :class:`~PyTango.DevError`
            object. A default value *PyTango.ErrSeverity.ERR* is defined for
            the :class:`~PyTango.DevError` severity field.
            
            The parameters are the same as the ones generates by a call to
            :func:`sys.exc_info`.
            
        Parameters :
            - type : (class)  the exception type of the exception being handled
            - value : (object) exception parameter (its associated value or the
                      second argument to raise, which is always a class instance
                      if the exception type is a class object)
            - traceback : (traceback) traceback object
        
        Return     : (PyTango.DevFailed) a tango exception object
        
        New in PyTango 7.2.1"""
    try:
        Except.throw_python_exception(exc_type, exc_value, traceback)
    except Exception as e:
        return e

def __init_Except():
    Except.to_dev_failed = staticmethod(__to_dev_failed)

def __doc_Except():
    def document_static_method(method_name, desc, append=True):
        return __document_static_method(Except, method_name, desc, append)
    
    Except.__doc__ = """
    A containner for the static methods:
    
        - throw_exception
        - re_throw_exception
        - print_exception
        - compare_exception"""
    
    document_static_method("throw_exception", """
    throw_exception(reason, desc, origin, sever=PyTango.ErrSeverity.ERR) -> None

            Generate and throw a TANGO DevFailed exception.
            The exception is created with a single :class:`~PyTango.DevError` 
            object. A default value *PyTango.ErrSeverity.ERR* is defined for 
            the :class:`~PyTango.DevError` severity field.
        
        Parameters :
            - reason : (str) The exception :class:`~PyTango.DevError` object reason field
            - desc   : (str) The exception :class:`~PyTango.DevError` object desc field
            - origin : (str) The exception :class:`~PyTango.DevError` object origin field
            - sever  : (PyTango.ErrSeverity) The exception DevError object severity field

        Throws     : DevFailed
    """ )

    document_static_method("re_throw_exception", """
    re_throw_exception(ex, reason, desc, origin, sever=PyTango.ErrSeverity.ERR) -> None

            Re-throw a TANGO :class:`~PyTango.DevFailed` exception with one more error.
            The exception is re-thrown with one more :class:`~PyTango.DevError` object.
            A default value *PyTango.ErrSeverity.ERR* is defined for the new
            :class:`~PyTango.DevError` severity field.
        
        Parameters :
            - ex     : (PyTango.DevFailed) The :class:`~PyTango.DevFailed` exception
            - reason : (str) The exception :class:`~PyTango.DevError` object reason field
            - desc   : (str) The exception :class:`~PyTango.DevError` object desc field
            - origin : (str) The exception :class:`~PyTango.DevError` object origin field
            - sever  : (PyTango.ErrSeverity) The exception DevError object severity field

        Throws     : DevFailed
    """ )
    
    document_static_method("print_error_stack", """
    print_error_stack(ex) -> None

            Print all the details of a TANGO error stack.
        
        Parameters :
            - ex     : (PyTango.DevErrorList) The error stack reference
    """ )

    document_static_method("print_exception", """
    print_exception(ex) -> None

            Print all the details of a TANGO exception.
        
        Parameters :
            - ex     : (PyTango.DevFailed) The :class:`~PyTango.DevFailed` exception
    """ )
    
    document_static_method("throw_python_exception", """
    throw_python_exception(type, value, traceback) -> None

            Generate and throw a TANGO DevFailed exception.
            The exception is created with a single :class:`~PyTango.DevError`
            object. A default value *PyTango.ErrSeverity.ERR* is defined for
            the :class:`~PyTango.DevError` severity field.
            
            The parameters are the same as the ones generates by a call to
            :func:`sys.exc_info`.
            
        Parameters :
            - type : (class)  the exception type of the exception being handled
            - value : (object) exception parameter (its associated value or the
                      second argument to raise, which is always a class instance
                      if the exception type is a class object)
            - traceback : (traceback) traceback object

        Throws     : DevFailed
        
        New in PyTango 7.2.1
    """ )
    
def __doc_DevError():
    DevError.__doc__ = """
    Structure describing any error resulting from a command execution,
    or an attribute query, with following members:
    
        - reason : (str) reason
        - severity : (ErrSeverity) error severty (WARN, ERR, PANIC)
        - desc : (str) error description
        - origin : (str) Tango server method in which the error happened"""

  
def exception_init(doc=True):
    __init_Except()
    if doc:
        __doc_Except()
        __doc_DevError()