################################################################################
##
## This file is part of Taurus, a Tango User Interface Library
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
"""

__docformat__ = "restructuredtext"

from utils import document_static_method as __document_static_method
from _PyTango import Except, DevError


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

def __doc_DevError():
    DevError.__doc__ = """
    Structure describing any error resulting from a command execution,
    or an attribute query, with following members:
        - reason : (str)
        - severity : (ErrSeverity) error severty (WARN, ERR, PANIC)
        - desc : (str) error description
        - origin : (str) Tango server method in which the error happened"""

  
def init(doc=True):
    if doc:
        __doc_Except()
        __doc_DevError()