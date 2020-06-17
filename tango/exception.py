# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

"""
This is an internal PyTango module.
"""

__all__ = ("exception_init",)

__docformat__ = "restructuredtext"

from .utils import document_static_method as __document_static_method
from ._tango import Except, DevError, ErrSeverity


def __to_dev_failed(exc_type=None, exc_value=None, traceback=None):
    """to_dev_failed(exc_type, exc_value, traceback) -> tango.DevFailed

            Generate a TANGO DevFailed exception.
            The exception is created with a single :class:`~tango.DevError`
            object. A default value *tango.ErrSeverity.ERR* is defined for
            the :class:`~tango.DevError` severity field.

            The parameters are the same as the ones generates by a call to
            :func:`sys.exc_info`.

        Parameters :
            - type : (class)  the exception type of the exception being handled
            - value : (object) exception parameter (its associated value or the
                      second argument to raise, which is always a class instance
                      if the exception type is a class object)
            - traceback : (traceback) traceback object

        Return     : (tango.DevFailed) a tango exception object

        New in PyTango 7.2.1"""
    try:
        Except.throw_python_exception(exc_type, exc_value, traceback)
    except Exception as e:
        return e


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# DevError pickle
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

def __DevError__getinitargs__(self):
    return ()


def __DevError__getstate__(self):
    return self.reason, self.desc, self.origin, int(self.severity)


def __DevError__setstate__(self, state):
    self.reason = state[0]
    self.desc = state[1]
    self.origin = state[2]
    self.severity = ErrSeverity(state[3])


def __init_DevError():
    DevError.__getinitargs__ = __DevError__getinitargs__
    DevError.__getstate__ = __DevError__getstate__
    DevError.__setstate__ = __DevError__setstate__


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
    throw_exception(reason, desc, origin, sever=tango.ErrSeverity.ERR) -> None

            Generate and throw a TANGO DevFailed exception.
            The exception is created with a single :class:`~tango.DevError`
            object. A default value *tango.ErrSeverity.ERR* is defined for
            the :class:`~tango.DevError` severity field.

        Parameters :
            - reason : (str) The exception :class:`~tango.DevError` object reason field
            - desc   : (str) The exception :class:`~tango.DevError` object desc field
            - origin : (str) The exception :class:`~tango.DevError` object origin field
            - sever  : (tango.ErrSeverity) The exception DevError object severity field

        Throws     : DevFailed
    """)

    document_static_method("re_throw_exception", """
    re_throw_exception(ex, reason, desc, origin, sever=tango.ErrSeverity.ERR) -> None

            Re-throw a TANGO :class:`~tango.DevFailed` exception with one more error.
            The exception is re-thrown with one more :class:`~tango.DevError` object.
            A default value *tango.ErrSeverity.ERR* is defined for the new
            :class:`~tango.DevError` severity field.

        Parameters :
            - ex     : (tango.DevFailed) The :class:`~tango.DevFailed` exception
            - reason : (str) The exception :class:`~tango.DevError` object reason field
            - desc   : (str) The exception :class:`~tango.DevError` object desc field
            - origin : (str) The exception :class:`~tango.DevError` object origin field
            - sever  : (tango.ErrSeverity) The exception DevError object severity field

        Throws     : DevFailed
    """)

    document_static_method("print_error_stack", """
    print_error_stack(ex) -> None

            Print all the details of a TANGO error stack.

        Parameters :
            - ex     : (tango.DevErrorList) The error stack reference
    """)

    document_static_method("print_exception", """
    print_exception(ex) -> None

            Print all the details of a TANGO exception.

        Parameters :
            - ex     : (tango.DevFailed) The :class:`~tango.DevFailed` exception
    """)

    document_static_method("throw_python_exception", """
    throw_python_exception(type, value, traceback) -> None

            Generate and throw a TANGO DevFailed exception.
            The exception is created with a single :class:`~tango.DevError`
            object. A default value *tango.ErrSeverity.ERR* is defined for
            the :class:`~tango.DevError` severity field.

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
    """)


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
    __init_DevError()
    if doc:
        __doc_Except()
        __doc_DevError()
