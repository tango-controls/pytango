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

__all__ = ('PipeConfig',)

from ._tango import Pipe, PipeWriteType, UserDefaultPipeProp, \
    CmdArgType, DevState, DispLevel, constants

from .utils import scalar_to_array_type, TO_TANGO_TYPE, \
    is_non_str_seq, is_pure_str, is_integer, is_number
from .utils import document_method as __document_method


class PipeConfig(object):
    """
    This class represents the python interface for the Tango IDL
    object PipeConfig."""

    def __init__(self):
        self.name = ''
        self.description = ''
        self.label = ''
        self.level = DispLevel.OPERATOR
        self.writable = PipeWriteType.PIPE_READ
        self.extensions = []


def __get_pipe_type_simple(obj):
    if is_non_str_seq(obj):
        if len(obj) == 2 and \
                is_pure_str(obj[0]) and \
                (is_non_str_seq(obj[1]) or isinstance(obj[1], dict)):
            tg_type = CmdArgType.DevPipeBlob
        else:
            tg_type = __get_pipe_type(obj[0])
            tg_type = scalar_to_array_type(tg_type)
    elif is_pure_str(obj):
        tg_type = CmdArgType.DevString
    elif isinstance(obj, DevState):
        tg_type = CmdArgType.DevState
    elif isinstance(obj, bool):
        tg_type = CmdArgType.DevBoolean
    elif is_integer(obj):
        tg_type = CmdArgType.DevLong64
    elif is_number(obj):
        tg_type = CmdArgType.DevDouble
    else:
        raise ValueError('Cannot determine object tango type')
    return tg_type


def __get_pipe_type_numpy_support(obj):
    try:
        ndim, dtype = obj.ndim, str(obj.dtype)
    except AttributeError:
        return __get_pipe_type_simple(obj)
    if ndim > 1:
        raise TypeError('cannot translate numpy array with {0} '
                        'dimensions to tango type'.format(obj.ndim))
    tg_type = TO_TANGO_TYPE[dtype]
    if ndim > 0:
        tg_type = scalar_to_array_type(dtype)
    return tg_type


def __get_tango_type(dtype):
    if is_non_str_seq(dtype):
        tg_type = dtype[0]
        if is_non_str_seq(tg_type):
            raise TypeError("Pipe doesn't support 2D data")
        tg_type = TO_TANGO_TYPE[tg_type]
        tg_type = scalar_to_array_type(tg_type)
    else:
        tg_type = TO_TANGO_TYPE[dtype]
    return tg_type


def __get_pipe_type(obj, dtype=None):
    if dtype is not None:
        return __get_tango_type(dtype)
    if constants.NUMPY_SUPPORT:
        return __get_pipe_type_numpy_support(obj)
    return __get_pipe_type_simple(obj)


def __sanatize_pipe_element(elem):
    if isinstance(elem, dict):
        result = dict(elem)
    else:
        result = dict(name=elem[0], value=elem[1])
    result['value'] = value = result.get('value', result.pop('blob', None))
    result['dtype'] = dtype = __get_pipe_type(value, dtype=result.get('dtype'))
    if dtype == CmdArgType.DevPipeBlob:
        result['value'] = value[0], __sanatize_pipe_blob(value[1])
    return result


def __sanatize_pipe_blob(blob):
    if isinstance(blob, dict):
        return [__sanatize_pipe_element((k, v)) for k, v in blob.items()]
    else:
        return [__sanatize_pipe_element(elem) for elem in blob]


def __Pipe__set_value(self, value):
    """
    Set the pipe value. Check ref:`pipe data types <pytango-pipe-data-types>`:
    for more information on which values are supported
    """
    root_blob_name, blob = value
    self.set_root_blob_name(root_blob_name)
    value = __sanatize_pipe_blob(blob)
    self._set_value(value)


def __Pipe__get_value(self):
    return (self.get_root_blob_name(), self._get_value())


def __init_Pipe():
    Pipe.set_value = __Pipe__set_value


def __doc_UserDefaultPipeProp():
    def document_method(method_name, desc, append=True):
        return __document_method(UserDefaultPipeProp, method_name, desc, append)

    UserDefaultPipeProp.__doc__ = """
    User class to set pipe default properties.
    This class is used to set pipe default properties.
    Three levels of pipe properties setting are implemented within Tango.
    The highest property setting level is the database.
    Then the user default (set using this UserDefaultPipeProp class) and finally
    a Tango library default value
    """

    document_method("set_label", """
    set_label(self, def_label) -> None

            Set default label property.

        Parameters :
            - def_label : (str) the user default label property
        Return     : None
    """)

    document_method("set_description", """
    set_description(self, def_description) -> None

            Set default description property.

        Parameters :
            - def_description : (str) the user default description property
        Return     : None
    """)


def pipe_init(doc=True):
    __init_Pipe()
    if doc:
        __doc_UserDefaultPipeProp()
