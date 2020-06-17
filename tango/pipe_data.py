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

from __future__ import with_statement
from __future__ import print_function

__all__ = ("PipeData",)

__docformat__ = "restructuredtext"

import inspect

from ._tango import Except, DispLevel, Pipe, PipeWriteType, UserDefaultPipeProp
from .utils import is_non_str_seq, is_pure_str


class PipeData(object):
    """A helper class that contains the same information one of the items in
    DeviceClass.pipe_list but in object form"""

    def __init__(self, name, class_name, pipe_info=None):
        self.class_name = class_name
        self.pipe_name = name
        self.pipe_write = PipeWriteType.PIPE_READ
        self.display_level = DispLevel.OPERATOR
        if name is None:
            self.read_method_name = None
            self.write_method_name = None
            self.is_allowed_name = None
        else:
            self.read_method_name = "read_" + name
            self.write_method_name = "write_" + name
            self.is_allowed_name = "is_" + name + "_allowed"
        self.pipe_class = None
        self.pipe_args = []
        self.pipe_prop = None
        if pipe_info is not None:
            self.from_pipe_info(pipe_info)

    @classmethod
    def from_dict(cls, pipe_dict):
        pipe_dict = dict(pipe_dict)
        name = pipe_dict.pop('name', None)
        class_name = pipe_dict.pop('class_name', None)
        self = cls(name, class_name)
        self.build_from_dict(pipe_dict)
        return self

    def build_from_dict(self, pipe_dict):
        self.display_level = pipe_dict.pop('display_level', DispLevel.OPERATOR)

        is_access_explicit = "access" in pipe_dict
        if is_access_explicit:
            self.pipe_write = pipe_dict.pop('access')
        else:
            # access is defined by which methods were defined
            r_explicit = "fread" in pipe_dict or "fget" in pipe_dict
            w_explicit = "fwrite" in pipe_dict or "fset" in pipe_dict
            if r_explicit and w_explicit:
                self.pipe_write = PipeWriteType.PIPE_READ_WRITE
            else:
                self.pipe_write = PipeWriteType.PIPE_READ

        fread = pipe_dict.pop('fget', pipe_dict.pop('fread', None))
        if fread is not None:
            if is_pure_str(fread):
                self.read_method_name = fread
            elif inspect.isroutine(fread):
                self.read_method_name = fread.__name__
        fwrite = pipe_dict.pop('fset', pipe_dict.pop('fwrite', None))
        if fwrite is not None:
            if is_pure_str(fwrite):
                self.write_method_name = fwrite
            elif inspect.isroutine(fwrite):
                self.write_method_name = fwrite.__name__
        fisallowed = pipe_dict.pop('fisallowed', None)
        if fisallowed is not None:
            if is_pure_str(fisallowed):
                self.is_allowed_name = fisallowed
            elif inspect.isroutine(fisallowed):
                self.is_allowed_name = fisallowed.__name__
        self.pipe_class = pipe_dict.pop("klass", Pipe)
        self.pipe_args.extend((self.pipe_name, self.display_level, self.pipe_write))
        if len(pipe_dict):
            self.pipe_prop = self.__create_user_default_pipe_prop(pipe_dict)
        return self

    def _set_name(self, name):
        old_name = self.pipe_name
        self.pipe_name = name
        self.pipe_args[0] = name
        if old_name is None:
            if self.read_method_name is None:
                self.read_method_name = "read_" + name
            if self.write_method_name is None:
                self.write_method_name = "write_" + name
            if self.is_allowed_name is None:
                self.is_allowed_name = "is_" + name + "_allowed"

    def __throw_exception(self, msg, meth="create_pipe()"):
        Except.throw_exception("PyDs_WrongPipeDefinition", msg, meth)

    def __create_user_default_pipe_prop(self, extra_info):
        """for internal usage only"""
        p = UserDefaultPipeProp()

        doc = extra_info.pop('doc', None)
        if doc is not None:
            extra_info['description'] = doc

        for k, v in extra_info.items():
            k_lower = k.lower()
            method_name = "set_%s" % k_lower.replace(' ', '_')
            if hasattr(p, method_name):
                method = getattr(p, method_name)
                method(str(v))
            else:
                msg = "Wrong definition of pipe. " \
                      "The object extra information '%s' " \
                      "is not recognized!" % (k,)
                Except.throw_exception("PyDs_WrongPipeDefinition", msg,
                                       "create_user_default_pipe_prop()")
        return p

    def from_pipe_info(self, pipe_info):
        name = self.class_name
        pipe_name = self.pipe_name
        throw_ex = self.__throw_exception
        # check for well defined pipe info

        # check parameter
        if not is_non_str_seq(pipe_info):
            throw_ex("Wrong data type for value for describing pipe %s in "
                     "class %s\nMust be a sequence with 1 or 2 elements"
                     % (pipe_name, name))

        if len(pipe_info) < 1 or len(pipe_info) > 2:
            throw_ex("Wrong number of argument for describing pipe %s in "
                     "class %s\nMust be a sequence with 1 or 2 elements"
                     % (pipe_name, name))

        extra_info = {}
        if len(pipe_info) == 2:
            # pipe_info[1] must be a dictionary
            # extra_info = pipe_info[1], with all the keys lowercase
            for k, v in pipe_info[1].items():
                extra_info[k.lower()] = v

        pipe_info = pipe_info[0]

        # get write type
        try:
            self.pipe_write = PipeWriteType(pipe_info)
        except:
            throw_ex("Wrong data write type in pipe argument for "
                     "pipe %s in class %s\nPipe write type must be a "
                     "tango.PipeWriteType" % (pipe_name, name))
        try:
            self.display_level = DispLevel(extra_info.get("display level",
                                                          DispLevel.OPERATOR))
        except:
            throw_ex("Wrong display level in pipe information for "
                     "pipe %s in class %s\nPipe information for "
                     "display level is not a tango.DispLevel"
                     % (pipe_name, name))

        self.pipe_class = extra_info.get("klass", Pipe)
        self.pipe_args.extend((self.pipe_name, self.display_level, self.pipe_write))

        pipe_prop = None
        if extra_info:
            pipe_prop = self.__create_user_default_pipe_prop(extra_info)
        self.pipe_prop = pipe_prop

    def to_pipe(self):
        pipe = self.pipe_class(*self.pipe_args)
        if self.pipe_prop is not None:
            pipe.set_default_properties(self.pipe_prop)
        return pipe
