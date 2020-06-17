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

__all__ = ("AttrData",)

__docformat__ = "restructuredtext"

import inspect

from ._tango import Except, CmdArgType, AttrDataFormat, AttrWriteType
from ._tango import DispLevel, UserDefaultAttrProp, UserDefaultFwdAttrProp
from ._tango import Attr, SpectrumAttr, ImageAttr, FwdAttr
from .utils import is_non_str_seq, is_pure_str


class AttrData(object):
    """A helper class that contains the same information one of the items in
    DeviceClass.attr_list but in object form"""

    def __init__(self, name, class_name, attr_info=None):
        self.class_name = class_name
        self.attr_name = name
        self.attr_type = CmdArgType.DevVoid
        self.attr_format = AttrDataFormat.SCALAR
        self.attr_write = AttrWriteType.READ
        self.dim_x = 1
        self.dim_y = 0
        self.display_level = DispLevel.OPERATOR
        self.polling_period = -1
        self.memorized = False
        self.hw_memorized = False
        if name is None:
            self.read_method_name = None
            self.write_method_name = None
            self.is_allowed_name = None
        else:
            self.read_method_name = "read_" + name
            self.write_method_name = "write_" + name
            self.is_allowed_name = "is_" + name + "_allowed"
        self.attr_class = None
        self.attr_args = []
        self.att_prop = None
        self.forward = False
        if attr_info is not None:
            self.from_attr_info(attr_info)

    @classmethod
    def from_dict(cls, attr_dict):
        attr_dict = dict(attr_dict)
        name = attr_dict.pop('name', None)
        class_name = attr_dict.pop('class_name', None)
        self = cls(name, class_name)
        self.build_from_dict(attr_dict)
        return self

    def build_from_dict(self, attr_dict):

        self.forward = attr_dict.pop("forwarded", False)
        if not self.forward:
            self.attr_type = attr_dict.pop('dtype', CmdArgType.DevDouble)
            self.attr_format = attr_dict.pop('dformat', AttrDataFormat.SCALAR)
            self.dim_x = attr_dict.pop('max_dim_x', 1)
            self.dim_y = attr_dict.pop('max_dim_y', 0)
            self.display_level = attr_dict.pop('display_level', DispLevel.OPERATOR)
            self.polling_period = attr_dict.pop('polling_period', -1)
            self.memorized = attr_dict.pop('memorized', False)
            self.hw_memorized = attr_dict.pop('hw_memorized', False)

            is_access_explicit = "access" in attr_dict
            if is_access_explicit:
                self.attr_write = attr_dict.pop('access')
            else:
                # access is defined by which methods were defined
                r_explicit = "fread" in attr_dict or "fget" in attr_dict
                w_explicit = "fwrite" in attr_dict or "fset" in attr_dict
                if r_explicit and w_explicit:
                    self.attr_write = AttrWriteType.READ_WRITE
                elif r_explicit:
                    self.attr_write = AttrWriteType.READ
                elif w_explicit:
                    self.attr_write = AttrWriteType.WRITE
                else:
                    self.attr_write = AttrWriteType.READ

            fread = attr_dict.pop('fget', attr_dict.pop('fread', None))
            if fread is not None:
                if is_pure_str(fread):
                    self.read_method_name = fread
                elif inspect.isroutine(fread):
                    self.read_method_name = fread.__name__
            fwrite = attr_dict.pop('fset', attr_dict.pop('fwrite', None))
            if fwrite is not None:
                if is_pure_str(fwrite):
                    self.write_method_name = fwrite
                elif inspect.isroutine(fwrite):
                    self.write_method_name = fwrite.__name__
            fisallowed = attr_dict.pop('fisallowed', None)
            if fisallowed is not None:
                if is_pure_str(fisallowed):
                    self.is_allowed_name = fisallowed
                elif inspect.isroutine(fisallowed):
                    self.is_allowed_name = fisallowed.__name__
            self.attr_class = attr_dict.pop("klass", self.DftAttrClassMap[self.attr_format])
            self.attr_args.extend((self.attr_name, self.attr_type, self.attr_write))
            if not self.attr_format == AttrDataFormat.SCALAR:
                self.attr_args.append(self.dim_x)
                if not self.attr_format == AttrDataFormat.SPECTRUM:
                    self.attr_args.append(self.dim_y)
        else:
            self.attr_class = FwdAttr
            self.attr_args = [self.name]

        if len(attr_dict):
            if self.forward:
                self.att_prop = self.__create_user_default_fwdattr_prop(attr_dict)
            else:
                self.att_prop = self.__create_user_default_attr_prop(attr_dict)
        return self

    def _set_name(self, name):
        old_name = self.attr_name
        self.attr_name = name
        self.attr_args[0] = name
        if old_name is None:
            if self.read_method_name is None:
                self.read_method_name = "read_" + name
            if self.write_method_name is None:
                self.write_method_name = "write_" + name
            if self.is_allowed_name is None:
                self.is_allowed_name = "is_" + name + "_allowed"

    def __throw_exception(self, msg, meth="create_attribute()"):
        Except.throw_exception("PyDs_WrongAttributeDefinition", msg, meth)

    def __create_user_default_fwdattr_prop(self, extra_info):
        """for internal usage only"""
        p = UserDefaultFwdAttrProp()
        p.set_label(extra_info["label"])
        return p

    def __create_user_default_attr_prop(self, extra_info):
        """for internal usage only"""
        p = UserDefaultAttrProp()

        doc = extra_info.pop('doc', None)
        if doc is not None:
            extra_info['description'] = doc

        for k, v in extra_info.items():
            k_lower = k.lower()
            method_name = "set_%s" % k_lower.replace(' ', '_')
            if hasattr(p, method_name):
                method = getattr(p, method_name)
                if method_name == 'set_enum_labels':
                    method(v)
                else:
                    method(str(v))
            elif k == 'delta_time':
                p.set_delta_t(str(v))
            elif k_lower not in ('display level', 'polling period', 'memorized'):
                msg = "Wrong definition of attribute. " \
                      "The object extra information '%s' " \
                      "is not recognized!" % (k,)
                Except.throw_exception("PyDs_WrongAttributeDefinition", msg,
                                       "create_user_default_attr_prop()")
        return p

    def from_attr_info(self, attr_info):
        name = self.class_name
        attr_name = self.attr_name
        throw_ex = self.__throw_exception
        # check for well defined attribute info

        # check parameter
        if not is_non_str_seq(attr_info):
            throw_ex("Wrong data type for value for describing attribute %s in "
                     "class %s\nMust be a sequence with 1 or 2 elements"
                     % (attr_name, name))

        if len(attr_info) < 1 or len(attr_info) > 2:
            throw_ex("Wrong number of argument for describing attribute %s in "
                     "class %s\nMust be a sequence with 1 or 2 elements"
                     % (attr_name, name))

        extra_info = {}
        if len(attr_info) == 2:
            # attr_info[1] must be a dictionary
            # extra_info = attr_info[1], with all the keys lowercase
            for k, v in attr_info[1].items():
                extra_info[k.lower()] = v

        attr_info = attr_info[0]

        attr_info_len = len(attr_info)
        # check parameter
        if not is_non_str_seq(attr_info) or \
           attr_info_len < 3 or attr_info_len > 5:
            throw_ex("Wrong data type for describing mandatory information for "
                     "attribute %s in class %s\nMust be a sequence with 3, 4 "
                     "or 5 elements" % (attr_name, name))

        # get data type
        try:
            self.attr_type = CmdArgType(attr_info[0])
        except:
            throw_ex("Wrong data type in attribute argument for attribute %s "
                     "in class %s\nAttribute data type (first element in first "
                     "sequence) must be a tango.CmdArgType"
                     % (attr_name, name))

        # get format
        try:
            self.attr_format = AttrDataFormat(attr_info[1])
        except:
            throw_ex("Wrong data format in attribute argument for attribute %s "
                     "in class %s\nAttribute data format (second element in "
                     "first sequence) must be a tango.AttrDataFormat"
                     % (attr_name, name))

        if self.attr_format == AttrDataFormat.SCALAR:
            if attr_info_len != 3:
                throw_ex("Wrong data type in attribute argument for attribute "
                         "%s in class %s\nSequence describing mandatory "
                         "attribute parameters for scalar attribute must have "
                         "3 elements" % (attr_name, name))
        elif self.attr_format == AttrDataFormat.SPECTRUM:
            if attr_info_len != 4:
                throw_ex("Wrong data type in attribute argument for attribute "
                         "%s in class %s\nSequence describing mandatory "
                         "attribute parameters for spectrum attribute must "
                         "have 4 elements" % (attr_name, name))
            try:
                self.dim_x = int(attr_info[3])
            except:
                throw_ex("Wrong data type in attribute argument for attribute "
                         "%s in class %s\n4th element in sequence describing "
                         "mandatory dim_x attribute parameter for spectrum "
                         "attribute must be an integer" % (attr_name, name))
        elif self.attr_format == AttrDataFormat.IMAGE:
            if attr_info_len != 5:
                throw_ex("Wrong data type in attribute argument for attribute "
                         "%s in class %s\nSequence describing mandatory "
                         "attribute parameters for image attribute must have "
                         "5 elements" % (attr_name, name))
            try:
                self.dim_x = int(attr_info[3])
            except:
                throw_ex("Wrong data type in attribute argument for attribute "
                         "%s in class %s\n4th element in sequence describing "
                         "mandatory dim_x attribute parameter for image "
                         "attribute must be an integer" % (attr_name, name))
            try:
                self.dim_y = int(attr_info[4])
            except:
                throw_ex("Wrong data type in attribute argument for attribute "
                         "%s in class %s\n5th element in sequence desribing "
                         "mandatory dim_y attribute parameter for image "
                         "attribute must be an integer" % (attr_name, name))

        # get write type
        try:
            self.attr_write = AttrWriteType(attr_info[2])
        except:
            throw_ex("Wrong data write type in attribute argument for "
                     "attribute %s in class %s\nAttribute write type (third "
                     "element in first sequence) must be a "
                     "tango.AttrWriteType" % (attr_name, name))
        try:
            self.display_level = DispLevel(extra_info.get("display level",
                                                          DispLevel.OPERATOR))
        except:
            throw_ex("Wrong display level in attribute information for "
                     "attribute %s in class %s\nAttribute information for "
                     "display level is not a tango.DispLevel"
                     % (attr_name, name))
        try:
            self.polling_period = int(extra_info.get("polling period", -1))
        except:
            throw_ex("Wrong polling period in attribute information for "
                     "attribute %s in class %s\nAttribute information for "
                     "polling period is not an integer" % (attr_name, name))

        try:
            memorized = extra_info.get("memorized", "false").lower()
        except:
            throw_ex("Wrong memorized value. for attribute %s in class %s."
                     "Allowed valued are the strings \"true\", \"false\" and "
                     "\"true_without_hard_applied\" (case incensitive)")
        if memorized == "true":
            self.memorized = True
            self.hw_memorized = True
        elif memorized == "true_without_hard_applied":
            self.memorized = True
        else:
            self.memorized = False

        if self.attr_type == CmdArgType.DevEnum:
            if 'enum_labels' not in extra_info:
                throw_ex("Missing 'enum_labels' key in attr_list definition "
                         "for enum attribute %s in class %s" % (attr_name, name))
            self.enum_labels = extra_info["enum_labels"]

        self.attr_class = extra_info.get("klass", self.DftAttrClassMap[self.attr_format])
        self.attr_args.extend((self.attr_name, self.attr_type, self.attr_write))
        if not self.attr_format == AttrDataFormat.SCALAR:
            self.attr_args.append(self.dim_x)
            if not self.attr_format == AttrDataFormat.SPECTRUM:
                self.attr_args.append(self.dim_y)

        att_prop = None
        if extra_info:
            att_prop = self.__create_user_default_attr_prop(extra_info)
        self.att_prop = att_prop

    def to_attr(self):
        attr = self.attr_class(*self.attr_args)
        if self.att_prop is not None:
            attr.set_default_properties(self.att_prop)
        attr.set_disp_level(self.display_level)
        if self.memorized:
            attr.set_memorized()
            attr.set_memorized_init(self.hw_memorized)
        if self.polling_period > 0:
            attr.set_polling_period(self.polling_period)
        return attr

    DftAttrClassMap = {AttrDataFormat.SCALAR: Attr,
                       AttrDataFormat.SPECTRUM: SpectrumAttr,
                       AttrDataFormat.IMAGE: ImageAttr}
