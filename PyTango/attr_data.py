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

__all__ = [ "AttrData" ]

__docformat__ = "restructuredtext"

from ._PyTango import Except, CmdArgType, AttrDataFormat, AttrWriteType, \
    DispLevel, UserDefaultAttrProp, Attr, SpectrumAttr, ImageAttr
from .utils import is_non_str_seq


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
        self.read_method_name = "read_%s" % name
        self.write_method_name = "write_%s" % name
        self.is_allowed_name = "is_%s_allowed" % name
        self.attr_class = None
        self.attr_args = []
        self.att_prop = None
        if attr_info is not None:
            self.from_attr_info(attr_info)

    def __throw_exception(self, msg, meth="create_attribute()"):
        Except.throw_exception("PyDs_WrongAttributeDefinition", msg, meth)

    def __create_user_default_attr_prop(self, attr_name, extra_info):
        """for internal usage only"""
        p = UserDefaultAttrProp()
        for k, v in extra_info.items():
            k_lower = k.lower()
            method_name = "set_%s" % k_lower.replace(' ','_')
            if hasattr(p, method_name):
                method = getattr(p, method_name)
                method(str(v))
            elif k == 'delta_time':
                p.set_delta_t(str(v))
            elif not k_lower in ('display level', 'polling period', 'memorized'):
                name = self.get_name()
                msg = "Wrong definition of attribute %s in " \
                      "class %s\nThe object extra information '%s' " \
                      "is not recognized!" % (attr_name, name, k)
                self.__throw_create_attribute_exception(msg)
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
                     "sequence) must be a PyTango.CmdArgType"
                     % (attr_name, name))
        
        # get format
        try:
            self.attr_format = AttrDataFormat(attr_info[1])
        except:
            throw_ex("Wrong data format in attribute argument for attribute %s "
                     "in class %s\nAttribute data format (second element in "
                     "first sequence) must be a PyTango.AttrDataFormat"
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
                         "attribute must be an integer"  % (attr_name, name))
            try:
                self.dim_y = int(attr_info[4])
            except:
                throw_ex("Wrong data type in attribute argument for attribute "
                         "%s in class %s\n5th element in sequence describing "
                         "mandatory dim_y attribute parameter for image "
                         "attribute must be an integer" % (attr_name, name))
        
        #get write type
        try:
            self.attr_write = AttrWriteType(attr_info[2])
        except:
            throw_ex("Wrong data write type in attribute argument for "
                     "attribute %s in class %s\nAttribute write type (third "
                     "element in first sequence) must be a "
                     "PyTango.AttrWriteType" % (attr_name, name))
        try:
            self.display_level = DispLevel(extra_info.get("display level", 
                                                          DispLevel.OPERATOR))
        except:
            throw_ex("Wrong display level in attribute information for "
                     "attribute %s in class %s\nAttribute information for "
                     "display level is not a PyTango.DispLevel"
                     % (attr_name, name))
        try:
            self.polling_period = int(extra_info.get("polling period", -1))
        except:
            throw_ex("Wrong polling period in attribute information for "
                     "attribute %s in class %s\nAttribute information for "
                     "polling period is not an integer" % (attr_name, name))
        
        self.memorized = extra_info.get("memorized", "false")
        if self.memorized == "true":
            self.memorized, self.hw_memorized = True, True
        elif self.memorized == "true_without_hard_applied":
            self.memorized = True
        else:
            self.memorized = False
        
        self.attr_class = extra_info.get("klass", self.DftAttrClassMap[self.attr_format])
        self.attr_args.extend((self.attr_name, self.attr_type, self.attr_write))
        if not self.attr_format == AttrDataFormat.SCALAR:
            self.attr_args.append(self.dim_x)
            if not self.attr_format == AttrDataFormat.SPECTRUM:
                self.attr_args.append(self.dim_y)
                
        att_prop = None
        if extra_info:
            att_prop = self.__create_user_default_attr_prop(attr_name, extra_info)
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
        
    DftAttrClassMap = { AttrDataFormat.SCALAR : Attr,
                        AttrDataFormat.SPECTRUM: SpectrumAttr,
                        AttrDataFormat.IMAGE : ImageAttr }
