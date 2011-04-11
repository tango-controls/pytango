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

import types

""" 
  Enumeration module.
  In C, enums allow you to declare a bunch of constants with unique values,
  without necessarily specifying the actual values (except in cases where you
  need to). Python has an accepted idiom that's fine for very small numbers of
  constants (A, B, C, D = range(4)) but it doesn't scale well to large numbers,
  and it doesn't allow you to specify values for some constants while leaving
  others unspecified. This approach does those things, while verifying that all
  values (specified and unspecified) are unique. Enum values then are attributes
  of an Enumeration class (Volkswagen.BEETLE, Volkswagen.PASSAT, etc.).
"""
    

class Enumeration:
    """ Enumeration class intended to provide the 'enum' feature present in many 
        programming languages.
        Usage:
        car = ThingWithType(Volkswagen.BEETLE)
        print whatkind(car.type, Volkswagen)
        bug = ThingWithType(Insect.BEETLE)
        print whatkind(bug.type, Insect)

        Notice that car's and bug's attributes don't include any of the
        enum machinery, because that machinery is all CLASS attributes and
        not INSTANCE attributes. So you can generate thousands of cars and
        bugs with reckless abandon, never worrying that time or memory will
        be wasted on redundant copies of the enum stuff.

        print car.__dict__
        print bug.__dict__
        pprint.pprint(Volkswagen.__dict__)
        pprint.pprint(Insect.__dict__)
        """
        
    def __init__(self, name, enumList):
        self.__doc__ = name
        lookup = { }
        reverseLookup = { }
        uniqueNames = [ ]
        self._uniqueValues = uniqueValues = [ ]
        self._uniqueId = 0
        for x in enumList:
            if type(x) == types.TupleType:
                x, i = x
                if type(x) != types.StringType:
                    raise EnumException, "enum name is not a string: " + x
                if type(i) != types.IntType:
                    raise EnumException, "enum value is not an integer: " + i
                if x in uniqueNames:
                    raise EnumException, "enum name is not unique: " + x
                if i in uniqueValues:
                    raise EnumException, "enum value is not unique for " + x
                uniqueNames.append(x)
                uniqueValues.append(i)
                lookup[x] = i
                reverseLookup[i] = x
        for x in enumList:
            if type(x) != types.TupleType:
                if type(x) != types.StringType:
                    raise EnumException, "enum name is not a string: " + x
                if x in uniqueNames:
                    raise EnumException, "enum name is not unique: " + x
                uniqueNames.append(x)
                i = self.generateUniqueId()
                uniqueValues.append(i)
                lookup[x] = i
                reverseLookup[i] = x
        self.lookup = lookup
        self.reverseLookup = reverseLookup
   
    def generateUniqueId(self):
        while self._uniqueId in self._uniqueValues:
            self._uniqueId += 1
        n = self._uniqueId
        self._uniqueId += 1
        return n
    
    def __getattr__(self, attr):
        if not self.lookup.has_key(attr):
            raise AttributeError
        return self.lookup[attr]
    
    def whatis(self, value):
        return self.reverseLookup[value]
