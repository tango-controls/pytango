# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2019 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

import tango
import pytest
import numpy as np
from time import sleep
from tango import DbDatum, DbData
from tango import DevState
from tango import AttributeProxy
from tango import CmdArgType

ap = AttributeProxy("sys/tg_test/1/double_scalar")
assert ap.state() == DevState.RUNNING
ap.is_polled() == False
#
# put the new properties
#
# 1. tango.DbDatum - single property data to be inserted
db_datum = DbDatum("property-1")
db_datum.value_string = ["abcdef"]
ap.put_property(db_datum)
# 2. tango.DbData this is the same a 3. as DbData is a List in python terms
# 3. sequence<DbDatum>
db_datum1 = DbDatum("property-2")
db_datum1.value_string = ["ghijk"]
db_datum2 = DbDatum("property-3")
db_datum2.value_string = ["lmnop"]
db_data = [db_datum1, db_datum2]
ap.put_property(db_data)
# 4. dict<str, DbDatum> -
db_datum = DbDatum("property-8")
db_datum.value_string = ["ABCDEF"]
ap.put_property({"property-8": db_datum})
# 5. dict<str, seq<str>>
ap.put_property({"property-4": ["qrstu","vwxyz"]})
# 6. dict<str, obj>
ap.put_property({"property-5": 3.142})
ap.put_property({"property-6": 3142})
ap.put_property({"property-7": True})
ap.put_property({"property-9": -456})
#
# get & test properties
#
# 1. string
db_dict=ap.get_property("property-1")
assert db_dict == {"property-1": ["abcdef"]}
# 1. string + DbDatum
# this seems pointless as output is a dict not a DbDatum
db_dict=ap.get_property("property-2", DbDatum())
assert db_dict == {"property-2": ["ghijk"]}
# 2. sequence<string>
db_dict=ap.get_property(["property-3", "property-4"])
assert db_dict == {"property-3": ["lmnop"], "property-4": ["qrstu","vwxyz"]}
# 3. tango.DbDatum
db_dict=ap.get_property(DbDatum('property-5'))
assert db_dict == {"property-5": ["3.142"]}
db_dict=ap.get_property(DbDatum('property-6'))
assert db_dict == {"property-6": ["3142"]}
db_dict=ap.get_property(DbDatum('property-7'))
assert db_dict == {"property-7": ["True"]}
db_dict=ap.get_property(DbDatum('property-9'))
assert db_dict == {"property-9": ["-456"]}
# 4. tango.DbData this is the same as 5. as DbData is a List in python terms
# 5. sequence<DbDatum>
db_dict=ap.get_property([DbDatum("property-1"), DbDatum("property-2"), DbDatum("property-3")])
assert db_dict == {"property-1": ["abcdef"], "property-2": ["ghijk"], "property-3": ["lmnop"]}
#
# delete properties
#
# 1. string
ap.delete_property("property-1")
db_dict=ap.get_property(DbDatum("property-1"))
assert db_dict == {"property-1": []}
# 2. tango.DbDatum
ap.delete_property(DbDatum("property-2"))
db_dict=ap.get_property(DbDatum("property-2"))
assert db_dict == {"property-2": []}
# 3. tango.DbData this is the same as 5. as DbData is a List in python terms
# 4. sequence<string>
ap.delete_property(["property-5","property-6"])
db_dict=ap.get_property(["property-5", "property-6"])
assert db_dict == {"property-5": [], "property-6": []}
# 5. sequence<DbDatum> [in] - several property data to be deleted
ap.delete_property([DbDatum("property-3"), DbDatum("property-4")])
db_dict=ap.get_property(["property-3", "property-4"])
assert db_dict == {"property-3": [], "property-4": []}
# 6. dict<str, obj>
ap.delete_property({"property-7": [32767]})
db_dict=ap.get_property(DbDatum("property-7"))
assert db_dict == {"property-7": []}
# 7. dict<str, DbDatum> 
ap.delete_property({"property-8": DbDatum("rubbish")})
ap.delete_property({"property-8": DbDatum("rubbish")})

print("passed attribute_proxy tests")