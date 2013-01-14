CREATE TABLE IF NOT EXISTS access_address (
  user varchar(255) default NULL,
  address varchar(255) default NULL,
  netmask varchar(255) default 'FF.FF.FF.FF',
  updated timestamp NOT NULL,
  accessed timestamp NOT NULL default '00000000000000'
) ;

CREATE TABLE IF NOT EXISTS access_device (
  user varchar(255) default NULL,
  device varchar(255) default NULL,
  rights varchar(255) default NULL,
  updated timestamp NOT NULL,
  accessed timestamp NOT NULL default '00000000000000'
) ;

CREATE TABLE IF NOT EXISTS attribute_alias (
  alias varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  device varchar(255) NOT NULL default '',
  attribute varchar(255) NOT NULL default '',
  updated timestamp NOT NULL,
  accessed timestamp NOT NULL,
  comment text
) ;

CREATE TABLE IF NOT EXISTS attribute_class (
  class varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  updated timestamp NOT NULL,
  accessed timestamp NOT NULL,
  comment text
) ;


CREATE TABLE IF NOT EXISTS device (
  name varchar(255) NOT NULL default 'nada',
  alias varchar(255) default NULL,
  domain varchar(85) NOT NULL default 'nada',
  family varchar(85) NOT NULL default 'nada',
  member varchar(85) NOT NULL default 'nada',
  exported int(11) default 0,
  ior text,
  host varchar(255) NOT NULL default 'nada',
  server varchar(255) NOT NULL default 'nada',
  pid int(11) default 0,
  class varchar(255) NOT NULL default 'nada',
  version varchar(8) NOT NULL default 'nada',
  started datetime default 0,
  stopped datetime default 0,
  comment text
) ;

#
# Table structure for table 'event'
#

CREATE TABLE IF NOT EXISTS event (
  name varchar(255) default NULL,
  exported int(11) default NULL,
  ior text,
  host varchar(255) default NULL,
  server varchar(255) default NULL,
  pid int(11) default NULL,
  version varchar(8) default NULL,
  started datetime default NULL,
  stopped datetime default NULL
) ;

#
# Table structure for table 'property'
#

CREATE TABLE IF NOT EXISTS property (
  object varchar(255) default NULL,
  name varchar(255) default NULL,
  count int(11) default NULL,
  value text default NULL,
  updated timestamp NOT NULL,
  accessed timestamp NOT NULL,
  comment text
) ;

#
# Table structure for table 'property_attribute_class'
#

CREATE TABLE IF NOT EXISTS property_attribute_class (
  class varchar(255) NOT NULL default '',
  attribute varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  count int(11) NOT NULL default '0',
  value text default NULL,
  updated timestamp NOT NULL,
  accessed timestamp NOT NULL,
  comment text
) ;

#
# Table structure for table 'property_attribute_device'
#

CREATE TABLE IF NOT EXISTS property_attribute_device (
  device varchar(255) NOT NULL default '',
  attribute varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  count int(11) NOT NULL default '0',
  value text default NULL,
  updated timestamp NOT NULL,
  accessed timestamp NOT NULL,
  comment text
) ;

#
# Table structure for table 'property_class'
#

CREATE TABLE IF NOT EXISTS property_class (
  class varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  count int(11) NOT NULL default '0',
  value text default NULL,
  updated timestamp NOT NULL,
  accessed timestamp NOT NULL,
  comment text
) ;

#
# Table structure for table 'property_device'
#

CREATE TABLE IF NOT EXISTS property_device (
  device varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  domain varchar(255) NOT NULL default '',
  family varchar(255) NOT NULL default '',
  member varchar(255) NOT NULL default '',  
  count int(11) NOT NULL default '0',
  value text default NULL,
  updated timestamp NOT NULL,
  accessed timestamp NOT NULL,
  comment text
) ;

#
# Table structure for table 'server'
#

CREATE TABLE IF NOT EXISTS server (
  name varchar(255) NOT NULL default '',
  host varchar(255) NOT NULL default '',
  mode int(11) default '0',
  level int(11) default '0'
) ;

#
# Tables for history identifiers
#

CREATE TABLE IF NOT EXISTS device_history_id (
  id int(11) NOT NULL default '0'
) ;

CREATE TABLE IF NOT EXISTS device_attribute_history_id (
  id int(11) NOT NULL default '0'
) ;

CREATE TABLE IF NOT EXISTS class_history_id (
  id int(11) NOT NULL default '0'
) ;

CREATE TABLE IF NOT EXISTS class_attribute_history_id (
  id int(11) NOT NULL default '0'
) ;

CREATE TABLE IF NOT EXISTS object_history_id (
  id int(11) NOT NULL default '0'
) ;

#
# Tables for history
#

CREATE TABLE IF NOT EXISTS property_hist (
  id int(10) NOT NULL default '0',
  date timestamp NOT NULL,
  object varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  count int(11) NOT NULL default '0',
  value text
) ;

CREATE TABLE IF NOT EXISTS property_device_hist (
  id int(10) NOT NULL default '0',
  date timestamp NOT NULL,
  device varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  count int(11) NOT NULL default '0',
  value text
) ;

CREATE TABLE IF NOT EXISTS property_class_hist (
  id int(10) NOT NULL default '0',
  date timestamp NOT NULL,
  class varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  count int(11) NOT NULL default '0',
  value text
) ;

CREATE TABLE IF NOT EXISTS property_attribute_class_hist (
  id int(10) NOT NULL default '0',
  date timestamp NOT NULL,
  class varchar(255) NOT NULL default '',
  attribute varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  count int(11) NOT NULL default '0',
  value text
) ;

CREATE TABLE IF NOT EXISTS property_attribute_device_hist (
  id int(10) NOT NULL default '0',
  date timestamp NOT NULL,
  device varchar(255) NOT NULL default '',
  attribute varchar(255) NOT NULL default '',
  name varchar(255) NOT NULL default '',
  count int(11) NOT NULL default '0',
  value text
) ;

