#
# Create all database tables
#

source create_db_tables.sql

#
# Init the history identifiers
#

INSERT INTO device_history_id VALUES (0);
INSERT INTO device_attribute_history_id VALUES (0);
INSERT INTO class_history_id VALUES (0);
INSERT INTO class_attribute_history_id VALUES (0);
INSERT INTO object_history_id VALUES (0);

#
# Create entry for database device server in device table
#

INSERT INTO device VALUES ('sys/database/2',NULL,'sys','database','2','nada','nada','nada','DataBaseds/2','nada','DataBase','nada','nada','nada','nada');
INSERT INTO device VALUES ('dserver/DataBaseds/2',NULL,'dserver','DataBaseds','2','nada','nada','nada','DataBaseds/2','nada','DServer','nada','nada','nada','nada');
INSERT INTO server VALUES ('databaseds/2','',0,0);

INSERT INTO device VALUES ('sys/database/pydb-test',NULL,'sys','database','pydb-test','nada','nada','nada','DataBaseds/pydb-test','nada','DataBase','nada','nada','nada','nada');
INSERT INTO device VALUES ('dserver/DataBaseds/pydb-test',NULL,'dserver','DataBaseds','pydb-test','nada','nada','nada','DataBaseds/pydb-test','nada','DServer','nada','nada','nada','nada');
INSERT INTO server VALUES ('databaseds/pydb-test','',0,0);

#
# Create entry for test device server in device table
#

INSERT INTO device VALUES ('sys/tg_test/1',NULL,'sys','tg_test','1','nada','nada','nada','TangoTest/test','nada','TangoTest','nada','nada','nada','nada');
INSERT INTO device VALUES ('dserver/TangoTest/test',NULL,'dserver','TangoTest','test','nada','nada','nada','TangoTest/test','nada','DServer','nada','nada','nada','nada');
INSERT INTO server VALUES ('tangotest/test','',0,0);

#
# Create entry for Tango Control Access in device table
#

INSERT INTO device VALUES ('sys/access_control/1',NULL,'sys','access_control','1','nada','nada','nada','TangoAccessControl/1','nada','TangoAccessControl','nada','nada','nada','nada');
INSERT INTO device VALUES ('dserver/TangoAccessControl/1',NULL,'dserver','TangoAccessControl','1','nada','nada','nada','TangoAccessControl/1','nada','DServer','nada','nada','nada','nada');
INSERT INTO server VALUES ('tangoaccesscontrol/1','',0,0);

#
# Create default user access
#

INSERT INTO access_address VALUES ('*','*.*.*.*','FF.FF.FF.FF',20060824131221,00000000000000);
INSERT INTO access_device VALUES ('*','*/*/*','write',20060824131221,00000000000000);

#
# Create entries in the property_class tables for controlled access service
#

INSERT INTO property_class VALUES('Database','AllowedAccessCmd',1,'DbGetServerInfo','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',2,'DbGetServerNameList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',3,'DbGetInstanceNameList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',4,'DbGetDeviceServerClassList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',5,'DbGetDeviceList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',6,'DbGetDeviceDomainList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',7,'DbGetDeviceFamilyList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',8,'DbGetDeviceMemberList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',9,'DbGetClassList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',10,'DbGetDeviceAliasList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',11,'DbGetObjectList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',12,'DbGetPropertyList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',13,'DbGetProperty','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',14,'DbGetClassPropertyList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',15,'DbGetClassProperty','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',16,'DbGetDevicePropertyList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',17,'DbGetDeviceProperty','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',18,'DbGetClassAttributeList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',19,'DbGetDeviceAttributeProperty','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',20,'DbGetDeviceAttributeProperty2','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',21,'DbGetLoggingLevel','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',22,'DbGetAliasDevice','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',23,'DbGetClassForDevice','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',24,'DbGetClassInheritanceForDevice','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',25,'DbGetDataForServerCache','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',26,'DbInfo','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',27,'DbGetClassAttributeProperty','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',28,'DbGetClassAttributeProperty2','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',29,'DbMysqlSelect','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',30,'DbGetDeviceInfo','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',31,'DbGetDeviceWideList','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',32,'DbImportEvent','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',33,'DbGetDeviceAlias','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('Database','AllowedAccessCmd',34,'DbGetCSDbServerList','1980-01-01 ','1980-01-01 ',NULL);

#
#
#

INSERT INTO property_class VALUES('DServer','AllowedAccessCmd',1,'QueryClass','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('DServer','AllowedAccessCmd',2,'QueryDevice','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('DServer','AllowedAccessCmd',3,'EventSubscriptionChange','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('DServer','AllowedAccessCmd',4,'DevPollStatus','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('DServer','AllowedAccessCmd',5,'GetLoggingLevel','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('DServer','AllowedAccessCmd',6,'GetLoggingTarget','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('DServer','AllowedAccessCmd',7,'QueryWizardDevProperty','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('DServer','AllowedAccessCmd',8,'QueryWizardClassProperty','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES('DServer','AllowedAccessCmd',9,'QuerySubDevice','1980-01-01 ','1980-01-01 ',NULL);

#
#
#

INSERT INTO property_class VALUES ('Starter','AllowedAccessCmd',1,'DevReadLog','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES ('Starter','AllowedAccessCmd',2,'DevStart','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES ('Starter','AllowedAccessCmd',3,'DevGetRunningServers','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES ('Starter','AllowedAccessCmd',4,'DevGetStopServers','1980-01-01 ','1980-01-01 ',NULL);

#
#
#

INSERT INTO property_class VALUES ('TangoAccessControl','AllowedAccessCmd',1,'GetUsers','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES ('TangoAccessControl','AllowedAccessCmd',2,'GetAddressByUser','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES ('TangoAccessControl','AllowedAccessCmd',3,'GetDeviceByUser','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES ('TangoAccessControl','AllowedAccessCmd',4,'GetAccess','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES ('TangoAccessControl','AllowedAccessCmd',5,'GetAllowedCommands','1980-01-01 ','1980-01-01 ',NULL);
INSERT INTO property_class VALUES ('TangoAccessControl','AllowedAccessCmd',6,'GetAllowedCommandClassList','1980-01-01 ','1980-01-01 ',NULL);

#
# Load the stored procedures
#

source stored_proc.sql

