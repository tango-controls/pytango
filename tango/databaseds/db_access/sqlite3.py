from __future__ import print_function

import os
import logging
import functools
import threading
import Queue

import tango

th_exc = tango.Except.throw_exception

from db_errors import *

from concurrent.futures import ThreadPoolExecutor

Executor = ThreadPoolExecutor(1)

def get_create_db_statements():
    statements = []
    with open("create_db_tables.sql") as f:
        lines = f.readlines()
    # strip comments
    lines = (line for line in lines if not line.startswith('#'))
    lines = (line for line in lines if not line.lower().strip().startswith('key'))
    lines = (line for line in lines if not line.lower().strip().startswith('key'))
    lines = "".join(lines)
    lines = lines.replace("ENGINE=MyISAM", "")
    statements += lines.split(";")

    with open("create_db.sql") as f:
        lines = f.readlines()
    # strip comments
    lines = (line for line in lines if not line.lower().startswith('#'))
    lines = (line for line in lines if not line.lower().startswith('create database'))
    lines = (line for line in lines if not line.lower().startswith('use'))
    lines = (line for line in lines if not line.lower().startswith('source'))
    lines = "".join(lines)
    statements += lines.split(";")

    return statements

def replace_wildcard(text):
    # escape '%' with '\'
    text = text.replace("%", "\\%")
    # escape '_' with '\'
    text = text.replace("_", "\\_")
    # escape '"' with '\'
    text = text.replace('"', '\\"')
    # escape ''' with '\'
    text = text.replace("'", "\\'")
    # replace '*' with '%'
    text = text.replace("*", "%")
    return text

def use_cursor(f):
    @functools.wraps(f)
    def wrap(*args, **kwargs):
        self = args[0]
        has_cursor = 'cursor' in kwargs
        cursor = kwargs.pop('cursor', None)
        if not has_cursor:
            cursor = Executor.submit(self.get_cursor).result()
        self.cursor = cursor
        try:
            ret = Executor.submit(f, *args, **kwargs).result()
            if not has_cursor:
                Executor.submit(cursor.connection.commit).result()
            return ret
        finally:
            if not has_cursor:
                Executor.submit(cursor.close).result()
                del self.cursor
    return wrap


class Tango_dbapi2(object):

    DB_API_NAME = 'sqlite3'

    def __init__(self, db_name="tango_database.db", history_depth=10, fire_to_starter=True):
        self._db_api = None
        self._db_conn = None
        self.db_name = db_name
        self.history_depth = history_depth
        self.fire_to_starter = fire_to_starter
        self._logger = logging.getLogger(self.__class__.__name__)
        self._debug = self._logger.debug
        self._info = self._logger.info
        self._warn = self._logger.warn
        self._error = self._logger.error
        self._critical = self._logger.critical
        self.initialize()

    def close_db(self):
        if self._db_conn is not None:
            self._db_conn.commit()
            self._db_conn.close()
        self._db_api = None
        self._db_conn = None

    def get_db_api(self):
        if self._db_api is None:
            self._db_api = __import__(self.DB_API_NAME)
        return self._db_api

    @property
    def db_api(self):
        return self.get_db_api()

    @property
    def db_conn(self):
        if self._db_conn is None:
            self._db_conn = self.db_api.connect(self.db_name)
        return self._db_conn

    def get_cursor(self):
        return self.db_conn.cursor()

    def initialize(self):
        self._info("Initializing database...")
        if not os.path.isfile(self.db_name):
            self.create_db()
        else:
            # trigger connection
            self.db_conn

    @use_cursor
    def create_db(self):
        self._info("Creating database...")
        statements = get_create_db_statements()
        cursor = self.cursor
        for statement in statements:
            cursor.execute(statement)

    @use_cursor
    def get_id(self, name):
        cursor = self.cursor
        name += '"_history_id'
        _id = cursor.execute('SELECT id FROM ?', (name,)).fetchone()[0] + 1
        cursor.execute('UPDATE ? SET id=?', (name, _id))
        return _id

    @use_cursor
    def purge_att_property(self, table, field, obj, attr, name):
        cursor = self.cursor
        cursor.execute(\
            'SELECT DISTINCT id FROM ? WHERE ? = ? AND name = ? AND ' \
            'attribute = ? ORDER BY date', (table, field, obj, name, attr))
        rows = cursor.fetchall()
        to_del = len(rows) - self.history_depth
        if to_del > 0:
            for row in rows[:to_del]:
                cursor.execute('DELETE FROM ? WHERE id=?', (table, row[0]))

    @use_cursor
    def purge_property(self, table, field, obj, name):
        cursor = self.cursor
        cursor.execute(\
            'SELECT DISTINCT id FROM ? WHERE ? = ? AND name = ? ' \
            'ORDER BY date', (table, field, obj, name))
        rows = cursor.fetchall()
        to_del = len(rows) - self.history_depth
        if to_del > 0:
            for row in rows[:to_del]:
                cursor.execute('DELETE FROM ? WHERE id=?', (table, row[0]))

    @use_cursor
    def get_device_host(self, name):
        cursor = self.cursor
        name = replace_wildcard(name)
        cursor.execute('SELECT host FROM device WHERE name LIKE ?', (name,))
        row = cursor.fetchone()
        if row is None:
            raise Exception("No host for device '" + name + "'")
        else:
            return row[0]


    def send_starter_cmd(self, starter_dev_names):
        for name in starter_dev_names:
            pos = name.find('.')
            if pos != -1:
                name = name[0:pos]
            dev = tango.DeviceProxy(name)
            dev.UpdateServersInfo()
            
            

    # TANGO API

    def get_stored_procedure_release(self):
        return 'release 1.8'

    @use_cursor
    def add_device(self, server_name, dev_info, klass_name, alias=None):
        self._info("delete_attribute_alias(server_name=%s, dev_info=%s, klass_name=%s, alias=%s)",
                   server_name, dev_info, klass_name, alias)
        dev_name, (domain, family, member) = dev_info
        cursor = self.cursor

        # first delete the tuple (device,name) from the device table
        cursor.execute('DELETE FROM device WHERE name LIKE ?', (dev_name,))

        # then insert the new value for this tuple
        cursor.execute(\
            'INSERT INTO device (name, alias, domain, family, member, exported, ' \
            'ior, host, server, pid, class, version, started, stopped) ' \
            'VALUES (?, ?, ?, ?, ?, 0, "nada", "nada", ?, 0, ?, "0", NULL, NULL)',
            (dev_name, alias, domain, family, member, server_name, klass_name))

        # Check if a DServer device entry for the process already exists
        cursor.execute('SELECT name FROM device WHERE server LIKE ? AND class LIKE "DServer"', (server_name,))
        if cursor.fetchone() is None:
            dev_name = "dserver/" + server_name
            domain, family, member = dev_name.split("/", 2)
            cursor.execute(\
            'INSERT INTO device (name, domain, family, member, exported, ior, ' \
            'host, server, pid, class, version, started, stopped) ' \
            'VALUES (?, ?, ?, ?, 0, "nada", "nada", ?, 0, "DServer", "0", NULL, NULL)',
            (dev_name, domain, family, member, server_name))

    @use_cursor
    def delete_attribute_alias(self, alias):
        self._info("delete_attribute_alias(alias=%s)", alias)
        self.cursor.execute('DELETE FROM attribute_alias WHERE alias=?', (alias,))

    @use_cursor
    def delete_class_attribute(self, klass_name, attr_name):
        self.cursor.execute(\
            'DELETE FROM property_attribute_class WHERE class LIKE ? AND ' \
            'attribute LIKE ?', (klass_name, attr_name))

    @use_cursor
    def delete_class_attribute_property(self, klass_name, attr_name, prop_name):
        cursor = self.cursor

        # Is there something to delete ?
        cursor.execute(\
            'SELECT count(*) FROM property_attribute_class WHERE class = ? ' \
            'AND attribute = ? AND name = ?', (klass_name, attr_name, prop_name))
        if cursor.fetchone()[0] > 0:
            # then delete property from the property_attribute_class table
            cursor.execute(\
                'DELETE FROM property_attribute_class WHERE class = ? AND ' \
                'attribute = ? and name = ?', (klass_name, attr_name, prop_name))
            # mark this property as deleted
            hist_id = self.get_id('class_attribute', cursor=cursor)
            cursor.execute(\
                'INSERT INTO property_attribute_class_hist (class, attribute, ' \
                'name, id, count, value) VALUES ' \
                '(?, ?, ?, ?, "0", "DELETED")',
                (klass_name, attr_name, prop_name, hist_id))
            self.purge_att_property("property_attribute_class_hist", "class",
                                    klass_name, attr_name, prop_name, cursor=cursor)

    @use_cursor
    def delete_class_property(self, klass_name, prop_name):
        cursor = self.cursor

        prop_name = replace_wildcard(prop_name)
        # Is there something to delete ?
        cursor.execute(\
            'SELECT DISTINCT name FROM property_class WHERE class=? AND ' \
            'name LIKE ?', (klass_name, prop_name))
        for row in cursor.fetchall():
            # delete the tuple (device,name,count) from the property table
            name = row[0]
            cursor.execute(\
                'DELETE FROM property_class WHERE class=? AND name=?',
                (klass_name, name))
            # Mark this property as deleted
            hist_id = self.get_id("class", cursor=cursor)
            cursor.execute(\
                'INSERT INTO property_class_hist (class, name, id, count, value) ' \
                'VALUES (?, ?, ?, "0", "DELETED")',
                (klass_name, name, hist_id))
            self.purge_property("property_class_hist", "class", klass_name,
                                name, cursor=cursor)

    @use_cursor
    def delete_device(self, dev_name):
        self._info("delete_device(dev_name=%s)", dev_name)
        cursor = self.cursor
        dev_name = replace_wildcard(dev_name)

        # delete the device from the device table
        cursor.execute('DELETE FROM device WHERE name LIKE ?', (dev_name,))

        # delete device from the property_device table
        cursor.execute('DELETE FROM property_device WHERE device LIKE ?', (dev_name,))

        # delete device from the property_attribute_device table
        cursor.execute('DELETE FROM property_attribute_device WHERE device LIKE ?', (dev_name,))

    @use_cursor
    def delete_device_alias(self, dev_alias):
        self._info("delete_device_alias(dev_alias=%s)", dev_alias)
        self.cursor.execute('UPDATE device SET alias=NULL WHERE alias=?', (dev_alias,))

    @use_cursor
    def delete_device_attribute(self, dev_name, attr_name):
        dev_name = replace_wildcard(dev_name)
        self.cursor.execute(\
            'DELETE FROM property_attribute_device WHERE device LIKE ? AND ' \
            'attribute LIKE ?', (dev_name, attr_name))

    @use_cursor
    def delete_device_attribute_property(self, dev_name, attr_name, prop_name):
        cursor = self.cursor
        # Is there something to delete ?
        cursor.execute(\
            'SELECT count(*) FROM property_attribute_device WHERE device = ?' \
            'AND attribute = ? AND name = ?', (dev_name, attr_name, prop_name))
        if cursor.fetchone()[0] > 0:
            # delete property from the property_attribute_device table
            cursor.execute(\
                'DELETE FROM property_attribute_device WHERE device = ? AND '
                'attribute = ? AND name = ?', (dev_name, attr_name, prop_name))
            # Mark this property as deleted
            hist_id = self.get_id("device_attribute", cursor=cursor)
            cursor.execute(\
                'INSERT INTO property_attribute_device_hist ' \
                '(device, attribute, name, id, count, value) VALUES ' \
                '(?, ?, ?, ?, "0", "DELETED")', (dev_name, attr_name, prop_name, hist_id))
            self.purge_att_property("property_attribute_device_hist", "device",
                                    dev_name, attr_name, prop_name, cursor=cursor)

    @use_cursor
    def delete_device_property(self, dev_name, prop_name):
        cursor = self.cursor
        prop_name = replace_wildcard(prop_name)

        # Is there something to delete ?
        cursor.execute(\
            'SELECT DISTINCT name FROM property_device WHERE device=? AND ' \
            'name LIKE ?', (dev_name, prop_name))
        for row in cursor.fetchall():
            # delete the tuple (device,name,count) from the property table
            cursor.execute(\
                'DELETE FROM property_device WHERE device=? AND name LIKE ?',
                (dev_name, prop_name))
            # Mark this property as deleted
            hist_id = self.get_id("device", cursor=cursor)
            cursor.execute(\
                'INSERT INTO property_device_hist (device, id, name, count, value) ' \
                'VALUES (?, ?, ?, "0", "DELETED")', (dev_name, hist_id, row[0]))
            self.purge_property("property_device_hist", "device", dev_name, row[0])

    @use_cursor
    def delete_property(self, obj_name, prop_name):
        cursor = self.cursor
        prop_name = replace_wildcard(prop_name)

        # Is there something to delete ?
        cursor.execute(\
            'SELECT DISTINCT name FROM property WHERE object=? AND ' \
            'name LIKE ?', (obj_name, prop_name))
        for row in cursor.fetchall():
            # delete the tuple (object,name,count) from the property table
            cursor.execute(\
                'DELETE FROM property_device WHERE device=? AND name LIKE ?',
                (obj_name, prop_name))
            # Mark this property as deleted
            hist_id = self.get_id("object", cursor=cursor)
            cursor.execute(\
                'INSERT INTO property_hist (object, name, id, count, value) ' \
                'VALUES (?, ?, ?, "0", "DELETED")', (obj_name, row[0], hist_id))
            self.purge_property("property_hist", "object", obj_name, row[0])

    @use_cursor
    def delete_server(self, server_instance):
        cursor = self.cursor
        server_instance = replace_wildcard(server_instance)

        previous_host = None
        # get host where running
        if self.fire_to_starter:
            adm_dev_name = "dserver/" + server_instance
            previous_host = self.get_device_host(adm_dev_name)

        # then delete the device from the device table
        cursor.execute('DELETE FROM device WHERE server LIKE ?', (server_instance,))

        # Update host's starter to update controlled servers list
        if self.fire_to_starter and previous_host:
            self.send_starter_cmd(previous_host)
            pass

    @use_cursor
    def delete_server_info(self, server_instance):
        self.cursor.execute('DELETE FROM server WHERE name=?', (server_instance,))

    @use_cursor
    def export_device(self, dev_name, IOR, host, pid, version):
        self._info("export_device(dev_name=%s, host=%s, pid=%s, version=%s)",
                   dev_name, host, pid, version)
        self._info("export_device(IOR=%s)", IOR)
        cursor = self.cursor
        do_fire = False
        previous_host = None
        
        if self.fire_to_starter:
            if dev_name[0:8] == "dserver/":
                # Get database server name
                tango_util = tango.Util.instance()
                db_serv = tango_util.get_ds_name()
                adm_dev_name = "dserver/" + db_serv.lower()
                if dev_name != adm_dev_name and dev_name[0:16] != "dserver/starter/":
                    do_fire = True
                    previous_host = self.get_device_host(dev_name)

        cursor.execute('SELECT server FROM device WHERE name LIKE ?', (dev_name,))
        row = cursor.fetchone()
        if row is None:
            th_exc(DB_DeviceNotDefined,
                   "device " + dev_name + " not defined in the database !",
                   "DataBase::ExportDevice()")
        server = row[0]

        # update the new value for this tuple
        cursor.execute(\
            'UPDATE device SET exported=1, ior=?, host=?, pid=?, version=?, ' \
            'started=datetime("now") WHERE name LIKE ?',
            (IOR, host, pid, version, dev_name))

        # update host name in server table
        cursor.execute('UPDATE server SET host=? WHERE name LIKE ?', (host, server))

        if do_fire:
            hosts = []
            hosts.append(host)
            if previous_host != "" and previous_host != "nada" and previous_host != host:
                hosts.append(previous_host)
            self.send_starter_cmd(hosts)

    @use_cursor
    def export_event(self, event, IOR, host, pid, version):
        cursor = self.cursor
        cursor.execute(\
            'INSERT event (name,exported,ior,host,server,pid,version,started) ' \
            'VALUES (?, 1, ?, ?, ?, ?, ?, datetime("now")',
            (event, IOR, host, event, pid, version))

    @use_cursor
    def get_alias_device(self, dev_alias):
        cursor = self.cursor
        cursor.execute('SELECT name FROM device WHERE alias LIKE ?',
                       (dev_alias,))
        row = cursor.fetchone()
        if row is None:
            th_exc(DB_DeviceNotDefined,
                   "No device found for alias '" + dev_alias + "'",
                   "DataBase::GetAliasDevice()")
        return row[0]

    @use_cursor
    def get_attribute_alias(self, attr_alias):
        cursor = self.cursor
        cursor.execute('SELECT name from attribute_alias WHERE alias LIKE ?',
                       (attr_alias,))
        row = cursor.fetchone()
        if row is None:
            th_exc(DB_SQLError,
                   "No attribute found for alias '" + attr_alias + "'",
                   "DataBase::GetAttributeAlias()")
        return row[0]

    @use_cursor
    def get_attribute_alias_list(self, attr_alias):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT alias FROM attribute_alias WHERE alias LIKE ? ORDER BY attribute',
                       (attr_alias,))
        return [ row[0] for row in cursor.fetchall() ]

    @use_cursor
    def get_class_attribute_list(self, class_name, wildcard):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT attribute FROM property_attribute_class WHERE class=? and attribute like ?',
                       (class_name, wildcard))
        return [ row[0] for row in cursor.fetchall() ]

    @use_cursor
    def get_class_attribute_property(self, class_name, attributes):
        cursor = self.cursor
        stmt = 'SELECT name,value FROM property_attribute_class WHERE class=? AND attribute LIKE ?'
        result = [class_name, str(len(attributes))]
        for attribute in attributes:
            cursor.execute(stmt, (class_name, attribute))
            rows = cursor.fetchall()
            result.append(attribute)
            result.append(str(len(rows)))
            for row in rows:
                result.append(row[0])
                result.append(row[1])
        return result

    @use_cursor
    def get_class_attribute_property2(self, class_name, attributes):
        cursor = self.cursor
        stmt = 'SELECT name,value FROM property_attribute_class WHERE class=? AND attribute LIKE ? ORDER BY name,count'
        result = [class_name, str(len(attributes))]
        for attribute in attributes:
            cursor.execute(stmt, (class_name, attribute))
            rows = cursor.fetchall()
            result.append(attribute) 
            j = 0
            new_prop = True
            nb_props = 0
            prop_size = 0
            prop_names = []
            prop_sizes = []
            prop_values = []
            for row in rows:
                prop_values.append(row[1])
                if j == 0:
                    old_name = row[0]
                else:
                    name = row[0]
                    if name != old_name:
                        new_prop = True
                        old_name = name
                    else:
                        new_prop = False
                j  = j + 1
                if new_prop == True:
                    nb_props = nb_props + 1
                    prop_names.append(row[0])
                    if prop_size != 0:
                        prop_sizes.append(prop_size)
                    prop_size = 1
                else:
                    prop_size = prop_size + 1
                    
            result.append(str(nb_props))
            j = 0
            k = 0
            for name in prop_names:
                result.append(name)
                result.append(prop_sizes[j])
                for i in range(0, prop_sizes[j]):
                    result.append(prop_values[k])
                    k = k + 1
                j = j + 1
        return result

    @use_cursor
    def get_class_attribute_property_hist(self, class_name, attribute, prop_name):
        cursor = self.cursor
        stmt = 'SELECT  DISTINCT id FROM property_attribute_class_hist WHERE class=? AND attribute LIKE ? AND name LIKE ? ORDER by date ASC'
        
        result = []
        
        cursor.execute(stmt, (class_name, attribute, prop_name))
        
        for row in cursor.fetchall():
            idr = row[0]
        
            stmt = 'SELECT DATE_FORMAT(date,\'%Y-%m-%d %H:%i:%s\'),value,attribute,name,count FROM property_attribute_class_hist WHERE id =? AND class =?'
            
            cursor.execute(stmt, (idr, class_name))
        
            rows = cursor.fetchall()
        
            result.append(rows[2])
            result.append(rows[3])
            result.append(rows[0])
            result.append(str(rows[4]))
            for value in rows[1]:
                result.append(value)

        return result

    @use_cursor
    def get_class_for_device(self, dev_name):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT class FROM device WHERE name=?', (dev_name,))
        row = cursor.fetchone()
        if row is None:
            th_exc(DB_IncorrectArguments, "Class not found for " + dev_name,
                   "Database.GetClassForDevice")
        return row

    @use_cursor
    def get_class_inheritance_for_device(self, dev_name):
        cursor = self.cursor
        class_name = self.get_class_for_device(dev_name, cursor=cursor)
        props = self.get_class_property(class_name, "InheritedFrom", cursor=cursor)
        return [class_name] + props[4:]

    @use_cursor
    def get_class_list(self, server):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT class FROM device WHERE class LIKE ? ORDER BY class', (server,))
        return [ row[0] for row in cursor.fetchall() ]


    @use_cursor
    def get_class_property(self, class_name, properties):        
        cursor = self.cursor
        stmt = 'SELECT count,value FROM property_class WHERE class=? AND name LIKE ? ORDER BY count'
        result.append(class_name)
        result.append(len(properties))
        for prop_name in properties:
            cursor.execute(stmt, (class_name, prop_name))
            rows = cursor.fetchall()
            result.append(prop_name)
            result.append(str(len(rows)))
            for row in rows:
                result.append(row[1])
        return result
        
    @use_cursor
    def get_class_property_hist(self, class_name, prop_name):
        cursor = self.cursor
        stmt = 'SELECT  DISTINCT id FROM property_class_hist WHERE class=? AND AND name LIKE ? ORDER by date ASC'
        
        result = []
        
        cursor.execute(stmt, (class_name, prop_name))
        
        for row in cursor.fetchall():
            idr = row[0]
        
            stmt = 'SELECT DATE_FORMAT(date,\'%Y-%m-%d %H:%i:%s\'),value,name,count FROM property_class_hist WHERE id =? AND class =?'
            
            cursor.execute(stmt, (idr, class_name))
        
            rows = cursor.fetchall()
        
            result.append(rows[2])
            result.append(rows[0])
            result.append(str(rows[3]))
            for value in rows[1]:
                result.append(value)

        return result      
        
    @use_cursor
    def get_class_property_list(self, class_name):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT name FROM property_class WHERE class LIKE ? order by NAME',
                       (class_name,))
        return [ row[0] for row in cursor.fetchall() ]

    @use_cursor
    def get_device_alias(self, dev_name):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT alias FROM device WHERE name LIKE ?',
                       (dev_name,))
        row = cursor.fetchone()
        if row is None:
            th_exc(DB_DeviceNotDefined,
                   "No alias found for device '" + dev_name + "'",
                   "DataBase::GetDeviceAlias()")
        return row[0]

    @use_cursor
    def get_device_alias_list(self, alias):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT alias FROM device WHERE alias LIKE ? ORDER BY alias',
                       (alias,))
        return [ row[0] for row in cursor.fetchall() ]

    @use_cursor
    def get_device_attribute_list(self, dev_name, attribute):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT  attribute FROM property_attribute_device WHERE device=?  AND attribute LIKE ? ORDER BY attribute',
                       (dev_name, attribute,))
        return [ row[0] for row in cursor.fetchall() ]


    @use_cursor
    def get_device_attribute_property(self, dev_name, attributes):
        cursor = self.cursor
        stmt = 'SELECT name,value FROM property_attribute_device WHERE device=? AND attribute LIKE ?'
        result = [dev_name, str(len(attributes))]
        for attribute in attributes:
            cursor.execute(stmt, (dev_name, attribute))
            rows = cursor.fetchall()
            result.append(attribute)
            result.append(str(len(rows)))
            for row in rows:
                result.append(row[0])
                result.append(row[1])
        return result  

    @use_cursor
    def get_device_attribute_property2(self, dev_name, attributes):
        cursor = self.cursor
        stmt = 'SELECT name,value FROM property_attribute_device WHERE device=? AND attribute LIKE ? ORDER BY name,count' 
        result = [dev_name, str(len(attributes))]
        for attribute in attributes:
            cursor.execute(stmt, (dev_name, attribute))
            rows = cursor.fetchall()
            result.append(attribute) 
            j = 0
            new_prop = True
            nb_props = 0
            prop_size = 0
            prop_names = []
            prop_sizes = []
            prop_values = []
            for row in rows:
                prop_values.append(row[1])
                if j == 0:
                    old_name = row[0]
                else:
                    name = row[0]
                    if name != old_name:
                        new_prop = True
                        old_name = name
                    else:
                        new_prop = False
                j  = j + 1
                if new_prop == True:
                    nb_props = nb_props + 1
                    prop_names.append(row[0])
                    if prop_size != 0:
                        prop_sizes.append(prop_size)
                    prop_size = 1
                else:
                    prop_size = prop_size + 1
                    
            result.append(str(nb_props))
            j = 0
            k = 0
            for name in prop_names:
                result.append(name)
                result.append(prop_sizes[j])
                for i in range(0, prop_sizes[j]):
                    result.append(prop_values[k])
                    k = k + 1
                j = j + 1
        return result

    @use_cursor
    def get_device_attribute_property_hist(self, dev_name, attribute, prop_name):
        cursor = self.cursor
        stmt = 'SELECT  DISTINCT id FROM property_attribute_device_hist WHERE device=? AND attribute LIKE ? AND name LIKE ? ORDER by date ASC'
        
        result = []
        
        cursor.execute(stmt, (dev_name, attribute, prop_name))
        
        for row in cursor.fetchall():
            idr = row[0]
        
            stmt = 'SELECT DATE_FORMAT(date,\'%Y-%m-%d %H:%i:%s\'),value,attribute,name,count FROM property_attribute_device_hist WHERE id =? AND device =? ORDER BY count ASC'
            
            cursor.execute(stmt, (idr, class_name))
        
            rows = cursor.fetchall()
        
            result.append(rows[2])
            result.append(rows[3])
            result.append(rows[0])
            result.append(str(rows[4]))
            for value in rows[1]:
                result.append(value)

        return result

    
    @use_cursor
    def get_device_class_list(self, server_name):
        cursor = self.cursor
        result = []
        cursor.execute('SELECT name,class FROM device WHERE server =?  ORDER BY name',
                       (server_name,))
        for row in cursor.fetchall():
            result.append(row[0])
            result.append(row[1])
        
        return result

    @use_cursor
    def get_device_domain_list(self, wildcard):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT domain FROM device WHERE name LIKE ? OR alias LIKE ? ORDER BY domain',
                       (wildcard,wildcard))
        return [ row[0] for row in cursor.fetchall() ]

   
    @use_cursor
    def get_device_exported_list(self, wildcard):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT name FROM device WHERE (name LIKE ? OR alias LIKE ?) AND exported=1 ORDER BY name',
                       (wildcard,wildcard))
        return [ row[0] for row in cursor.fetchall() ]   

    @use_cursor
    def get_device_family_list(self, wildcard):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT family FROM device WHERE name LIKE ? OR alias LIKE ? ORDER BY family',
                       (wildcard,wildcard))
        return [ row[0] for row in cursor.fetchall() ]   
    
    @use_cursor
    def get_device_info(self, dev_name):
        cursor = self.cursor
        cursor.execute('SELECT exported,ior,version,pid,server,host,started,stopped,class FROM device WHERE name =?  or alias =?',
                       (dev_name,dev_name))
        result_long = []
        result_str = []
        for row in cursor.fetchall():
            if ((row[4] == None) or (row[5] == None)):
                th_exc(DB_SQLError,
                       "Wrong info in database for device '" + dev_name + "'",
                       "DataBase::GetDeviceInfo()")
            result_str.append(dev_name)
            if raw[1] != None:
                result_str.append(str(raw[1]))
            else:
               result_str.append("")
            result_str.append(str(raw[2]))
            result_str.append(str(raw[4]))
            result_str.append(str(raw[5]))
           
            for i in range(0,2):
                cursor.execute('SELECT DATE_FORMAT(?,\'%D-%M-%Y at %H:%i:%s\')', raw[6 + i])
                tmp_date = cursor.fetchone()
                if tmp_date == None:
                    result_str.append("?")
                else:               
                    result_str.append(str(tmp_date))

            for i in range(0,2):
                if raw[i] != None:
                    result_long.append(raw[i])

        result = (result_long, result_str)
        return result

    @use_cursor
    def get_device_list(self,server_name, class_name ):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT name FROM device WHERE server LIKE ? AND class LIKE ? ORDER BY name',
                       (server_name, class_name))
        return [ row[0] for row in cursor.fetchall() ]
    
    @use_cursor
    def get_device_wide_list(self, wildcard):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT name FROM device WHERE name LIKE ? ORDER BY name',
                       (wildcard,))
        return [ row[0] for row in cursor.fetchall() ]
    
    @use_cursor
    def get_device_member_list(self, wildcard):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT  member FROM device WHERE name LIKE ? ORDER BY member',
                       (wildcard,))
        return [ row[0] for row in cursor.fetchall() ]    

    
    @use_cursor
    def get_device_property(self, dev_name, properties):
        cursor = self.cursor
        stmt = 'SELECT count,value,name FROM property_device WHERE device = ? AND name LIKE ?  ORDER BY count'
        result = []
        result.append(dev_name)
        result.append(str(len(properties)))
        for prop in properties:
            result.append(prop)
            tmp_name = replace_wildcard(prop)
            cursor.execute(stmt, (dev_name, tmp_name))
            rows = cursor.fetchall()
            result.append(attribute)
            result.append(str(len(rows)))
            for row in rows:
                result.append(row[1])
        return result    

    @use_cursor
    def get_device_property_hist(self, device_name, prop_name):
        cursor = self.cursor
        stmt = 'SELECT  DISTINCT id FROM property_device_hist WHERE device=? AND name LIKE ? ORDER by date ASC'
        
        result = []
	
        tmp_name   = replace_wildcard(prop_name);
        
        cursor.execute(stmt, (class_name, device_name, tmp_name))

        stmt = 'SELECT DATE_FORMAT(date,\'%Y-%m-%d %H:%i:%s\'),value,name,count FROM property_device_hist WHERE id =? AND device =? ORDER BY count ASC'

        for row in cursor.fetchall():
            idr = row[0]
            cursor.execute(stmt, (idr, device_name))
            rows = cursor.fetchall()
            result.append(rows[2])
            result.append(rows[0])
            result.append(str(rows[3]))
            for value in rows[1]:
                result.append(value)

        return result

    @use_cursor
    def get_device_server_class_list(self, server_name):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT  class FROM device WHERE server LIKE ? ORDER BY class',
                       (sever_name,))
        return [ row[0] for row in cursor.fetchall() ]
   
    @use_cursor
    def get_exported_device_list_for_class(self, class_name):
        cursor = self.cursor
        cursor.execute('SELECT  DISTINCT name FROM device WHERE class LIKE ? AND exported=1 ORDER BY name',
                       (class_name,))
        return [ row[0] for row in cursor.fetchall() ]   
    
    @use_cursor
    def get_host_list(self, host_name):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT host FROM device WHERE host LIKE ?  ORDER BY host',
                       (host_name,))
        return [ row[0] for row in cursor.fetchall() ]       
    
    @use_cursor
    def get_host_server_list(self, host_name):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT server FROM device WHERE host LIKE ?  ORDER BY server',
                       (host_name,))
        return [ row[0] for row in cursor.fetchall() ]     
     
     
    def get_host_servers_info(self, host_name):
        servers = self.get_host_server_list(host_name)
        result = []
        for server in servers:
            result.append(server)
            info = self.get_server_info(server)
            result.append(info[2]) 
            result.append(info[3])
        return result

     
    def get_instance_name_list(self, server_name):
        server_name = server_name + "\*"
        server_list = self.get_server_list(server_name)
        result = []
        for server in server_list:
            names = server.split("/")
            result.append(names[1])
        return result

    @use_cursor
    def get_object_list(self, name):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT object FROM property WHERE object LIKE ?  ORDER BY object',
                       (name,))
        return [ row[0] for row in cursor.fetchall() ]

    @use_cursor
    def get_property(self, object_name, properties):
        cursor = self.cursor
        result = []
        result.append(object_name)
        result.append(str(len(properties))) 
        stmt = 'SELECT count,value,name FROM property WHERE object LIKE ?  AND name LIKE ? ORDER BY count'
        for prop_name in properties:
            result.append(prop_name)
            prop_name = replace_wildcard(prop_name)
            cursor.execute(stmt, (object_name,prop_name))
            rows = cursor.fetchall()
            n_rows = len(rows)
            result.append(n_rows)
            if n_rows:
                for row in rows:
                    result.append(row[1])
                else:
                    result.append(" ")
        return result


    @use_cursor
    def get_property_hist(self, object_name, prop_name):
        cursor = self.cursor
        result = []
        
        stmt = 'SELECT  DISTINCT id FROM property_hist WHERE object=? AND name LIKE ? ORDER by date'        
        prop_name = replace_wildcard(prop_name)        
        cursor.execute(stmt, (object_name, prop_name))

        stmt = 'SELECT DATE_FORMAT(date,\'%Y-%m-%d %H:%i:%s\'),value,name,count FROM property_hist WHERE id =? AND object =?'
        for row in cursor.fetchall():
            idr = row[0]
            
            cursor.execute(stmt, (idr, object_name))
            rows = cursor.fetchall()
            count = len(rows)
            if rows[3] == 0:
                count = 0
            result.append(rows[2])
            result.append(rows[0])
            result.append(str(count))
            for tmp_row in rows:
                result.append(tmp_row[1])

        return result

    @use_cursor
    def get_property_list(self, object_name, wildcard):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT name FROM property WHERE object LIKE ? AND name LIKE ? ORDER BY name',
                       (object_name,wildcard))
        return [ row[0] for row in cursor.fetchall() ]

    @use_cursor
    def get_server_info(self, server_name):
        cursor = self.cursor
        cursor.execute('SELECT host,mode,level FROM server WHERE name =?',
                       (server_name,))
        result = []
        result.append(server_name)
        row = cursor.fetchone()
        if row is None:
            result.append(" ")
            result.append(" ")
            result.append(" ")
        else:
            result.append(row[0])
            result.append(row[1])
            result.append(row[2])
            
        return result
     
    @use_cursor
    def get_server_list(self, wildcard):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT server FROM device WHERE server LIKE ? ORDER BY server',
                       (wildcard,))
        return [ row[0] for row in cursor.fetchall() ]

    def get_server_list(self, wildcard):
        result = []
        server_list = self.get_server_list(wildcard)
        for server in server_list:
            found = 0
            server_name = server.split("/")[0]
            for res in result:
                if server_name.lower() == res.lower():
                    found = 1
            if not found:
                result.append(server_name)
        return result

    @use_cursor
    def import_device(self, dev_name):
        cursor = self.cursor
        result_long = []
        result_str = []
        # Search first by server name and if nothing found by alias
        # Using OR takes much more time
        cursor.execute('"SELECT exported,ior,version,pid,server,host,class FROM device WHERE name =?',
                       (dev_name,))
        rows = cursor.fetchall()
        if len(rows) == 0:
            cursor.execute('"SELECT exported,ior,version,pid,server,host,class FROM device WHERE alias =?',
                           (dev_name,))
            rows = cursor.fetchall()
            if len(rows) == 0:
                th_exc(DB_DeviceNotDefined,
                       "device " + dev_name + " not defined in the database !",
                       "DataBase::ImportDevice()")
        for row in rows:
            result_str.append(dev_name)
            result_str.append(row[2])
            result_str.append(row[4])
            result_str.append(row[5])
            result_str.append(row[6])
            if row[1] != None:
                result_str.append(row[1])
            else:
                result_str.append("")
            result_long.append(row[0])
            result_long.append(row[3])
        result = (result_long, result_str)
        return result    
     
    @use_cursor
    def import_event(self, event_name):
        cursor = self.cursor
        result_long = []
        result_str = []
        cursor.execute('"SELECT exported,ior,version,pid,host FROM event WHERE name =?',
                       (event_name,))
        rows = cursor.fetchall()
        if len(rows) == 0:
            th_exc(DB_DeviceNotDefined,
                   "event " + event_name + " not defined in the database !",
                   "DataBase::ImportEvent()")
        for row in rows:
            result_str.append(event_name)
            result_str.append(row[1])
            result_str.append(row[2])
            result_str.append(row[4])
            exported = -1
            if row[0] != None:
                exported = row[0]
            result_long.append(exported)
            result_long.append(row[3])
        result = (result_long, result_str)
        return result

     
    @use_cursor
    def info(self):
        cursor = self.cursor
        result = []
         # db name
        info_str = "TANGO Database " + self.db_name
        result.append(info_str)
         # new line
        result.append("")
         # get start time of database
        cursor.execute('SELECT started FROM device WHERE name =?',
                       (self.db_name,))
        row = cursor.fetchone()
        info_str = "Running since ..." + str(row[0])
        result.append(info_str)
        # new line
        result.append("")
        # get number of devices defined
        cursor.execute('SELECT COUNT(*) FROM device')
        row = cursor.fetchone()
        info_str = "Devices defined = " + str(row[0])
        result.append(info_str)
        # get number of devices exported
        cursor.execute('SELECT COUNT(*) FROM device WHERE exported = 1')
        row = cursor.fetchone()
        info_str = "Devices exported = " + str(row[0])
        result.append(info_str)
        # get number of device servers defined
        cursor.execute('SELECT COUNT(*) FROM device WHERE class = \"DServer\" ')
        row = cursor.fetchone()
        info_str = "Device servers defined = " + str(row[0])
        result.append(info_str)
        # get number of device servers exported
        cursor.execute('SELECT COUNT(*) FROM device WHERE class = \"DServer\"  AND exported = 1')
        row = cursor.fetchone()
        info_str = "Device servers exported = " + str(row[0])
        result.append(info_str)
        # new line
        result.append("")
        # get number of device properties
        cursor.execute('SELECT COUNT(*) FROM property_device')
        row = cursor.fetchone()
        info_str = "Device properties defineed = " + str(row[0])
        cursor.execute('SELECT COUNT(*) FROM property_device_hist')
        row = cursor.fetchone()
        info_str = info_str + " [History lgth = " + str(row[0]) + "]"
        result.append(info_str)
        # get number of class properties
        cursor.execute('SELECT COUNT(*) FROM property_class')
        row = cursor.fetchone()
        info_str = "Class properties defined = " + str(row[0])
        cursor.execute('SELECT COUNT(*) FROM property_class_hist')
        row = cursor.fetchone()
        info_str = info_str + " [History lgth = " + str(row[0]) + "]"
        result.append(info_str)
        # get number of device attribute properties
        cursor.execute('SELECT COUNT(*) FROM property_attribute_device')
        row = cursor.fetchone()
        info_str = "Device attribute properties defined = " + str(row[0])
        cursor.execute('SELECT COUNT(*) FROM property_attribute_device_hist')
        row = cursor.fetchone()
        info_str = info_str + " [History lgth = " + str(row[0]) + "]"
        result.append(info_str)
        # get number of class attribute properties
        cursor.execute('SELECT COUNT(*) FROM property_attribute_class')
        row = cursor.fetchone()
        info_str = "Class attribute properties defined = " + str(row[0])
        cursor.execute('SELECT COUNT(*) FROM property_attribute_class_hist')
        row = cursor.fetchone()
        info_str = info_str + " [History lgth = " + str(row[0]) + "]"
        result.append(info_str)
        # get number of object properties
        cursor.execute('SELECT COUNT(*) FROM property')
        row = cursor.fetchone()
        info_str = "Object properties defined = " + str(row[0])
        cursor.execute('SELECT COUNT(*) FROM property_hist')
        row = cursor.fetchone()
        info_str = info_str + " [History lgth = " + str(row[0]) + "]"
        result.append(info_str)
        
        return result
         
    @use_cursor
    def put_attribute_alias(self, attribute_name, attribute_alias):
        cursor = self.cursor
        attribute_name = attribute_name.lower()
        # first check if this alias exists
        cursor.execute('SELECT alias from attribute_alias WHERE alias=? AND name <> ? ',
                       (attribute_alias,attribute_name))
        rows = cursor.fetchall()
        if len(rows) > 0:
            self.warn_stream("DataBase::DbPutAttributeAlias(): this alias exists already ")
            th_exc(DB_SQLError,
                   "alias " + attribute_alias + " already exists !",
                   "DataBase::DbPutAttributeAlias()")
        tmp_names = attribute_name.split("/")
        if len(tmp_names) != 4:
            self.warn_stream("DataBase::DbPutAttributeAlias(): attribute name has bad syntax, must have 3 / in it")
            th_exc(DB_SQLError,
                   "attribute name " + attribute_name + " has bad syntax, must have 3 / in it",
                   "DataBase::DbPutAttributeAlias()")
         # first delete the current entry (if any)
        cursor.execute('DELETE FROM attribute_alias WHERE name=?',
                       (attribute_name,))
         # update the new value for this tuple
        tmp_device = tmp_names[0] + "/" + tmp_names[1] + "/" + tmp_names[2]
        tmp_attribute = tmp_names[3]
        cursor.execute('INSERT attribute_alias SET alias=? ,name=?, device=?,updated=NOW()',
                       (attribute_alias, tmp_device, tmp_attribute)) 

         
    @use_cursor
    def put_class_attribute_property(self, class_name, nb_attributes, attr_prop_list):
        cursor = self.cursor
        k = 0
        for i in range(0,nb_attributes):
            tmp_attribute = attr_prop_list[k]
            nb_properties = int(attr_prop_list[k+1])
            for j in range(k+2,k+nb_properties*2+2,2):
                tmp_name = attr_prop_list[j]
                tmp_value = attr_prop_list[j+1]
                 # first delete the tuple (device,name,count) from the property table
                cursor.execute('DELETE FROM property_attribute_class WHERE class LIKE ? AND attribute LIKE ? AND name LIKE ?', (class_name, tmp_attribute, tmp_name))
                # then insert the new value for this tuple
                cursor.execute('INSERT INTO property_attribute_class SET class=? ,attribute=?,name=?,count=\'1\',value=?,updated=NULL,accessed=NULL', (class_name, tmp_attribute, tmp_name, tmp_value))
                # then insert the new value into the history table
                hist_id = self.get_id("class_attribute", cursor=cursor)
                cursor.execute('INSERT INTO property_attribute_class_hist SET class=?,attribute=?,name=?,id=?,count=\'1\',value=?', (class_name, tmp_attribute, tmp_name,hist_id,tmp_value))

                self.purge_att_property("property_attribute_class_hist", "class",
                                        class_name, tmp_attribute, tmp_name, cursor=cursor)
            k = k + nb_properties*2+2

    @use_cursor
    def put_class_attribute_property2(self, class_name, nb_attributes, attr_prop_list):
        cursor = self.cursor
        k = 0
        for i in range(0,nb_attributes):
            tmp_attribute = attr_prop_list[k]
            nb_properties = int(attr_prop_list[k+1])
            for jj in range(0,nb_properties,1):
                j = k + 2
                tmp_name = attr_prop_list[j]
                # first delete the tuple (device,name,count) from the property table
                cursor.execute('DELETE FROM property_attribute_class WHERE class LIKE ? AND attribute LIKE ? AND name LIKE ?', (class_name, tmp_attribute, tmp_name))
                n_rows = attr_prop_list[j+1]
                tmp_count = 0
                for l in range(j+1,j+n_rows+1,1):
                    tmp_value = attr_prop_list[l+1]
                    tmp_count = tmp_count + 1
                    # then insert the new value for this tuple
                    cursor.execute('INSERT INTO property_attribute_class SET class=? ,attribute=?,name=?,count=?,value=?,updated=NULL,accessed=NULL', (class_name, tmp_attribute, tmp_name, str(tmp_count), tmp_value))
                    # then insert the new value into the history table
                    hist_id = self.get_id("class_attribute", cursor=cursor)
                    cursor.execute('INSERT INTO property_attribute_class_hist SET class=?,attribute=?,name=?,id=?,count=?,value=?', (class_name, tmp_attribute, tmp_name,hist_id, str(tmp_count),tmp_value))

                    self.purge_att_property("property_attribute_class_hist", "class",
                                            class_name, tmp_attribute, tmp_name, cursor=cursor)
                k = k + n_rows + 2
            k = k + 2    

    @use_cursor
    def put_class_property(self, class_name, nb_properties, attr_prop_list):
        cursor = self.cursor
        k = 0
        for i in range(0,nb_properties):
            tmp_count = 0
            tmp_name = attr_prop_list[k]
            n_rows = attr_prop_list[k+1]
             # first delete all tuples (device,name) from the property table
            cursor.execute('DELETE FROM property_class WHERE class LIKE ? AND name LIKE ?', (class_name, tmp_name))

            for j in range(k+2,k+n_rows+2,1):
                tmp_value = attr_prop_list[j]
                tmp_count = tmp_count+1
                # then insert the new value for this tuple
                cursor.execute('INSERT INTO property_class SET class=? ,name=?,count=?,value=?,updated=NULL,accessed=NULL', (class_name, tmp_name, str(tmp_count), tmp_value))
                # then insert the new value into the history table
                hist_id = self.get_id("class", cursor=cursor)
                cursor.execute('INSERT INTO property_class_hist SET class=?,name=?,id=?,count=?,value=?', (class_name, tmp_name,hist_id, str(tmp_count),tmp_value))
                self.purge_att_property("property_class_hist", "class",
                                        class_name, tmp_name, cursor=cursor)
            k = k + n_rows + 2

    @use_cursor
    def put_device_alias(self, device_name, device_alias):
        cursor = self.cursor
        device_name = device_name.lower()
        # first check if this alias exists
        cursor.execute('SELECT alias from device WHERE alias=? AND name <>?',
                       (device_alias, device_name))
        rows = cursor.fetchall()
        if len(rows) > 0:
            self.warn_stream("DataBase::DbPutDeviceAlias(): this alias exists already ")
            th_exc(DB_SQLError,
                   "alias " + device_alias + " already exists !",
                   "DataBase::DbPutDeviceAlias()")
        # update the new value for this tuple
        cursor.execute('UPDATE device SET alias=? ,started=NOW() where name LIKE ?',
                       (device_alias, device_name)) 

    @use_cursor
    def put_device_attribute_property(self, device_name, nb_attributes, attr_prop_list):
        cursor = self.cursor
        k = 0
        for i in range(0,nb_attributes):
            tmp_attribute = attr_prop_list[k]
            nb_properties = int(attr_prop_list[k+1])
            for j in range(k+2,k+nb_properties*2+2,2):
                tmp_name = attr_prop_list[j]
                tmp_value = attr_prop_list[j+1]
                # first delete the tuple (device,name,count) from the property table
                cursor.execute('DELETE FROM property_attribute_device WHERE device LIKE ? AND attribute LIKE ? AND name LIKE ?', (device_name, tmp_attribute, tmp_name))
                # then insert the new value for this tuple
                cursor.execute('INSERT INTO property_attribute_device SET device=? ,attribute=?,name=?,count=\'1\',value=?,updated=NULL,accessed=NULL', (device_name, tmp_attribute, tmp_name, tmp_value))
                # then insert the new value into the history table
                hist_id = self.get_id("device_attribute", cursor=cursor)
                cursor.execute('INSERT INTO property_attribute_device_hist SET device=?,attribute=?,name=?,id=?,count=\'1\',value=?', (device_name, tmp_attribute, tmp_name,hist_id,tmp_value))

                self.purge_att_property("property_attribute_device_hist", "device",
                                         device_name, tmp_attribute, tmp_name, cursor=cursor)
            k = k + nb_properties*2+2         


    @use_cursor
    def put_device_attribute_property2(self, device_name, nb_attributes, attr_prop_list):
        cursor = self.cursor
        k = 0
        for i in range(0,nb_attributes):
            tmp_attribute = attr_prop_list[k]
            nb_properties = int(attr_prop_list[k+1])
            for jj in range(0,nb_properties,1):
                j = k + 2
                tmp_name = attr_prop_list[j]
                # first delete the tuple (device,name,count) from the property table
                cursor.execute('DELETE FROM property_attribute_device WHERE device LIKE ? AND attribute LIKE ? AND name LIKE ?', (device_name, tmp_attribute, tmp_name))
                n_rows = attr_prop_list[j+1]
                tmp_count = 0
                for l in range(j+1,j+n_rows+1,1):
                    tmp_value = attr_prop_list[l+1]
                    tmp_count = tmp_count + 1
                    # then insert the new value for this tuple
                    cursor.execute('INSERT INTO property_attribute_device SET device=? ,attribute=?,name=?,count=?,value=?,updated=NULL,accessed=NULL', (device_name, tmp_attribute, tmp_name, str(tmp_count), tmp_value))
                    # then insert the new value into the history table
                    hist_id = self.get_id("device_attribute", cursor=cursor)
                    cursor.execute('INSERT INTO property_attribute_device_hist SET device=?,attribute=?,name=?,id=?,count=?,value=?', (device_name, tmp_attribute, tmp_name,hist_id, str(tmp_count),tmp_value))

                    self.purge_att_property("property_attribute_device_hist", "device",
                                            device_name, tmp_attribute, tmp_name, cursor=cursor)
                k = k + n_rows + 2
            k = k + 2    

    @use_cursor
    def put_device_property(self, device_name, nb_properties, attr_prop_list):
        cursor = self.cursor
        k = 0
        hist_id = self.get_id("device", cursor=cursor)
        for i in range(0,nb_properties):
            tmp_count = 0
            tmp_name = attr_prop_list[k]
            n_rows = attr_prop_list[k+1]
            # first delete all tuples (device,name) from the property table
            cursor.execute('DELETE FROM property_device WHERE device LIKE ? AND name LIKE ?', (device_name, tmp_name))

            for j in range(k+2,k+n_rows+2,1):
                tmp_value = attr_prop_list[j]
                tmp_count = tmp_count+1
                # then insert the new value for this tuple
                cursor.execute('INSERT INTO property_device SET device=? ,name=?,count=?,value=?,updated=NULL,accessed=NULL', (device_name, tmp_name, str(tmp_count), tmp_value))
                # then insert the new value into the history table
                cursor.execute('INSERT INTO property_device_hist SET device=?,name=?,id=?,count=?,value=?', (device_name, tmp_name,hist_id, str(tmp_count),tmp_value))
            self.purge_att_property("property_device_hist", "device",
                                    device_name, tmp_name, cursor=cursor)
            k = k + n_rows + 2

    @use_cursor
    def put_property(self, object_name, nb_properties, attr_prop_list):
        cursor = self.cursor
        k = 0
        hist_id = self.get_id("object", cursor=cursor)
        for i in range(0,nb_properties):
            tmp_count = 0
            tmp_name = attr_prop_list[k]
            n_rows = attr_prop_list[k+1]
            # first delete the property from the property table
            cursor.execute('DELETE FROM property WHERE object =? AND name =?', (object_name, tmp_name))

            for j in range(k+2,k+n_rows+2,1):
                tmp_value = attr_prop_list[j]
                tmp_count = tmp_count+1
                # then insert the new value for this tuple
                cursor.execute('INSERT INTO property SET object=? ,name=?,count=?,value=?,updated=NULL,accessed=NULL', (object_name, tmp_name, str(tmp_count), tmp_value))
                # then insert the new value into the history table
                cursor.execute('INSERT INTO property_hist SET object=?,name=?,id=?,count=?,value=?', (object_name, tmp_name,hist_id, str(tmp_count),tmp_value))
            self.purge_att_property("property_hist", "object",
                                    object_name, tmp_name, cursor=cursor)
            k = k + n_rows + 2

    @use_cursor
    def put_server_info(self, tmp_server, tmp_host, tmp_mode, tmp_level, tmp_extra):
        cursor = self.cursor         
         # If it is an empty host name -> get previous host where running
        previous_host = ""
        if self.fire_to_starter:
            if tmp_host == "":
                adm_dev_name = "dserver/" + tmp_server
                previous_host = self.get_device_host(adm_dev_name)
        # first delete the server from the server table         
        cursor.execute('DELETE FROM server WHERE name=?', (tmp_server,))
        # insert the new info for this server
        cursor.execute('INSERT INTO server SET name=? ,host=? ,mode=? ,level=?', ( tmp_server, tmp_host, tmp_mode, tmp_level))
        #  Update host's starter to update controlled servers list
        if self.fire_to_starter:
            hosts = []
            if previous_host == "":
                hosts.append(tmp_host)
            else:
                hosts.append(previous_host)
            self.send_starter_cmd(hosts)
                 
    @use_cursor
    def uexport_device(self, dev_name):
        cursor = self.cursor         
        self._info("un-export device(dev_name=%s)", dev_name)
        cursor.execute('UPDATE device SET exported=0,stopped=NOW() WHERE name LIKE ?', (dev_name,))
        
    @use_cursor
    def uexport_event(self, event_name):
        cursor = self.cursor         
        self._info("un-export event (event_name=%s)", event_name)
        cursor.execute('UPDATE event SET exported=0,stopped=NOW() WHERE name LIKE ?', (event_name,))
                               
    @use_cursor
    def uexport_server(self, server_name):
        cursor = self.cursor         
        self._info("un-export all devices from server ", server_name)
        cursor.execute('UPDATE device SET exported=0,stopped=NOW() WHERE server LIKE ?', (server_name,))
        
    @use_cursor
    def delete_all_device_attribute_property(self, dev_name, attr_list):
        cursor = self.cursor  
        for attr_name in attr_list:
            self._info("_delete_all_device_attribute_property(): delete device %s attribute %s property(ies) from database", dev_name, attr_name)
             #Is there something to delete ?   
            cursor.execute('SELECT DISTINCT name FROM property_attribute_device WHERE device =? AND attribute = ?', (dev_name,attr_name))
            rows = cursor.fetchall()
            if len(rows) != 0:
                cursor.execute('DELETE FROM property_attribute_device WHERE device = ? AND attribute = ?', (dev_name,attr_name))
            # Mark this property as deleted
            for row in rows:
                hist_id = self.get_id('device_attribute', cursor=cursor)
                cursor.execute('INSERT INTO property_attribute_device_hist SET device=?,attribute=?,name=?,id=?,count=\'0\',value=\'DELETED\'', (dev_name,attr_name,row[0], hist_id))
                self.purge_att_property("property_attribute_device_hist", "device",
                                         dev_name, attr_name, row[0], cursor=cursor)
                
    @use_cursor
    def my_sql_select(self, cmd):
        cursor = self.cursor
        cursor.execute(cmd)
        result_long = []
        result_str = []
        rows = cursor.fetchall()
        nb_fields = 0
        for row in rows:
            if row == None:
                result_str.append("")
                result_long.append(0)
            else:
                for field in row:
                    nb_fields = nb_fields + 1
                    if field != None:
                        result_str.append(str(field))
                        result_long.append(1)
                    else:                        
                        result_str.append("")
                        result_long.append(0)
        result_long.append(len(rows))
        result_long.append(nb_fields)
                    
        result = (result_long, result_str)
        return result



    @use_cursor
    def get_csdb_server_list(self):
        cursor = self.cursor
        
        cursor.execute('SELECT DISTINCT ior FROM device WHERE exported=1 AND domain=\'sys\' AND family=\'database\'')
        return [ row[0] for row in cursor.fetchall() ]
     
    @use_cursor
    def get_attribute_alias2(self, attr_name):
        cursor = self.cursor
        cursor.execute('SELECT alias from attribute_alias WHERE name LIKE ? ',(attr_name,))        
        return [ row[0] for row in cursor.fetchall() ]
    
    @use_cursor
    def get_alias_attribute(self, alias_name):
        cursor = self.cursor
        cursor.execute('SELECT name from attribute_alias WHERE alias LIKE ? ',(alias_name,))        
        return [ row[0] for row in cursor.fetchall() ]     
    
    @use_cursor
    def rename_server(self, old_name, new_name):
        cursor = self.cursor
        # Check that the new name is not already used
        new_adm_name = "dserver/" + new_name
        cursor.execute('SELECT name from device WHERE name = ? ',(new_adm_name,))
        rows = cursor.fetchall()
        if len(rows) != 0:
            th_exc(DB_SQLError,
                   "Device server process name " + attribute_alias + "is already used !",
                   "DataBase::DbRenameServer()")
            
        # get host where running
        previous_host = ""
        if self.fire_to_starter:
            try:
                adm_dev = "dserver/" + old_name
                previous_host = self.get_device_host(adm_dev)
            except:
                th_exc(DB_IncorrectServerName,
                       "Server " + old_name + "not defined in database!",
                       "DataBase::DbRenameServer()")
        # Change ds exec name. This means
        #  1 - Update the device's server column
        #  2 - Change the ds admin device name
        #  3 - Change admin device property (if any)
        #  4 - Change admin device attribute property (if any)
     
        old_adm_name = "dserver/" + old_name
        tmp_new = new_name.split('/')
        new_exec = tmp_new[0]
        new_inst = tmp_new[1]    
        new_adm_name = "dserver/" + new_name
        
        cursor.execute('UPDATE device SET name =?, family =?, mamber =? WHERE name =?', (new_adm_name, new_exec, new_inst, old_adm_name))
        
        cursor.execute('UPDATE property_device set device=? WHERE device=?', (new_adm_name, old_adm_name))
        
        cursor.execute('UPDATE property_attribute_device set device=? WHERE device=?', (new_adm_name, old_adm_name))
              
        #  Update host's starter to update controlled servers list
        if self.fire_to_starter:
            hosts = []
            if previous_host == "":
                hosts.append(tmp_host)
            else:
                hosts.append(previous_host)
            self.send_starter_cmd(hosts)

   
class sqlite3(Tango_dbapi2):

    DB_API_NAME = 'sqlite3'

def main():
    db = Tango_sqlite3()
    db.add_device("MyServer/my1", ("a/b/c", ("a", "b", "c")), "MyClass")
    db.close_db()

def get_db(**keys):
    return Executor.submit(sqlite3).result()

if __name__ == "__main__":
    main()
