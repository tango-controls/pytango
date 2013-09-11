from __future__ import print_function

import os
import functools

import PyTango

th_exc = PyTango.Except.throw_exception

from db_errors import *

def get_create_db_statements():
    statements = []
    with open("create_db_tables.sql") as f:
        lines = f.readlines()
    # strip comments
    lines = ( line for line in lines if not line.startswith('#') )
    lines = ( line for line in lines if not line.lower().strip().startswith('key') )
    lines = ( line for line in lines if not line.lower().strip().startswith('key') )
    lines = "".join(lines)
    lines = lines.replace("ENGINE=MyISAM","")
    statements += lines.split(";")
        
    with open("create_db.sql") as f:
        lines = f.readlines()
    # strip comments
    lines = ( line for line in lines if not line.lower().startswith('#') )
    lines = ( line for line in lines if not line.lower().startswith('create database') )
    lines = ( line for line in lines if not line.lower().startswith('use') )
    lines = ( line for line in lines if not line.lower().startswith('source') )
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
            cursor = self.get_cursor()
        self.cursor = cursor
        try:
            ret = f(*args, **kwargs)
            if not has_cursor:
                cursor.connection.commit()
            return ret
        finally:
            if not has_cursor:
                cursor.close()
                del self.cursor
    return wrap
    
class Tango_dbapi2(object):
    
    DB_API_NAME = 'sqlite3'
    
    def __init__(self, db_name="tango_database.db", history_depth=10, fire_to_starter=True):
        self._db_api = None
        self._db_conn = None
        self.db_name = db_name
        self.history_depth = history_depth;
        self.fire_to_starter = fire_to_starter
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
        if not os.path.isfile(self.db_name):
            self.create_db()
    
    @use_cursor
    def create_db(self):
        print("Creating database...")
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

    # TANGO API
    
    def get_stored_procedure_release(self):
        return 'release 1.8'

    @use_cursor
    def add_device(self, server_name, dev_info, klass_name, alias=None):
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
            hist_id = self.get_id('class_attibute', cursor=cursor)
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
            # TODO send to starter
            pass

    @use_cursor
    def delete_server_info(self, server_instance):
        self.cursor.execute('DELETE FROM server WHERE name=?', (server_instance,))

    @use_cursor
    def export_device(self, dev_name, IOR, host, pid, version):
        cursor = self.cursor
        do_fire = False
        previous_host = None
        if self.fire_to_starter:
            # TODO send to starter
            pass
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
        cursor.execute('UPDATE server SET host=?, WHERE name LIKE ?', (host, server))
        
        if do_fire:
            # TODO send to starter
            pass
        
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
        cursor.execute('SELECT name FROM device WHERE alias LIKE ?', (dev_alias,))
        row = cursor.fetchone()
        if row is None:
            th_exc(DB_DeviceNotDefined,
                   "No device found for alias '" + dev_alias + "'",
                   "DataBase::GetAliasDevice()")
        return row[0]   
    
    @use_cursor
    def get_attribute_alias(self, attr_alias):
        cursor = self.cursor
        cursor.execute('SELECT name from attribute_alias WHERE alias LIKE ?', (attr_alias,))
        row = cursor.fetchone()
        if row is None:
            th_exc(DB_SQLError,
                   "No attribute found for alias '" + attr_alias + "'",
                   "DataBase::GetAttributeAlias()")        
        return row[0]
    
    @use_cursor
    def get_attribute_alias_list(self, attr_alias):
        cursor = self.cursor
        cursor.execute('SELECT DISTINCT alias FROM attribute_alias WHERE alias LIKE ? ORDER BY attribute', (attr_alias,))
        return [ row[0] for row in cursor.fetchall() ]
    
        
class Tango_sqlite3(Tango_dbapi2):
    
    DB_API_NAME = 'sqlite3'
    

def main():
    db = Tango_sqlite3()
    db.add_device("MyServer/my1", ("a/b/c", ("a","b","c")), "MyClass")
    db.close_db()
    
if __name__ == "__main__":
    main()
