from __future__ import print_function
from __future__ import absolute_import

import os
import logging
import functools
import threading
import Queue
import re
import weakref
import datetime
import tango

th_exc = tango.Except.throw_exception

from ..db_errors import *

from bliss.config import static,settings
import itertools

if logging.getLogger().isEnabledFor(logging.INFO):
    def _info(funct) :
        def f(self,*args,**kwargs) :
            self._info("%s: %s %s", funct.__name__,args,kwargs)
            returnVal = funct(self,*args,**kwargs)
            if returnVal is not None:
                self._info("return %s : %s",funct.__name__,returnVal)
            else:
                self._info("return %s",funct.__name__)
            return returnVal
        return f
else:
    def _info(funct):
        return funct

def _filter(wildcard,l) :
    wildcard = wildcard.replace('*','.*')
    m = re.compile(wildcard,re.IGNORECASE)
    return [x for x in l if x is not None and m.match(x)]

class beacon(object):

    DB_API_NAME = 'beacon'
    TANGO_ATTR_ALIAS = 'tango.attr.alias'

    def __init__(self, personal_name = "",**keys):
        self._config = static.Config('',3.)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._debug = self._logger.debug
        self._info = self._logger.info
        self._warn = self._logger.warn
        self._error = self._logger.error
        self._critical = self._logger.critical
        
        self._index()
        
        #Trick to start
        self._beacon_dserver_node = static.Node(self._config)
        self._beacon_dserver_node['server'] = 'DataBaseds'
        self._beacon_dserver_node['personal_name'] = personal_name
        tango_name =  'sys/database/%s' % personal_name
        databse_device_node = static.Node(self._config,self._beacon_dserver_node)
        databse_device_node['class'] = 'DataBase'
        databse_device_node['tango_name'] = tango_name
        self._beacon_dserver_node['device'] = [databse_device_node]
        self._beacon_dserver_node['tango_name'] = tango_name
        self._tango_name_2_node[tango_name] = databse_device_node
        tango_name =  'dserver/databaseds/%s' % personal_name
        self._tango_name_2_node[tango_name] = self._beacon_dserver_node
        server_name = 'DataBaseds/%s' % personal_name
        self._personal_2_node[server_name] = self._beacon_dserver_node

    def _index(self):
        #Tango indexing
        self._strong_node_ref = set()
        self._personal_2_node = weakref.WeakValueDictionary()
        self._tango_name_2_node = weakref.WeakValueDictionary()
        self._class_name_2_node = weakref.WeakValueDictionary()

        for key,values in self._config.root.iteritems():
            indexing_flag = key == 'tango'
            if isinstance(values,list) :
                self._parse_list(values,indexing_flag)
            elif isinstance(values,dict):
                self._parse_dict(values,indexing_flag)
                self._index_tango(values)

    def _parse_list(self,l,indexing_flag) :
        for v in l:
            if isinstance(v,list):
                self._parse_list(v,indexing_flag)
            elif isinstance(v,dict):
                if indexing_flag:
                    self._index_tango(v)
                self._parse_dict(v,indexing_flag)

    def _parse_dict(self,d,indexing_flag):
        for k,v in d.iteritems():
            if isinstance(v,dict):
                if not indexing_flag: 
                    indexing_flag = k == 'tango'
                if indexing_flag:
                    self._index_tango(v)
                self._parse_dict(v,indexing_flag)
            elif isinstance(v,list) :
                self._parse_list(v,indexing_flag)

    def _index_tango(self,v) :
        klass = v.get('class')
        if klass is not None and v.parent.get('device') is None:
            self._class_name_2_node[klass] = v

        personal_name = v.get('personal_name')
        if personal_name is not None :
            server = v.get('server')
            if server is None:
                self._error("_index_tango(personal_name=%) didn't specify server key (executable name)",
                            personal_name)
                return

            personal_name = personal_name.lower()
            dserver_name = '%s/%s' % (server,personal_name)
            self._personal_2_node[dserver_name] = v
            self._tango_name_2_node['dserver/%s' % dserver_name.lower()] = v

        tango_name = v.get('tango_name')
        if tango_name is not None:
            tango_name = tango_name.lower()
            self._tango_name_2_node[tango_name.lower()] = v

        alias = v.get('alias')
        if alias is not None:
            self._tango_name_2_node[alias] = v

    # TANGO API
    @_info
    def get_stored_procedure_release(self):
        return 'release 0.0'

    @_info
    def add_device(self, server_name, dev_info, klass_name, alias=None):
        self._info("add_device(server_name=%s, dev_info=%s, klass_name=%s, alias=%s)",
                   server_name, dev_info, klass_name, alias)
        tango_name, _ = dev_info
        tango_name = tango_name.lower()
        device_node = self._tango_name_2_node.get(tango_name)
        if device_node is not None:    # There is a problem?
            return
        server_exe_name,personal_name = server_name.split('/')
        personal_name = personal_name.lower()
        server_name = '%s/%s' % (server_exe_name,personal_name)
        server_node = self._personal_2_node.get(server_name)
        if server_node is None:
            server_node = static.Node(self._config,filename = 'tango/%s.yml' % server_name.replace('/','_'))
            server_node['server'] = server_exe_name
            server_node['personal_name'] = personal_name
            self._personal_2_node[server_name] = server_node
            self._tango_name_2_node['dserver/%s' % server_name.lower()] = server_node
            self._strong_node_ref.add(server_node)
            
        device_node = static.Node(self._config,parent=server_node)
        self._strong_node_ref.add(device_node)
        device_node['tango_name'] = tango_name
        device_node['class'] = klass_name
        if alias is not None:
            device_node['alias'] = alias
        device_node_list = server_node.get('device',[])
        device_node_list.append(device_node)
        server_node['device'] = device_node_list
        self._tango_name_2_node[tango_name] = device_node


        server_node.save()

    @_info
    def delete_attribute_alias(self, alias):
        self._info("delete_attribute_alias(alias=%s)",alias)
        attr_alias = settings.HashObjSetting(self.TANGO_ATTR_ALIAS)
        del attr_alias[alias]
    
    def _get_class_attribute(self,klass_name, attr_name) :
        self._info("_get_class_attribute(klass_name=%s,attr_name=%s)",
                   klass_name, attr_name)
        key_name = 'tango.class.attribute.%s.%s' % (klass_name,attr_name)
        return settings.HashObjSetting(key_name)

    @_info
    def delete_class_attribute(self, klass_name, attr_name):
        class_attribute = self._get_class_attribute(klass_name,attr_name)
        class_attribute.clear()

    @_info
    def delete_class_attribute_property(self, klass_name, attr_name, prop_name):
        class_attribute = self._get_class_attribute(klass_name,attr_name)
        del class_attribute[prop_name]

    def _get_class_properties(self,klass_name,prop_name):
        #key_name = 'tango.class.properties.%s.%s' % (klass_name,prop_name)
        #return settings.QueueSetting(key_name)
        return self._class_name_2_node.get(klass_name,dict()).get('properties',dict()).get(prop_name,'')

    @_info
    def delete_class_property(self, klass_name, prop_name):
        #class_property = self._get_class_properties(klass_name,prop_name)
        #class_property.clear()
        pass

    def _get_property_attr_device(self,dev_name) :
        key_name = 'tango.%s' % dev_name.lower().replace('/','.')
        return settings.HashObjSetting(key_name)

    @_info
    def delete_device(self, dev_name):
        dev_name = dev_name.lower()
        self._info("delete_device(dev_name=%s)", dev_name)

        device_node = self._tango_name_2_node.pop(dev_name,None)
        if device_node is None:
            return
        
        server_node = device_node.parent
        if server_node is None: # weird
            return
        device_list = server_node.get('device',[])
        device_list.remove(device_node)
        server_node.save()
        
        prop_attr_device = self._get_property_attr_device(dev_name)
        prop_attr_device.clear()

    @_info
    def delete_device_alias(self, dev_alias):
        self._info("delete_device_alias(dev_alias=%s)", dev_alias)
        device_node = self._tango_name_2_node.pop(dev_alias)
        if device_node is None:
            return

        server_node = device_node.parent
        if server_node is None: # weird
            return
        del device_node['alias']
        server_node.save()

    @_info
    def delete_device_attribute(self, dev_name, attr_name):
        prop_attr_device = self._get_property_attr_device(dev_name)
        del prop_attr_device[attr_name]

    @_info
    def delete_device_attribute_property(self, dev_name, attr_name, prop_name):
        prop_attr_device = self._get_property_attr_device(dev_name)
        d = prop_attr_device.get(attr_name)
        if d is not None:
            del d[prop_name]
            prop_attr_device[attr_name] = d

    @_info
    def delete_device_property(self, dev_name, prop_name):
        properties = self._get_property_node(dev_name)
        if properties is None: return
        try:
            del properties[prop_name]
            properties.save()
        except KeyError:
            pass

    @_info
    def delete_property(self, obj_name, prop_name):
        self._warn("Not implemented delete_property(obj_name=%s, prop_name=%s)", (obj_name,prop_name))

    @_info
    def delete_server(self, server_name):
        server_node = self._personal_2_node.get(server_name)
        if server_node is None:
            return

        server_node.clear()
        server_node.save()

    @_info
    def delete_server_info(self, server_instance):
        self._warn("Not implemented delete_server_info(server_instance=%s)", (server_instance))

    def _get_export_device_info(self,dev_name):
        key_name = 'tango.info.%s' % dev_name
        return settings.HashSetting(key_name)

    @_info
    def export_device(self, dev_name, IOR, host, pid, version):
        self._info("export_device(dev_name=%s, host=%s, pid=%s, version=%s)",
                   dev_name, host, pid, version)
        dev_name = dev_name.lower()
        device_node = self._tango_name_2_node.get(dev_name)
        if device_node is None:
            th_exc(DB_DeviceNotDefined,
                   "device " + dev_name + " not defined in the database !",
                   "DataBase::ExportDevice()")

        self._info("export_device(IOR=%s)", IOR)
        export_device_info = self._get_export_device_info(dev_name)
        start_time = datetime.datetime.now()
        export_device_info.set({'IOR':IOR,'host':host,'pid':pid,'version':version,
                                'start-time': '%s' % start_time})

    @_info
    def export_event(self, event, IOR, host, pid, version):
        self._warn("Not implemented export_event(event=%s, IOR=%s, host=%s, pid=%s, version=%s)",
                   (event, IOR, host, pid, version))
                

    @_info
    def get_alias_device(self, dev_alias):
        device_node = self._tango_name_2_node.get(dev_alias)
        if device_node is None:
            th_exc(DB_DeviceNotDefined,
                   "No device found for alias '" + dev_alias + "'",
                   "DataBase::GetAliasDevice()")
        
        return device_node.get('tango_name')

    @_info
    def get_attribute_alias(self, attr_alias_name):
        attr_alias = settings.HashObjSetting(self.TANGO_ATTR_ALIAS)
        attr_alias_info = attr_alias.get(attr_alias_name)
        if attr_alias_info is None:
            th_exc(DB_SQLError,
                   "No attribute found for alias '" + attr_alias + "'",
                   "DataBase::GetAttributeAlias()")
        return attr_alias_info.get('name')

    @_info
    def get_attribute_alias_list(self, attr_alias_name):
        attr_alias = settings.HashObjSetting(self.TANGO_ATTR_ALIAS)
        attr_alias_info = attr_alias.get(attr_alias_name)
        if attr_alias_info is None:
            return []
        else:
            return [attr_alias_info.get('name')]

    @_info
    def get_class_attribute_list(self, class_name, wildcard):
        redis = settings.get_cache()
        attributes = [x for x in redis.scan_iter(match='tango.class.attribute.%s' % class_name)]
        return _filter(wildcard,attributes)

    @_info
    def get_class_attribute_property(self, class_name, attributes):
        result = [class_name, str(len(attributes))]
        for att_name in attributes:
            class_attribute_properties = self._get_class_attribute(class_name,att_name)
            attr_property = [x for p in class_attribute_properties.iteritems() for x in p]
            result.extend([att_name,str(len(attr_property)/2)] + attr_property)
        return result

    @_info
    def get_class_attribute_property2(self, class_name, attributes):
        result = [class_name, str(len(attributes))]
        for attr_name in attributes:
            class_properties = self._get_class_attribute(class_name,attr_name)
            attr_property = []
            for nb,(name,values) in enumerate(class_properties.iteritems()):
                if isinstance(values,list) :
                    attr_property.extend([name,str(len(values))] + [str(x) for x in values])
                else:
                    attr_property.extend([name,'1',str(values)])
            
            if(attr_property):
                result.extend([attr_name,str(nb)] + attr_property)
            else:
                result.extend([attr_name,'0'])
        return result

    @_info
    def get_class_attribute_property_hist(self, class_name, attribute, prop_name):
        return []

    @_info
    def get_class_for_device(self, dev_name):
        dev_name = dev_name.lower()
        device_node = self._tango_name_2_node.get(dev_name)
        if device_node is None:
            th_exc(DB_IncorrectArguments, "Device not found for " + dev_name,
                   "Database.GetClassForDevice")
        class_name = device_node.get('class')
        if class_name is None:
            th_exc(DB_IncorrectArguments, "Class not found for " + dev_name,
                   "Database.GetClassForDevice")
        return class_name

    @_info
    def get_class_inheritance_for_device(self, dev_name):
        class_name = self.get_class_for_device(dev_name)
        props = self.get_class_property(class_name, "InheritedFrom")
        return [class_name] + props[4:]

    @_info
    def get_class_list(self, server):
        server_name_list = [x.get('server') for x in self._personal_2_node.values() if x.has_key('server')]
        res =  _filter(server,set(server_name_list))
        res.sort()
        return res

    @_info
    def get_class_property(self, class_name, properties):
        result = [class_name,str(len(properties))]
        for prop_name in properties:
            properties_array = []
            values = self._get_class_properties(class_name,prop_name)
            if isinstance(values,list):
                values = [str(x) for x in values]
                properties_array.extend([prop_name,str(len(values))] + values)
            elif values:
                properties_array.extend([prop_name,'1',str(values)])
            else:
                properties_array.extend([prop_name,'0'])
            result.extend(properties_array)
        return result
        
    @_info
    def get_class_property_hist(self, class_name, prop_name):
        return []
        
    @_info
    def get_class_property_list(self, class_name):
        properties = self._class_name_2_node.get(class_name,dict()).get("properties", dict())
        return [k for k,v in properties.iteritems() if not isinstance(v,dict)]
        #cache = settings.get_cache()
        #return cache.keys('tango.class.properties.%s*' % class_name)

    @_info
    def get_device_alias(self, dev_name):
        dev_name = dev_name.lower()
        device_node = self._tango_name_2_node.get(dev_name)
        if device_node is None:
            th_exc(DB_DeviceNotDefined,
                   "No alias found for device '" + dev_name + "'",
                   "DataBase::GetDeviceAlias()")
        alias = device_node.get('alias')
        if alias is None:
            th_exc(DB_DeviceNotDefined,
                   "No alias found for device '" + dev_name + "'",
                   "DataBase::GetDeviceAlias()")
        return alias

    @_info
    def get_device_alias_list(self, alias):
        alias_list = [node.get('alias') for node in self._tango_name_2_node.values()]
        return _filter(alias,alias_list)

    @_info
    def get_device_attribute_list(self, dev_name, attribute):
        prop_attr_device = self._get_property_attr_device(dev_name)
        return _filter(attribute,prop_attr_device.keys())

    @_info
    def get_device_attribute_property(self, dev_name, attributes):
        prop_attr_device = self._get_property_attr_device(dev_name)
        result = [dev_name,str(len(attributes))]
        for attr_name in attributes:
            prop_attr = prop_attr_device.get(attr_name)
            if prop_attr is None:
                result.extend([attr_name,'0'])
            else:
                result.extend([attr_name,str(len(prop_attr))] +
                              [str(x) for p in prop_attr.iteritems() for x in p])
        return result  

    @_info
    def get_device_attribute_property2(self, dev_name, attributes):
        prop_attr_device_handler = self._get_property_attr_device(dev_name)
        result = [dev_name, str(len(attributes))]
        prop_attr_device = prop_attr_device_handler.get_all()
        for attr_name in attributes:
            prop_attr = prop_attr_device.get(attr_name)
            if prop_attr is None:
                result.extend((attr_name,'0'))
            else:
                result.extend((attr_name,str(len(prop_attr))))
                for name,values in prop_attr.iteritems():
                    if isinstance(values,list):
                        result.extend([name,len(values)] + [str(x) for x in values])
                    else:
                        result.extend((name,'1',str(values)))
        return result

    @_info
    def get_device_attribute_property_hist(self, dev_name, attribute, prop_name):
        return []
    
    @_info
    def get_device_class_list(self, server_name):
        server_node = self._personal_2_node.get(server_name)
        if server_node is None:
            return []
        devices = server_node.get('device')
        if isinstance(devices,list):
            name_class = [(n.get('tango_name'),n.get('class')) for n in devices]
        else:
            name_class = [(devices.get('tango_name'),devices.get('class'))]

        return [x for p in name_class for x in p]

    @_info
    def get_device_domain_list(self, wildcard):
        filtered_names = _filter(wildcard,
                                 [n.get('tango_name') for n in self._tango_name_2_node.values()])
        res = list(set([x.split('/')[0] for x in  filtered_names]))
        res.sort()
        return res
   
    @_info
    def get_device_exported_list(self, wildcard):
        cache = settings.get_cache()
        return [x.replace('tango.','') for x in cache.keys('tango.%s' % wildcard)]

    @_info
    def get_device_family_list(self, wildcard):
        filtered_names = _filter(wildcard,
                                 [n.get('tango_name') for n in self._tango_name_2_node.values()])
        return list(set([x.split('/')[1] for x in  filtered_names]))
    
    def get_device_info(self, dev_name):
        dev_name = dev_name.lower()
        device_info = self._get_export_device_info(dev_name)
        device_node = self._tango_name_2_node.get(dev_name)
        result_long = []
        result_str = []
        
        info = device_info.get_all()
        if device_node:
            if dev_name.startswith('dserver'):
                server_node = device_node
            else:
                server_node = device_node.parent
            result_str.extend((dev_name,
                               info.get('IOR',''),
                               str(info.get('version','0')),
                               server_node.get('server','') + '/' + server_node.get('personal_name',''),
                               info.get('host','?'),info.get('start-time','?'),'?',
                               device_node.get('class','DServer')))
            result_long.extend((info and 1 or 0,info.get('pid',-1)))
        return (result_long,result_str)

    @_info
    def get_device_list(self,server_name, class_name):
        if server_name == '*':
            r_list = list()
            for server_node in self._personal_2_node.values():
                device_list = server_node.get('device')
                r_list.extend(self._tango_name_from_class(device_list,class_name))
            return r_list

        server_node = self._personal_2_node.get(server_name)
        if server_node is None:
            return []
        device_list = server_node.get('device')
        return self._tango_name_from_class(device_list,class_name)
    
    def _tango_name_from_class(self,device_list,class_name):
        m = re.compile(class_name.replace('*','.*'),re.IGNORECASE)
        if isinstance(device_list,list) :
            return [x.get('tango_name') for x in device_list if m.match(x.get('class',''))]
        elif isinstance(device_list,dict) and m.match(device_list.get('class','')) :
            return [device_list.get('tango_name')]
        else:
            return []
    
    @_info
    def get_device_wide_list(self, wildcard):
        return _filter(wildcard,self._tango_name_2_node.keys())
    
    @_info
    def get_device_member_list(self, wildcard):
        wildcard = wildcard.lower()
        filtered_names = _filter(wildcard,self._tango_name_2_node.keys())
        return list(set([x.split('/')[-1] for x in filtered_names]))
    
    def _get_property_node(self,dev_name) :
        dev_name = dev_name.lower()
        device_node = self._tango_name_2_node.get(dev_name)
        if device_node is None:
            properties = {}
        else:
            properties = device_node.get('properties')

        if isinstance(properties,str) : # reference
            properties_key = properties.split('/')
            node_refname = properties_key[0]
            property_node = self._config.get_config(node_refname)
            if properties_key == node_refname:
                properties = property_node
            else:
                for key in properties_key[1:]:
                    property_node = property_node.get(key)
                    if property_node is None:
                        break
                properties = property_node
        return properties

    @_info
    def get_device_property(self, dev_name, properties_query_list):
        properties = self._get_property_node(dev_name)
        
        if properties is None:
            result = [dev_name,str(len(properties_query_list))]
            for p_name in properties_query_list:
                result.extend([p_name,'0',''])
            return result
        else:
            nb_properties = 0
            properties_array = []
            properties_key = properties.keys()
            for property_name in properties_query_list:
                ask_keys = _filter(property_name,properties_key)
                if not ask_keys:
                    properties_array.extend([property_name,'0',''])
                    nb_properties += 1

                for key in ask_keys:
                    values = properties.get(key,'')
                    if isinstance(values,list):
                        values = [str(x) for x in values]
                        properties_array.extend([property_name,str(len(values))] + values)
                    else:
                        properties_array.extend([property_name,'1',str(values)])
  
                nb_properties += len(ask_keys)
            return [dev_name,str(nb_properties)] + properties_array

    @_info
    def get_device_property_list(self,device_name,prop_filter) :
        properties = self._get_property_node(device_name)
        if properties is None:
            return []
        else:
            return _filter(prop_filter,[k for k,v in properties.iteritems() if not isinstance(v,dict)])

    @_info
    def get_device_property_hist(self, device_name, prop_name):
        return []

    @_info
    def get_device_server_class_list(self, server_name):
        server_name = server_name.lower()
        server_node = self._personal_2_node.get(server_name)
        if server_node is None:
            return []
        else:
            devices = server_node.get('device')
            if isinstance(devices,list):
                return [x.get('class') for x in devices]
            else:
                return [devices.get('class')]
   
    @_info
    def get_exported_device_list_for_class(self, class_name):
        result = []
        cache = settings.get_cache()
        exported_devices = cache.keys('tango.info.*')
        for exp_dev in exported_devices:
            dev_name = exp_dev.replace('tango.info.','')
            dev_node = self._tango_name_2_node.get(dev_name)
            if dev_node:
                dev_class_name = dev_node.get('class')
                if dev_class_name == class_name:
                    result.append(dev_name)
        return result
    
    @_info
    def get_host_list(self, host_name):
        cache = settings.get_cache()
        host_list =  [settings.HashSetting(key_name).get('host') for key_name in cache.keys('tango.info.*')]
        return _filter(host_name,host_list)
    
    @_info
    def get_host_server_list(self, host_name):
        result = []
        wildcard = host_name.replace('*','.*')
        m = re.compile(wildcard)
        cache = settings.get_cache()
        for key_name in cache.keys('tango.info.*'):
            host = settings.HashSetting(key_name).get('host')
            if not m.match(host): continue
            dev_name = key_name.replace('tango.info.','')
            dev_node = self._tango_name_2_node.get(dev_name)
            if dev_node is None: continue
            server_node = dev_node.parent
            result.append('%s/%s' % (server_node.get('server',''),server_node.get('personal_name')))
        return result
     
    @_info
    def get_host_servers_info(self, host_name):
        #Don't know what it is?
        return []

     
    @_info
    def get_instance_name_list(self, server_name):
        server_name = server_name + "\*"
        server_list = self.get_server_list(server_name)
        result = []
        for server in server_list:
            names = server.split("/")
            result.append(names[1])
        return result

    @_info
    def get_object_list(self, name):
        return []

    @_info
    def get_property(self, object_name, properties):
        result = [object_name,str(len(properties))]
        for prop in properties:
            result.extend([prop,'0',''])
        return result


    @_info
    def get_property_hist(self, object_name, prop_name):
        return []

    @_info
    def get_property_list(self, object_name, wildcard):
        return []

    @_info
    def get_server_info(self, server_name):
        return ["","",""]
     
    @_info
    def get_server_list(self, wildcard):
        return _filter(wildcard,self._personal_2_node.keys())
    
    @_info
    def get_server_name_list(self,wildcard) :
        res = list(set(_filter(wildcard,[x.split('/')[0] for x in self._personal_2_node.keys()])))
        res.sort()
        return res
    @_info
    def get_server_class_list(self,wildcard):
        server_names = _filter(wildcard,self._personal_2_node.keys())
        result = set()
        for ser_name in server_names:
            server_node = self._personal_2_node.get(ser_name)
            for device_node in server_node.get('device',[]) :
                class_name = device_node.get('class')
                if class_name is not None:
                    result.add(class_name)

        result.add('DServer')
        result = list(result)
        result.sort()
        return result

    def import_device(self, dev_name):
        dev_node = self._tango_name_2_node.get(dev_name)
        if dev_node is not None:
            return self.get_device_info(dev_name)
        else:
            th_exc(DB_DeviceNotDefined,
                   "device " + dev_name + " not defined in the database !",
                   "DataBase::ImportDevice()")
    @_info
    def import_event(self, event_name):
        th_exc(DB_DeviceNotDefined,
               "event " + event_name + " not defined in the database !",
               "DataBase::ImportEvent()")
     
    @_info
    def info(self):
        return ["Beacon Beacon files"]
         
    @_info
    def put_attribute_alias(self, attribute_name, attr_alias_name):
        attr_alias = settings.HashObjSetting(self.TANGO_ATTR_ALIAS)
        attr_alias_info = attr_alias.get(attr_alias_name)
        if attr_alias_info is not None:
            self.warn_stream("DataBase::DbPutAttributeAlias(): this alias exists already ")
            th_exc(DB_SQLError,
                   "alias " + attribute_alias + " already exists !",
                   "DataBase::DbPutAttributeAlias()")
        attr_alias[attr_alias_name] = attribute_name
         
    @_info
    def put_class_attribute_property(self, class_name, nb_attributes, attr_prop_list):
        attr_id = 0
        for k in range(nb_attributes):
            attr_name,nb_properties = attr_prop_list[attr_id],int(attr_prop_list[attr_id + 1])
            attr_id += 2
            class_properties = self._get_class_attribute(class_name,att_name)
            new_values = {}
            for prop_id in range(attr_id,attr_id + nb_properties * 2,2) :
                prop_name,prop_val = attr_prop_list[prop_id],attr_prop_list[prop_id + 1]
                new_values[prop_name] = prop_val
            attr_id += nb_properties * 2
            class_properties.set(new_values)

    @_info
    def put_class_attribute_property2(self, class_name, nb_attributes, attr_prop_list):
        attr_id = 0
        for j in range(nb_attributes) :
            attr_name,nb_properties = attr_prop_list[attr_id],attr_prop_list[attr_id + 1]
            attr_id += 2
            class_properties = self._get_class_attribute(class_name,att_name)
            new_values = {}
            for prop_id in range(nb_properties) :
                prop_name,prop_number = attr_prop_list[attr_id],int(attr_prop_list[attr_id + 1])
                attr_id += 2
                prop_values = []
                for prop_sub_id in range(prop_number):
                    prop_values.append(attr_prop_list[attr_id])
                    attr_id += 1
                if len(prop_values) == 1:
                    prop_values = prop_values[0]
                new_values[prop_name] = prop_values
            class_properties.set(new_values)

    @_info
    def put_class_property(self, class_name, nb_properties, attr_prop_list):
        attr_id = 0
        class_node = self._class_name_2_node.get(class_name)
        if class_node is None:
            class_node = static.Node(self._config,parent=self._config.root,
                                     filename = 'tango/%s.yml' % class_name.replace('/','_'))
            class_node['class'] = class_name
            self._strong_node_ref.add(class_node)
            self._class_name_2_node[class_name] = class_node
            
        properties = class_node.get('properties',dict())
        for k in range(nb_properties):
            prop_name,nb_values = attr_prop_list[attr_id],int(attr_prop_list[attr_id + 1])
            attr_id += 2
            if nb_values == 1:
                properties[prop_name] = attr_prop_list[attr_id]
            else:
                properties[prop_name] = list(attr_prop_list[attr_id:])
        class_node['properties'] = properties
        class_node.save()
    @_info
    def put_device_alias(self, device_name, device_alias):
        device_node = self._tango_name_2_node.get(device_name)
        device_node['alias'] = device_alias
        device_node.save()

    @_info
    def put_device_attribute_property(self, device_name, nb_attributes, attr_prop_list):
        attr_id = 0
        prop_attr_device = self._get_property_attr_device(device_name)
        for k in range(nb_attributes):
            attr_name,prop_nb = attr_prop_list[attr_id],int(attr_prop_list[attr_id + 1])
            attr_id += 2
            new_values = {}
            for prop_id in range(attr_id,attr_id + prop_nb * 2,2) :
                prop_name,prop_val = attr_prop_list[prop_id],attr_prop_list[prop_id + 1]
                new_values[prop_name] = prop_val
            prop_attr_device[attr_name] = new_values
            attr_id += prop_nb * 2

    @_info
    def put_device_attribute_property2(self, device_name, nb_attributes, attr_prop_list):
        attr_id = 0
        prop_attr_device = self._get_property_attr_device(device_name)
        for k in range(nb_attributes):
            attr_name,prop_nb = attr_prop_list[attr_id],int(attr_prop_list[attr_id + 1])
            attr_id += 2
            new_values = {}
            for prop_id in range(prop_nb) :
                prop_name,prop_nb = attr_prop_list[attr_id],int(attr_prop_list[attr_id + 1])
                attr_id += 2
                prop_values = []
                for prop_sub_id in range(prop_nb):
                    prop_values.append(attr_prop_list[attr_id])
                    attr_id += 1
                if len(prop_values) == 1:
                    prop_values = prop_values[0]
                new_values[prop_name] = prop_values
            prop_attr_device[attr_name] = new_values

    @_info
    def put_device_property(self, device_name, nb_properties, attr_prop_list):
        device_name = device_name.lower()
        device_node = self._tango_name_2_node.get(device_name)
        old_properties = device_node.get('properties')
        if isinstance(old_properties,str): #reference
            properties_key = old_properties.split('/')
            node_refname = properties_key[0]
            property_node = self._config.get_config(node_refname)
            if properties_key == node_refname:
                old_properties = property_node
            else:
                for key in properties_key[1:]:
                    property_node = property_node.get(key)
                    if property_node is None:
                        break
                old_properties = property_node
        if old_properties is None:
            properties = static.Node(self._config,parent=device_node)
            device_node['properties'] = properties
        else:
            properties = old_properties

        id_prop = 0
        for i in range(nb_properties) :
            prop_name,prop_nb_values = attr_prop_list[id_prop],int(attr_prop_list[id_prop + 1])
            id_prop += 2
            if prop_nb_values == 1:
                properties[prop_name] = attr_prop_list[id_prop]
            else:
                properties[prop_name] = attr_prop_list[id_prop:id_prop + prop_nb_values]
            id_prop += prop_nb_values
        properties.save()

    @_info
    def put_property(self, object_name, nb_properties, attr_prop_list):
        #Not use in our case
        pass

    @_info
    def put_server_info(self, tmp_server, tmp_host, tmp_mode, tmp_level, tmp_extra):
        #Not use in our case
        pass

    def unexport_device(self, dev_name):
        device_info = self._get_export_device_info(dev_name)
        device_info.clear()
        
    @_info
    def unexport_event(self, event_name):
        #Not use in our case
        pass

    @_info
    def unexport_server(self, server_name):
        server_node = self._personal_2_node.get(server_name)
        if server_node is None: return

        for device in server_node.get('device'):
            tango_name = device.get('tango_name')
            if tango_name:
                self.unexport_device(tango_name)

    @_info
    def delete_all_device_attribute_property(self, dev_name, attr_list):
        prop_attr_device = self._get_property_attr_device(dev_name)
        for attr_name in attr_list:
            del prop_attr_device[attr_name]
                
    @_info
    def my_sql_select(self, cmd):
        #todo see if it's really needed
        return ([0,0],[])


    @_info
    def get_csdb_server_list(self):
        cache = settings.get_cache()
        exported_devices = cache.keys('tango.info.sys/database*')
        result = []
        for key_name in exported_devices:
            info = settings.HashSetting(key_name)
            result.append(info.get('IOR'))
        return result
     
    @_info
    def get_attribute_alias2(self, attr_name):
        attr_alias = settings.HashObjSetting(self.TANGO_ATTR_ALIAS)
        result = []
        for alias,name in attr_alias.iteritems():
            if name == attr_name:
                result.append(alias)
        return result
    
    @_info
    def get_alias_attribute(self, alias_name):
        attr_alias = settings.HashObjSetting(self.TANGO_ATTR_ALIAS)
        attr_name = attr_alias.get(alias_name)
        return attr_name and [attr_name] or []
    
    @_info
    def rename_server(self, old_name, new_name):
        device_node = self._tango_name_2_node.get(new_name)
        if device_node is not None:
            th_exc(DB_SQLError,
                   "Device server process name " + attribute_alias + "is already used !",
                   "DataBase::DbRenameServer()")
        device_node = self._tango_name_2_node.pop(old_name)
        device_node['tango_name'] = new_name
        self._tango_name_2_node[new_name] = device_node
        device_node.save()
   

def get_db(personal_name = "",**keys):
    return beacon(personal_name = personal_name)

def get_wildcard_replacement():
    return False
