/*******************************************************************************

   This file is part of PyTango, a python binding for Tango

   http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html

   Copyright 2011 CELLS / ALBA Synchrotron, Bellaterra, Spain
   
   PyTango is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   PyTango is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.
  
   You should have received a copy of the GNU Lesser General Public License
   along with PyTango.  If not, see <http://www.gnu.org/licenses/>.
   
*******************************************************************************/

class Connection {

%TypeHeaderCode
#include <tango.h>
using namespace Tango;
%End

protected:

	virtual std::string get_corba_name(bool)=0;
	virtual std::string build_corba_name()=0;
	virtual int get_lock_ctr()=0;
	virtual void set_lock_ctr(int)=0;

public:
	virtual std::string dev_name()=0;

	std::string &get_db_host();
	std::string &get_db_port();
	int get_db_port_num();
	bool get_from_env_var();
	static void get_fqdn(std::string &);

	bool is_dbase_used();
	std::string &get_dev_host();
	std::string &get_dev_port();

	int get_idl_version();

	virtual void set_timeout_millis(int timeout);
	virtual int get_timeout_millis();
	virtual DevSource get_source();
	virtual void set_source(DevSource );
	virtual void set_transparency_reconnection(bool val);
	virtual bool get_transparency_reconnection();

	AccessControlType get_access_control();
	void set_access_control(AccessControlType acc);
	AccessControlType get_access_right();
};
