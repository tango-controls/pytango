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

try:
    import IPython
except:
    IPython = None

def get_ipython_version():
    """Returns the current IPython version"""
    if IPython is None:return None
    v = None
    try:
        try:
            v = IPython.Release.version
        except Exception, e1:
            try:
                v = IPython.release.version
            except Exception, e2:
                pass
    except Exception, e3:
        pass
    return v

def default_init_ipython(ip, store=True, pytango=True, colors=True, console=True, magic=True):
    print "Unsupported IPython version (%s) for spock profile" % get_ipython_version()
    print "Supported IPython versions are: 0.10"
    print "Starting normal IPython console..."

def __define_init():
    _ipv_str = get_ipython_version()

    if _ipv_str is None:
        _ipv = 0,0
    else:
        _ipv = tuple(map(int,_ipv_str.split(".")[:3]))

    ret = default_init_ipython
    if _ipv >= (0,10) and _ipv <= (0,11):
        import ipython_00_10
        ret = ipython_00_10.init_ipython
    return ret

def get_ipython_dir():
    """Find the ipython local directory. Usually is <home>/.ipython"""
    if hasattr(IPython.iplib, 'get_ipython_dir'):
        # Starting from ipython 0.9 they hadded this method
        return IPython.iplib.get_ipython_dir()
    
    # Try to find the profile in the current directory and then in the 
    # default IPython dir
    userdir = os.path.realpath(os.path.curdir)
    home_dir = IPython.genutils.get_home_dir()
    
    if os.name == 'posix':
        ipdir = '.ipython'
    else:
        ipdir = '_ipython'
    ipdir = os.path.join(home_dir, ipdir)
    ipythondir = os.path.abspath( os.environ.get('IPYTHONDIR', ipdir) )
    return ipythondir

def get_ipython_profiles():
    """Helper functions to find ipython profiles"""
    ret = []
    ipydir = get_ipython_dir()
    if os.path.isdir(ipydir):
        for i in os.listdir(ipdir):
            if i.startswith("ipy_profile_") and i.endswith(".py") and \
                os.path.isfile(i):
                ret.append(i[len("ipy_profile_"):s.rfind(".")])
    return ret

init_ipython = __define_init()
