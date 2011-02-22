#############################################################################
##
## This file is part of PyTango, a python binding for Tango
##
## http://www.tango-controls.org/static/PyTango/latest/doc/html/index.html
##
## (copyleft) CELLS / ALBA Synchrotron, Bellaterra, Spain
##
## This is free software; you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This software is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with this program; if not, see <http://www.gnu.org/licenses/>.
###########################################################################

import os
import sys
import errno
import platform

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup
from setuptools import Extension, Distribution

#from distutils.core import setup, Extension
#from distutils.dist import Distribution
import distutils.sysconfig


try:
    import sphinx
except:
    sphinx = None

try:
    import IPython
    import IPython.genutils
    _IPY_ROOT = os.path.dirname(os.path.abspath(IPython.__file__))
    _IPY_LOCAL = str(IPython.genutils.get_ipython_dir())
except:
    IPython = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PyTango'))

from release import Release

BOOST_ROOT = OMNI_ROOT = TANGO_ROOT = NUMPY_ROOT = '/usr'

TANGO_ROOT = os.environ.get('TANGO_ROOT', TANGO_ROOT)
OMNI_ROOT  = os.environ.get('OMNI_ROOT', OMNI_ROOT)
BOOST_ROOT = os.environ.get('BOOST_ROOT', BOOST_ROOT)
NUMPY_ROOT = os.environ.get('NUMPY_ROOT', NUMPY_ROOT)

# if there is no numpy then for sure disable usage of it in PyTango

numpy_capi_available = os.path.isdir(os.path.join(NUMPY_ROOT, 'include','numpy'))

numpy_available = False
try:
    import numpy
    numpy_available = True
except Exception, e:
    pass

print '-- Compilation information -------------------------------------------'
print 'Build %s %s' % (Release.name, Release.version_long)
print 'Using Python %s' % distutils.sysconfig.get_python_version()
print '\tinclude: %s' % distutils.sysconfig.get_python_inc()
print '\tlibrary: %s' % distutils.sysconfig.get_python_lib()
print 'Using omniORB from %s' % OMNI_ROOT
print 'Using Tango from %s' % TANGO_ROOT
print 'Using boost python from %s' % BOOST_ROOT
if numpy_available:
    if numpy_capi_available:
        print 'Using numpy %s' % numpy.version.version
        print '\tinclude: %s' % os.path.join(NUMPY_ROOT, 'include','numpy')
    else:
        print 'NOT using numpy (numpy available but C source is not)'
else:
    print 'NOT using numpy (it is not available)'
print '----------------------------------------------------------------------'

author = Release.authors['Coutinho']

please_debug = False

packages = [
    'PyTango',
    'PyTango.ipython',
    'PyTango3'
]

provides = [
    'PyTango',
]

requires = [
    'boost_python (>=1.33)',
    'numpy (>=1.1)'
]

package_data = {
    'PyTango' : []
}

data_files = []

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Other Environment',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
    'Natural Language :: English',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',
]

def uniquify(seq):
    no_dups = []
    [ no_dups.append(i) for i in seq if not no_dups.count(i) ]
    return no_dups

include_dirs = [
    os.path.abspath('src'),
    os.path.join(TANGO_ROOT, 'include'),
    os.path.join(OMNI_ROOT, 'include'),
    os.path.join(NUMPY_ROOT, 'include'),
]

libraries = [
        'tango',
        'log4tango',
    ]

extra_compile_args = []

extra_link_args = []

macros = []

if not numpy_available or not numpy_capi_available:
    macros.append( ('DISABLE_PYTANGO_NUMPY', None) )

library_dirs = [
    os.path.join(TANGO_ROOT, 'lib'),
    os.path.join(BOOST_ROOT, 'lib'),
]

if os.name == 'nt':
    include_dirs += [ BOOST_ROOT ]

    if please_debug:
        libraries += [
            #'libboost_python-vc80-mt-1_38', Boost in windows autodetects the
            #proper library to link itself with...
            'omniORB414d_rt',
            'omniDynamic414d_rt',
            'omnithread34d_rt',
            'COS414d_rt',
        ]
        extra_compile_args += []
        extra_link_args += ['/DEBUG']
        macros += [ ('_DEBUG', None) ]
    else:
        libraries += [
            #'libboost_python-vc80-mt-1_38', Boost in windows autodetects the
            #proper library to link itself with...
            'omniORB414_rt',
            'omniDynamic414_rt',
            'omnithread34_rt',
            'COS414_rt',
        ]

    library_dirs += [ os.path.join(OMNI_ROOT, 'lib', 'x86_win32') ]
    
    extra_compile_args += [
        '/EHsc'
    ]

    extra_link_args += []

    macros += [
        #('_WINDOWS', None),
        #('_USRDLL', None),
        #('_TANGO_LIB', None),
        #('JPG_USE_ASM', None),
        ('LOG4TANGO_HAS_DLL', None),
        ('TANGO_HAS_DLL', None),
        ('WIN32', None),
    ]

else:
    if please_debug:
        extra_compile_args += ['-g', '-O0']
        extra_link_args += ['-g' , '-O0']
    
    include_dirs += [ os.path.join(BOOST_ROOT, 'include') ]
    
    libraries += [
        'pthread',
        'rt',
        'dl',
        'omniORB4',
        'omniDynamic4',
        'omnithread',
        'COS4',
    ]

    # when building with multiple version of python on debian we need
    # to link against boost_python-py25/-py26 etc...
    pyver = "py" + "".join(map(str, platform.python_version_tuple()[:2]))
    dist = platform.dist()[0].lower()
    if dist in ['debian']:
        libraries.append('boost_python-' + pyver)
    else:
        libraries.append('boost_python')

    library_dirs += [ os.path.join(OMNI_ROOT, 'lib') ]


    # Note for PyTango developers:
    # Compilation time can be greatly reduced by compiling the file
    # src/precompiled_header.hpp as src/precompiled_header.hpp.gch
    # and then uncommenting this line. Someday maybe this will be
    # automated...
    extra_compile_args += [
#        '-includesrc/precompiled_header.hpp',
    ]

    #if not please_debug:
    #    extra_compile_args += [ '-g0' ]

    extra_link_args += [
        '-Wl,-h',
        '-Wl,--strip-all',
    ]

    macros += []

include_dirs = uniquify(include_dirs)
library_dirs = uniquify(library_dirs)

_cppfiles_exclude = []
_cppfiles  = [ os.path.join('src',fname) for fname in os.listdir('src') if fname.endswith('.cpp') and not fname in _cppfiles_exclude]
_cppfiles += [ os.path.join('src','server',fname) for fname in os.listdir(os.path.join('src','server')) if fname.endswith('.cpp') and not fname in _cppfiles_exclude]

_pytango = Extension(name               = '_PyTango',
                     sources            = _cppfiles,
                     include_dirs       = include_dirs,
                     library_dirs       = library_dirs,
                     libraries          = libraries,
                     define_macros      = macros,
                     extra_compile_args = extra_compile_args,
                     extra_link_args    = extra_link_args,
                     language           = 'c++',
                     depends            = []
                     )

from setuptools import Command
#from distutils.cmd import Command
from distutils.command.build import build as dftbuild
from distutils.command.build_ext import build_ext as dftbuild_ext
from distutils.unixccompiler import UnixCCompiler

class build(dftbuild):

    def has_doc(self):
        if sphinx is None: return False
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.isdir(os.path.join(setup_dir, 'doc'))

    def has_ipython(self):
        return IPython is not None

    sub_commands = dftbuild.sub_commands + [('build_doc', has_doc), ('build_spock', has_ipython)]

cmdclass = {'build' : build }

class build_ext(dftbuild_ext): 
    
    def build_extensions(self):
        if isinstance(self.compiler, UnixCCompiler):
            compiler_pars = self.compiler.compiler_so
            while '-Wstrict-prototypes' in compiler_pars:
                del compiler_pars[compiler_pars.index('-Wstrict-prototypes')]
            #self.compiler.compiler_so = " ".join(compiler_pars)
        dftbuild_ext.build_extensions(self)

cmdclass['build_ext'] = build_ext

if sphinx:
    from sphinx.setup_command import BuildDoc

    class build_doc(BuildDoc):
        
        def run(self):
            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version
            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))
            sphinx.setup_command.BuildDoc.run(self)
            sys.path.pop(0)
    
    cmdclass['build_doc'] = build_doc

if IPython:
    class build_spock(Command):
        
        description = "Build Spock, the PyTango's IPython extension"

        user_options = [
            ('ipython-local', None, "install spock as current user profile instead of as an ipython extension"),
            ('ipython-dir=', None, "Location of the ipython installation. (Defaults to '%s' if ipython-local is NOT set or to '%s' otherwise" % (_IPY_ROOT, _IPY_LOCAL) ) ]

        boolean_options = [ 'ipython-local' ]

        def initialize_options (self):
            self.ipython_dir = None
            self.ipython_local = False
        
        def finalize_options(self):
            if self.ipython_dir is None:
                if self.ipython_local:
                    global _IPY_LOCAL
                    self.ipython_dir = _IPY_LOCAL
                else:
                    global _IPY_ROOT
                    self.ipython_dir = os.path.join(_IPY_ROOT, "Extensions")
            else:
                if ipython-local:
                    self.warn("Both options 'ipython-dir' and 'ipython-local' were given. " \
                              "'ipython-dir' will be used.")
            self.ensure_dirname('ipython_dir')
        
        def run(self):
            added_path = False
            try:
                # make sure the python path is pointing to the newly built
                # code so that the documentation is built on this and not a
                # previously installed version
                build = self.get_finalized_command('build')
                sys.path.insert(0, os.path.abspath(build.build_lib))
                added_path=True
                import PyTango.ipython
                PyTango.ipython.install(self.ipython_dir, verbose=False)
            except IOError, ioerr:
                self.warn("Unable to install Spock IPython extension. Reason:")
                self.warn(str(ioerr))
                if ioerr.errno == errno.EACCES:
                    self.warn("Probably you don't have enough previledges to install spock as an ipython extension.")
                    self.warn("Try executing setup.py with sudo or otherwise give '--ipython-local' parameter to")
                    self.warn("setup.py to install spock as a current user ipython profile.")
                    self.warn("type: setup.py --help build_spock for more information")
            except Exception, e:
                self.warn("Unable to install Spock IPython extension. Reason:")
                self.warn(str(e))
                
            if added_path:
                sys.path.pop(0)
            
    cmdclass['build_spock'] = build_spock
            
dist = setup(
    name             = 'PyTango',
    version          = Release.version,
    description      = Release.description,
    long_description = Release.long_description,
    author           = author[0],
    author_email     = author[1],
    url              = Release.url,
    download_url     = Release.download_url,
    platforms        = Release.platform,
    license          = Release.license,
    packages         = packages,
    package_dir      = { 'PyTango' : 'PyTango', 'PyTango3' : 'PyTango3' },
    classifiers      = classifiers,
    package_data     = package_data,
    data_files       = data_files,
    provides         = provides,
    keywords         = Release.keywords,
    requires         = requires,
    ext_package      = 'PyTango',
    ext_modules      = [_pytango],
    cmdclass         = cmdclass)
