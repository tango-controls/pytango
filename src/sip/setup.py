# ------------------------------------------------------------------------------
# This file is part of PyTango (http://www.tinyurl.com/PyTango)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

import os
import sys
import imp
import struct
import platform
import subprocess

from distutils.core import setup, Extension
from distutils.unixccompiler import UnixCCompiler
from distutils.version import StrictVersion as V

import sipdistutils


def abspath(*path):
    """A method to determine absolute path for a given relative path to the
    directory where this setup.py script is located"""
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(setup_dir, *path)


def get_release_info():
    import release
    return release.Release


def uniquify(seq):
    no_dups = []
    [ no_dups.append(i) for i in seq if not no_dups.count(i) ]
    return no_dups


class build_ext(sipdistutils.build_ext):

    def build_extensions(self):
        self.use_cpp_0x = False
        if isinstance(self.compiler, UnixCCompiler):
            compiler_pars = self.compiler.compiler_so
            while '-Wstrict-prototypes' in compiler_pars:
                del compiler_pars[compiler_pars.index('-Wstrict-prototypes')]
            #self.compiler.compiler_so = " ".join(compiler_pars)

            # mimic tango check to activate C++0x extension
            compiler = self.compiler.compiler
            pipe = subprocess.Popen(compiler + ["-dumpversion"], stdout=subprocess.PIPE).stdout
            gcc_ver = pipe.readlines()[0].decode().strip()
            if V(gcc_ver) >= V("4.3.3"):
                self.use_cpp_0x = True
        sipdistutils.build_ext.build_extensions(self)

    def build_extension(self, ext):
        if self.use_cpp_0x:
            ext.extra_compile_args += ['-std=c++0x']
            ext.define_macros += [ ('PYTANGO_HAS_UNIQUE_PTR', '1') ]
        sipdistutils.build_ext.build_extension(self, ext)


def main():
    ZMQ_ROOT = LOG4TANGO_ROOT = OMNI_ROOT = TANGO_ROOT = '/usr'

    TANGO_ROOT = os.environ.get('TANGO_ROOT', TANGO_ROOT)
    OMNI_ROOT = os.environ.get('OMNI_ROOT', OMNI_ROOT)
    LOG4TANGO_ROOT = os.environ.get('LOG4TANGO_ROOT', LOG4TANGO_ROOT)
    ZMQ_ROOT = os.environ.get('ZMQ_ROOT', ZMQ_ROOT)

    Release = get_release_info()

    author = Release.authors['Coutinho']

    please_debug = False

    packages = [
        'Tango',
    ]

    provides = [
        'Tango',
    ]

    requires = [
        'sip (>=4.10)',
        'numpy (>=1.1)'
    ]

    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Library or Lesser General Public License v3 (LGPLv3)',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
    ]

    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
    # include directories
    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

    include_dirs = [ ]

    _tango_root_inc = os.path.join(TANGO_ROOT, 'include')
    include_dirs.append(_tango_root_inc)

    # $TANGO_ROOT/include/tango exists since tango 7.2.0
    # we changed the PyTango code include statements from:
    # #include <tango.h> to:
    # #include <tango/tango.h>
    # However tango itself complains that it doesn't know his own header files
    # if we don't add the $TANGO_ROOT/include/tango directory to the path. So we do it
    # here
    _tango_root_inc = os.path.join(_tango_root_inc, 'tango')
    if os.path.isdir(_tango_root_inc):
        include_dirs.append(_tango_root_inc)

    include_dirs.append(os.path.join(OMNI_ROOT, 'include'))
    include_dirs.append(os.path.join(LOG4TANGO_ROOT, 'include'))

    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
    # library directories
    #-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

    libraries = [
        'tango',
        'log4tango',
        'zmq',
    ]

    extra_compile_args = []

    extra_link_args = []

    macros = []

    library_dirs = []
    for f in (TANGO_ROOT, LOG4TANGO_ROOT, ZMQ_ROOT):
        is64 = 8 * struct.calcsize("P") == 64
        d = os.path.join(f, 'lib')
        if is64:
            d = os.path.join(f, 'lib64')
            try:
                if not os.stat(d): raise Exception('%s_doesnt_exist' % d)
            except: d = os.path.join(f, 'lib')
        library_dirs.append(d)

    if os.name == 'nt':
        include_dirs += [ ]

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
            '/EHsc',
            '/wd4005',  # supress redefinition of HAVE_STRFTIME between python and omniORB
            '/wd4996',  # same as /D_SCL_SECURE_NO_WARNINGS
            '/wd4250',  # supress base class inheritance warning
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

        libraries += [
            'pthread',
            'rt',
            'dl',
            'omniORB4',
            'omniDynamic4',
            'omnithread',
            'COS4',
        ]

        is64 = 8 * struct.calcsize("P") == 64
        omni_lib = os.path.join(OMNI_ROOT, 'lib')
        if is64:
            omni_lib = os.path.join(OMNI_ROOT, 'lib64')
            try:
                if not os.stat(d): raise Exception('%s_doesnt_exist' % d)
            except:
                omni_lib = os.path.join(OMNI_ROOT, 'lib')
        library_dirs += [ omni_lib ]


        # Note for PyTango developers:
        # Compilation time can be greatly reduced by compiling the file
        # src/precompiled_header.hpp as src/precompiled_header.hpp.gch
        # and then uncommenting this line. Someday maybe this will be
        # automated...
        extra_compile_args += [
#            '-includesrc/precompiled_header.hpp',
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
    src_dir = abspath('.')
    _cppfiles = [ os.path.join(src_dir, fname) for fname in os.listdir(src_dir) if fname.endswith('.cpp') ]
    _cppfiles.sort()
    sources = ["Tango.sip"] + _cppfiles

    cmdclass = {'build_ext': build_ext}

    _tango = Extension(
        name='Tango',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
        depends=[])

    dist = setup(
        name='Tango',
        version=Release.version,
        description=Release.description,
        long_description=Release.long_description,
        author=author[0],
        author_email=author[1],
        url=Release.url,
        download_url=Release.download_url,
        platforms=Release.platform,
        license=Release.license,
        packages=packages,
        package_dir={ 'Tango' : abspath(".") },
        classifiers=classifiers,
        provides=provides,
        keywords=Release.keywords,
        requires=requires,
        ext_modules=[_tango],
        cmdclass=cmdclass)

if __name__ == "__main__":
    main()
