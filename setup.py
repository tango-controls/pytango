# ------------------------------------------------------------------------------
# This file is part of PyTango (http://pytango.rtfd.io)
#
# Copyright 2006-2012 CELLS / ALBA Synchrotron, Bellaterra, Spain
# Copyright 2013-2014 European Synchrotron Radiation Facility, Grenoble, France
#
# Distributed under the terms of the GNU Lesser General Public License,
# either version 3 of the License, or (at your option) any later version.
# See LICENSE.txt for more info.
# ------------------------------------------------------------------------------

# pylint: disable=deprecated-method

import os
import sys
import runpy
import struct
import subprocess

from setuptools import setup, Extension
from setuptools import Command
from setuptools.command.build_ext import build_ext as dftbuild_ext
from setuptools.command.install import install as dftinstall

from distutils.command.build import build as dftbuild
from distutils.unixccompiler import UnixCCompiler
from distutils.version import LooseVersion as V

# Distro import
try:
    from pip._vendor import distro
except ImportError:
    import platform as distro

# Sphinx imports
try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda: False
    from sphinx.setup_command import BuildDoc
except ImportError:
    sphinx = None

# Detect numpy
try:
    import numpy
except ImportError:
    numpy = None

# Platform constants
POSIX = 'posix' in os.name
WINDOWS = 'nt' in os.name
IS64 = 8 * struct.calcsize("P") == 64
PYTHON_VERSION = sys.version_info
PYTHON2 = (2,) <= PYTHON_VERSION < (3,)
PYTHON3 = (3,) <= PYTHON_VERSION < (4,)

# Linux distribution
distribution = distro.linux_distribution()[0].lower() if POSIX else ""
distribution_match = lambda names: any(x in distribution for x in names)
DEBIAN = distribution_match(['debian', 'ubuntu', 'mint'])
REDHAT = distribution_match(['redhat', 'fedora', 'centos', 'opensuse'])
GENTOO = distribution_match(['gentoo'])

# Arguments
TESTING = any(x in sys.argv for x in ['test', 'pytest'])


def get_readme(name='README.rst'):
    """Get readme file contents without the badges."""
    with open(name) as f:
        return '\n'.join(
            line for line in f.read().splitlines()
            if not line.startswith('|') or not line.endswith('|'))


def pkg_config(*packages, **config):
    config_map = {
        "-I": "include_dirs",
        "-L": "library_dirs",
        "-l": "libraries",
    }
    cmd = ["pkg-config", "--cflags-only-I",
           "--libs-only-L", "--libs-only-l", " ".join(packages)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    result = proc.wait()
    result = str(proc.communicate()[0].decode("utf-8"))
    for elem in result.split():
        flag, value = elem[:2], elem[2:]
        config_values = config.setdefault(config_map.get(flag), [])
        if value not in config_values:
            config_values.append(value)
    return config


def abspath(*path):
    """A method to determine absolute path for a given relative path to the
    directory where this setup.py script is located"""
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(setup_dir, *path)


def get_release_info():
    namespace = runpy.run_path(
        abspath('tango/release.py'),
        run_name='tango.release')
    return namespace['Release']


def uniquify(seq):
    no_dups = []
    for elem in seq:
        if elem not in no_dups:
            no_dups.append(elem)
    return no_dups


def get_c_numpy():
    if numpy is None:
        return
    else:
        get_include = getattr(numpy, "get_include", None)
        if get_include is None:
            get_include = getattr(numpy, "get_numpy_include", None)
            if get_include is None:
                return
        inc = get_include()
        if os.path.isdir(inc):
            return inc


def has_c_numpy():
    return get_c_numpy() is not None


def has_numpy(with_src=True):
    ret = numpy is not None
    if with_src:
        ret &= has_c_numpy()
    return ret


def add_lib(name, dirs, sys_libs,
            env_name=None, lib_name=None, inc_suffix=None):
    if env_name is None:
        env_name = name.upper() + '_ROOT'
    ENV = os.environ.get(env_name)
    if lib_name is None:
        lib_name = name
    if ENV is None:
        sys_libs.append(lib_name)
        return
    else:
        inc_dir = os.path.join(ENV, 'include')
        dirs['include_dirs'].append(inc_dir)
        if inc_suffix is not None:
            inc_dir = os.path.join(inc_dir, inc_suffix)
            dirs['include_dirs'].append(inc_dir)

        lib_dirs = [os.path.join(ENV, 'lib')]
        if IS64:
            lib64_dir = os.path.join(ENV, 'lib64')
            if os.path.isdir(lib64_dir):
                lib_dirs.insert(0, lib64_dir)
        dirs['library_dirs'].extend(lib_dirs)

        if lib_name.startswith('lib'):
            lib_name = lib_name[3:]
        dirs['libraries'].append(lib_name)


class build(dftbuild):

    user_options = list(dftbuild.user_options)

    # Strip library option
    user_options.append((
        'strip-lib',
        None,
        "strips the shared library of debugging symbols"
        " (Unix like systems only)"))

    # No documentation option
    user_options.append((
        'no-doc',
        None,
        "do not build documentation"))

    boolean_options = dftbuild.boolean_options + ['strip-lib', 'no-doc']

    def initialize_options(self):
        dftbuild.initialize_options(self)
        self.strip_lib = None
        self.no_doc = None

    def finalize_options(self):
        dftbuild.finalize_options(self)

    def run(self):
        if numpy is None:
            self.warn('NOT using numpy: it is not available')
        elif get_c_numpy() is None:
            self.warn("NOT using numpy: numpy available but C source is not")
        dftbuild.run(self)
        if self.strip_lib:
            self.strip_debug_symbols()

    def strip_debug_symbols(self):
        if not POSIX:
            return
        if os.system("type objcopy") != 0:
            return
        d = abspath(self.build_lib, "tango")
        orig_dir = os.path.abspath(os.curdir)
        so = "_tango.so"
        dbg = so + ".dbg"
        try:
            os.chdir(d)
            stripped_cmd = 'file %s | grep -q "not stripped" || exit 1' % so
            not_stripped = os.system(stripped_cmd) == 0
            if not_stripped:
                os.system("objcopy --only-keep-debug %s %s" % (so, dbg))
                os.system("objcopy --strip-debug --strip-unneeded %s" % (so,))
                os.system("objcopy --add-gnu-debuglink=%s %s" % (dbg, so))
                os.system("chmod -x %s" % (dbg,))
        finally:
            os.chdir(orig_dir)

    def has_doc(self):
        if self.no_doc:
            return False
        if sphinx is None:
            return False
        if V(sphinx.__version__) <= V("0.6.5"):
            print("Documentation will not be generated:"
                  " sphinx version (%s) too low."
                  " Needs 0.6.6" % sphinx.__version__)
            return False
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.isdir(os.path.join(setup_dir, 'doc'))

    sub_commands = dftbuild.sub_commands + [('build_doc', has_doc), ]


class build_ext(dftbuild_ext):

    def build_extensions(self):
        self.use_cpp_0x = False
        if isinstance(self.compiler, UnixCCompiler):
            compiler_pars = self.compiler.compiler_so
            while '-Wstrict-prototypes' in compiler_pars:
                del compiler_pars[compiler_pars.index('-Wstrict-prototypes')]
            # self.compiler.compiler_so = " ".join(compiler_pars)

            # mimic tango check to activate C++0x extension
            compiler = self.compiler.compiler
            proc = subprocess.Popen(
                compiler + ["-dumpversion"],
                stdout=subprocess.PIPE)
            pipe = proc.stdout
            proc.wait()
            gcc_ver = pipe.readlines()[0].decode().strip()
            if V(gcc_ver) >= V("4.3.3"):
                self.use_cpp_0x = True
        dftbuild_ext.build_extensions(self)

    def build_extension(self, ext):
        if self.use_cpp_0x:
            ext.extra_compile_args += ['-std=c++0x']
            ext.define_macros += [('PYTANGO_HAS_UNIQUE_PTR', '1')]
        dftbuild_ext.build_extension(self, ext)


if sphinx:
    class build_doc(BuildDoc):

        def run(self):
            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version

            build_cmd = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build_cmd.build_lib))
            sphinx.setup_command.BuildDoc.run(self)
            sys.path.pop(0)


class install_html(Command):

    user_options = []

    # Install directory option
    user_options.append((
        'install-dir=',
        'd',
        'base directory for installing HTML documentation files'))

    def initialize_options(self):
        self.install_dir = None

    def finalize_options(self):
        self.set_undefined_options(
            'install',
            ('install_html', 'install_dir'))

    def run(self):
        build_doc_cmd = self.get_finalized_command('build_doc')
        src_html_dir = abspath(build_doc_cmd.build_dir, 'html')
        self.copy_tree(src_html_dir, self.install_dir)


class install(dftinstall):

    user_options = list(dftinstall.user_options)

    # HTML directory option
    user_options.append((
        'install-html=',
        None,
        "installation directory for HTML documentation"))

    def initialize_options(self):
        dftinstall.initialize_options(self)
        self.install_html = None

    def finalize_options(self):
        dftinstall.finalize_options(self)
        # We do a hack here. We  cannot trust the 'install_base' value because
        # it  is not  always  the final  target.  For  example,  in unix,  the
        # install_base is '/usr' and all other install_* are directly relative
        # to  it. However,in  unix-local (like  ubuntu) install_base  is still
        # '/usr'  but,  for  example, install_data,  is  '$install_base/local'
        # which breaks everything.

        # The  hack consists  in  using install_data  instead of  install_base
        # since install_data seems to be, in practice, the proper install_base
        # on all different systems.
        if self.install_html is None:
            self.install_html = os.path.join(self.install_data,
                                             'share', 'doc', 'pytango', 'html')

    def has_html(self):
        return sphinx is not None

    sub_commands = list(dftinstall.sub_commands)
    sub_commands.append(('install_html', has_html))


def setup_args():

    directories = {
        'include_dirs': [],
        'library_dirs': [],
        'libraries': [],
    }
    sys_libs = []

    # Link specifically to libtango version 9
    tangolib = ':libtango.so.9' if POSIX else 'tango'
    directories['libraries'].append(tangolib)

    add_lib('omni', directories, sys_libs, lib_name='omniORB4')
    add_lib('zmq', directories, sys_libs, lib_name='libzmq')
    add_lib('tango', directories, sys_libs, inc_suffix='tango')

    # special boost-python configuration

    BOOST_ROOT = os.environ.get('BOOST_ROOT')
    boost_library_name = 'boost_python'
    if BOOST_ROOT is None:
        if DEBIAN:
            suffix = "-py{v[0]}{v[1]}".format(v=PYTHON_VERSION)
            boost_library_name += suffix
        elif REDHAT:
            if PYTHON3:
                boost_library_name += '3'
        elif GENTOO:
            suffix = "-{v[0]}.{v[1]}".format(v=PYTHON_VERSION)
            boost_library_name += suffix
    else:
        inc_dir = os.path.join(BOOST_ROOT, 'include')
        lib_dirs = [os.path.join(BOOST_ROOT, 'lib')]
        if IS64:
            lib64_dir = os.path.join(BOOST_ROOT, 'lib64')
            if os.path.isdir(lib64_dir):
                lib_dirs.insert(0, lib64_dir)

        directories['include_dirs'].append(inc_dir)
        directories['library_dirs'].extend(lib_dirs)

    directories['libraries'].append(boost_library_name)

    # special numpy configuration

    numpy_c_include = get_c_numpy()
    if numpy_c_include is not None:
        directories['include_dirs'].append(numpy_c_include)

    macros = []
    if not has_numpy():
        macros.append(('DISABLE_PYTANGO_NUMPY', None))
    else:
        macros.append(('PYTANGO_NUMPY_VERSION', '"%s"' % numpy.__version__))

    if POSIX:
        directories = pkg_config(*sys_libs, **directories)

    Release = get_release_info()

    author = Release.authors['Coutinho']

    please_debug = False

    packages = [
        'tango',
        'tango.databaseds',
        'tango.databaseds.db_access',
    ]

    py_modules = [
        'PyTango',  # Backward compatibilty
    ]

    provides = [
        'tango',
        'PyTango',  # Backward compatibilty
    ]

    requires = [
        'boost_python (>=1.33)',
        'numpy (>=1.1)',
        'six',
    ]

    install_requires = [
        'six',
        'enum34;python_version<"3.4"',
    ]

    setup_requires = []

    if TESTING:
        setup_requires += ['pytest-runner']

    tests_require = [
        'pytest-xdist',
        'gevent',
        'psutil',
    ]

    if PYTHON2:
        tests_require += ['trollius']

    package_data = {
        'tango.databaseds': ['*.xmi', '*.sql', '*.sh', 'DataBaseds'],
    }

    data_files = []

    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Other Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved ::'
        ' GNU Library or Lesser General Public License (LGPL)',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
    ]

    # Note for PyTango developers:
    # Compilation time can be greatly reduced by compiling the file
    # src/precompiled_header.hpp as src/precompiled_header.hpp.gch
    # and then uncommenting this line. Someday maybe this will be
    # automated...
    extra_compile_args = [
        # '-include ext/precompiled_header.hpp',
    ]

    extra_link_args = [
    ]

    if please_debug:
        extra_compile_args += ['-g', '-O0']
        extra_link_args += ['-g', '-O0']

    src_dir = abspath('ext')
    client_dir = src_dir
    server_dir = os.path.join(src_dir, 'server')

    clientfiles = sorted(
        os.path.join(client_dir, fname)
        for fname in os.listdir(client_dir)
        if fname.endswith('.cpp'))

    serverfiles = sorted(
        os.path.join(server_dir, fname)
        for fname in os.listdir(server_dir)
        if fname.endswith('.cpp'))

    cppfiles = clientfiles + serverfiles
    directories['include_dirs'].extend([client_dir, server_dir])

    include_dirs = uniquify(directories['include_dirs'])
    library_dirs = uniquify(directories['library_dirs'])
    libraries = uniquify(directories['libraries'])

    pytango_ext = Extension(
        name='_tango',
        sources=cppfiles,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
        depends=[])

    cmdclass = {
        'build': build,
        'build_ext': build_ext,
        'install_html': install_html,
        'install': install}

    if sphinx:
        cmdclass['build_doc'] = build_doc

    long_description = get_readme()

    opts = dict(
        name='pytango',
        version=Release.version_long,
        description=Release.description,
        long_description=long_description,
        author=author[0],
        author_email=author[1],
        url=Release.url,
        download_url=Release.download_url,
        platforms=Release.platform,
        license=Release.license,
        packages=packages,
        py_modules=py_modules,
        classifiers=classifiers,
        package_data=package_data,
        data_files=data_files,
        provides=provides,
        keywords=Release.keywords,
        requires=requires,
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        ext_package='tango',
        ext_modules=[pytango_ext],
        cmdclass=cmdclass)

    return opts


def main():
    return setup(**setup_args())


if __name__ == "__main__":
    main()
