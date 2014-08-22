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
import imp
import sys
import struct
import platform
import subprocess

from distutils.core import setup, Extension
from distutils.cmd import Command
from distutils.command.build import build as dftbuild
from distutils.command.build_ext import build_ext as dftbuild_ext
from distutils.command.install import install as dftinstall
from distutils.unixccompiler import UnixCCompiler
from distutils.version import StrictVersion as V

try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda : False
    from sphinx.setup_command import BuildDoc
except ImportError:
    sphinx = None

try:
    import IPython
    _IPY_ROOT = os.path.dirname(os.path.abspath(IPython.__file__))
    if V(IPython.__version__) > V('0.10'):
        import IPython.utils.path
        get_ipython_dir = IPython.utils.path.get_ipython_dir
    else:
        import IPython.genutils
        get_ipython_dir = IPython.genutils.get_ipython_dir
    _IPY_LOCAL = str(get_ipython_dir())
except:
    IPython = None

try:
    import numpy
except ImportError:
    numpy = None

is64 = 8 * struct.calcsize("P") == 64


def pkg_config(*packages, **config):
    config_map = {
        "-I": "include_dirs",
        "-L": "library_dirs",
        "-l": "libraries",
    }
    cmd = ["pkg-config", "--cflags-only-I",
           "--libs-only-L", " ".join(packages)]
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
    name = "release"
    release_dir = abspath('src', 'boost', 'python')
    data = imp.find_module(name, [release_dir])
    release = imp.load_module(name, *data)
    return release.Release


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


def get_script_files():

    FILTER_OUT = []  # "winpostinstall.py",

    if os.name != "nt":
        FILTER_OUT.append("pytango_winpostinstall.py")

    scripts_dir = abspath("scripts")
    scripts = []
    items = os.listdir(scripts_dir)
    for item in items:
        # avoid hidden files
        if item.startswith("."):
            continue
        abs_item = os.path.join(scripts_dir, item)
        # avoid non files
        if not os.path.isfile(abs_item):
            continue
        if item.endswith('c') and item[:-1] in items:
            continue
        # avoid any core dump... of course there isn't any :-) but just in case
        if item.startswith('core'):
            continue
        if item in FILTER_OUT:
            continue
        scripts.append('scripts/' + item)
    return scripts


def add_lib(name, dirs, sys_libs, env_name=None, lib_name=None, inc_suffix=None):
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
        if is64:
            lib64_dir = os.path.join(ENV, 'lib64')
            if os.path.isdir(lib64_dir):
                lib_dirs.insert(0, lib64_dir)
        dirs['library_dirs'].extend(lib_dirs)
        
        if lib_name.startswith('lib'):
            lib_name = lib_name[3:]
        dirs['libraries'].append(lib_name)


class build(dftbuild):

    user_options = dftbuild.user_options + \
        [('without-ipython', None, "Tango IPython extension"),
         ('strip-lib', None, "strips the shared library of debugging symbols (Unix like systems only)"),
         ('no-doc', None, "do not build documentation") ]

    boolean_options = dftbuild.boolean_options + ['without-ipython', 'strip-lib', 'no-doc']

    def initialize_options (self):
        dftbuild.initialize_options(self)
        self.without_ipython = None
        self.strip_lib = None
        self.no_doc = None

    def finalize_options(self):
        dftbuild.finalize_options(self)

    def run(self):
        if numpy is None:
            self.warn('NOT using numpy: it is not available')
        elif get_c_numpy() is None:
            self.warn("NOT using numpy: numpy available but C source is not")

        if IPython and not self.without_ipython:
            if V(IPython.__version__) > V('0.10'):
                self.distribution.py_modules.append('IPython.config.profile.tango.ipython_config')
            else:
                self.distribution.py_modules.append('IPython.Extensions.ipy_profile_tango')

        dftbuild.run(self)

        if self.strip_lib:
            if 'posix' in os.name:
                has_objcopy = os.system("type objcopy") == 0
                if has_objcopy:
                    d = abspath(self.build_lib, "PyTango")
                    orig_dir = os.path.abspath(os.curdir)
                    so = "_PyTango.so"
                    dbg = so + ".dbg"
                    try:
                        os.chdir(d)
                        is_stripped = os.system('file %s | grep -q "not stripped" || exit 1' % so) != 0
                        if not is_stripped:
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
            print("Documentation will not be generated: sphinx version (%s) too low. Needs 0.6.6" % sphinx.__version__)
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
            #self.compiler.compiler_so = " ".join(compiler_pars)

            # mimic tango check to activate C++0x extension
            compiler = self.compiler.compiler
            proc = subprocess.Popen(compiler + ["-dumpversion"], stdout=subprocess.PIPE)
            pipe = proc.stdout
            proc.wait()
            gcc_ver = pipe.readlines()[0].decode().strip()
            if V(gcc_ver) >= V("4.3.3"):
                self.use_cpp_0x = True
        dftbuild_ext.build_extensions(self)

    def build_extension(self, ext):
        if self.use_cpp_0x:
            ext.extra_compile_args += ['-std=c++0x']
            ext.define_macros += [ ('PYTANGO_HAS_UNIQUE_PTR', '1') ]
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

    user_options = [
        ('install-dir=', 'd', 'base directory for installing HTML documentation files')]

    def initialize_options(self):
        self.install_dir = None

    def finalize_options(self):
        self.set_undefined_options('install',
                                   ('install_html', 'install_dir'))

    def run(self):
        build_doc_cmd = self.get_finalized_command('build_doc')
        src_html_dir = abspath(build_doc_cmd.build_dir, 'html')
        self.copy_tree(src_html_dir, self.install_dir)


class install(dftinstall):

    user_options = dftinstall.user_options + \
        [('install-html=', None, "installation directory for HTML documentation"), ]

    def initialize_options(self):
        dftinstall.initialize_options(self)
        self.install_html = None

    def finalize_options(self):
        dftinstall.finalize_options(self)
        # We do a hack here. We cannot trust the 'install_base' value because it
        # is not always the final target. For example, in unix, the install_base
        # is '/usr' and all other install_* are directly relative to it. However,
        # in unix-local (like ubuntu) install_base is still '/usr' but, for
        # example, install_data, is '$install_base/local' which breaks everything.
        #
        # The hack consists in using install_data instead of install_base since
        # install_data seems to be, in practice, the proper install_base on all
        # different systems.
        if self.install_html is None:
            self.install_html = os.path.join(self.install_data,
                                             'share', 'doc', 'PyTango', 'html')

    def has_html(self):
        return sphinx is not None

    sub_commands = list(dftinstall.sub_commands)
    sub_commands.append(('install_html', has_html))


def setup_args():
    macros = []

    directories = {
        'include_dirs': [],
        'library_dirs': [],
        'libraries':    ['tango'],
    }
    sys_libs = []

    add_lib('omni', directories, sys_libs, lib_name='omniORB4')
    add_lib('zmq', directories, sys_libs, lib_name='libzmq')
    add_lib('tango', directories, sys_libs, inc_suffix='tango')

    # special boost-python configuration

    BOOST_ROOT = os.environ.get('BOOST_ROOT')
    boost_library_name = 'boost_python'
    if BOOST_ROOT is None:
        if 'linux' in sys.platform:
            dist_name = platform.linux_distribution()[0].lower()
            debian_based = 'debian' in dist_name or 'ubuntu' in dist_name
            if debian_based:
                # when building with multiple version of python on debian we need
                # to link against boost_python-py25/-py26 etc...
                pyver = "-py"
                pyver += "".join(map(str, platform.python_version_tuple()[:2]))
                boost_library_name += pyver
    else:
        inc_dir = os.path.join(BOOST_ROOT, 'include')
        lib_dirs = [os.path.join(BOOST_ROOT, 'lib')]
        if is64:
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

    if not has_numpy():
        macros.append(('DISABLE_PYTANGO_NUMPY', None))
    else:
        macros.append(('PYTANGO_NUMPY_VERSION', '"' + str(numpy.__version__) + '"'))

    if 'posix' in os.name:
        directories = pkg_config(*sys_libs, **directories)

    Release = get_release_info()

    author = Release.authors['Coutinho']

    please_debug = False

    packages = [
        'PyTango',
        'PyTango.ipython',
        'PyTango.ipython.ipython_00_10',
        'PyTango.ipython.ipython_00_11',
        'PyTango.ipython.ipython_10_00',
    ]

    py_modules = []

    provides = [
        'PyTango',
    ]

    requires = [
        'boost_python (>=1.33)',
        'numpy (>=1.1)'
    ]

    package_data = {
        'PyTango' : [],
    }

    scripts = get_script_files()

    data_files = []
    if os.name == 'nt':
        data_files.append(('scripts', ['doc/_static/itango.ico']))

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
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
    ]

    # Note for PyTango developers:
    # Compilation time can be greatly reduced by compiling the file
    # src/precompiled_header.hpp as src/precompiled_header.hpp.gch
    # and then uncommenting this line. Someday maybe this will be
    # automated...
    extra_compile_args = [
#        '-includesrc/precompiled_header.hpp',
    ]

    extra_link_args = [
        '-Wl,-h',
        '-Wl,--strip-all',
    ]

    if please_debug:
        extra_compile_args += ['-g', '-O0']
        extra_link_args += ['-g' , '-O0']

    src_dir = abspath('src', 'boost', 'cpp')
    client_dir = src_dir
    server_dir = os.path.join(src_dir, 'server')
    _clientfiles = [ os.path.join(client_dir, fname)
                        for fname in os.listdir(client_dir)
                            if fname.endswith('.cpp') ]
    _clientfiles.sort()
    _serverfiles = [ os.path.join(server_dir, fname)
                         for fname in os.listdir(server_dir)
                             if fname.endswith('.cpp') ]
    _serverfiles.sort()
    _cppfiles = _clientfiles + _serverfiles

    include_dirs = uniquify(directories['include_dirs'] + [client_dir, server_dir])
    library_dirs = uniquify(directories['library_dirs'])
    libraries = uniquify(directories['libraries'])

    _pytango = Extension(
        name='_PyTango',
        sources=_cppfiles,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
        depends=[])

    cmdclass = {'build'        : build,
                'build_ext'    : build_ext,
                'install_html' : install_html,
                'install'      : install }

    if sphinx:
        cmdclass['build_doc'] = build_doc

    opts = dict(
        name='PyTango',
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
        package_dir={ 'PyTango' : os.path.join('src', 'boost', 'python') },
        py_modules=py_modules,
        classifiers=classifiers,
        package_data=package_data,
        data_files=data_files,
        scripts=scripts,
        provides=provides,
        keywords=Release.keywords,
        requires=requires,
        ext_package='PyTango',
        ext_modules=[_pytango],
        cmdclass=cmdclass)

    return opts

def main():
    return setup(**setup_args())

if __name__ == "__main__":
    main()
