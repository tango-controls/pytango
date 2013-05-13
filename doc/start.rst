.. highlight:: python
   :linenothreshold: 5

.. _getting-started:

Getting started
===============

Quick installation
------------------

If you have all :ref:`dependencies <dependencies>` installed on your system,
building and installing / updating PyTango can be as simple as::

    easy_install -U PyTango

.. _dependencies:

If you managed to run this line, the :ref:`quick tour <quick-tour>` can guide
you through the first steps on using PyTango.

Dependencies on other libraries
-------------------------------

.. graphviz::

    digraph dependencies {
        size="6,3";
        PyTango     [shape=box, label="PyTango 8.0"];
        Python      [shape=box, label="Python >=2.6"];
        boostpython [shape=box, label="boost python"];
        Tango       [shape=box, label="Tango >=8.0.5"];
        omniORB     [shape=box, label="omniORB >=4"];
        numpy       [shape=box, label="numpy >=1.1.0"];
        IPython     [shape=box, label="IPython >=0.10"];
        PyTango -> Python;
        PyTango -> Tango;
        PyTango -> numpy [style=dotted, label="mandatory in windows"];
        Tango -> omniORB;
        PyTango -> boostpython
        PyTango -> IPython [style=dotted, label="optional"];
    }   

Don't be scared by the graph. Probably most of the packages are already installed.
The current PyTango version has three major dependencies:

- python (>= 2.6) (http://www.python.org/)
- Tango (>= 8.0.5) (http://www.tango-controls.org/)
- boost python (http://www.boost.org):
    We **strongly** recommend always using boost python >= 1.41
  
plus two optional dependencies (activated by default) on:

- IPython (>=0.10) (http://www.ipython.org/) (necessary for :ref:`itango`)
- numpy (>= 1.1.0) (http://numpy.scipy.org/)

.. note::
    For the provided windows binary, numpy is MANDATORY!

Installing precompiled binaries
-------------------------------

Linux
~~~~~

PyTango is available on linux as an official debian/ubuntu package (python-pytango).

RPM packages are also available for RHEL & CentOS:

    * `RHEL5/CentOS5 5 32bits <ftp://ftp.maxlab.lu.se/pub/maxlab/packages/el5/i386/repoview/index.html>`_
    * `RHEL5/CentOS5 5 64bits <ftp://ftp.maxlab.lu.se/pub/maxlab/packages/el5/x86_64/repoview/index.html>`_
    * `RHEL6/CentOS6 5 32bits <ftp://ftp.maxlab.lu.se/pub/maxlab/packages/el6/i386/repoview/index.html>`_
    * `RHEL6/CentOS6 5 64bits <ftp://ftp.maxlab.lu.se/pub/maxlab/packages/el6/x86_64/repoview/index.html>`_        

.. _pytango-windows-bin:

Windows
~~~~~~~

PyTango team provides a limited set of binary PyTango distributables for
Windows XP/Vista/7. The complete list of binaries can be downloaded from
`PyTango PyPI website <http://pypi.python.org/pypi/PyTango/>`_.

Steps:

* Install `Tango C++ 32 bits <http://ftp.esrf.fr/pub/cs/tango/TangoSetup-8.0.5_win32.exe>`_
* For `Python 2.6 32 bits <http://www.python.org/ftp/python/2.6.6/python-2.6.6.msi>`_
    * `Numpy for python 2.6 <http://pypi.python.org/packages/2.6/n/numpy/numpy-1.6.2.win32-py2.6.exe>`_
    * `PyTango 8 for python 2.6 <http://pypi.python.org/packages/2.6/P/PyTango/PyTango-8.0.2.win32-py2.6.msi>`_
* For `Python 2.7 32 bits <http://www.python.org/ftp/python/2.7.3/python-2.7.3.msi>`_
    * `Numpy for python 2.7 <http://pypi.python.org/packages/2.7/n/numpy/numpy-1.6.2.win32-py2.7.exe>`_
    * `PyTango 8 for python 2.7 <http://pypi.python.org/packages/2.7/P/PyTango/PyTango-8.0.2.win32-py2.7.msi>`_
* For `Python 3.1 32 bits <http://www.python.org/ftp/python/3.1.4/python-3.1.4.msi>`_
    * `Numpy for python 3.1 <http://pypi.python.org/packages/3.1/n/numpy/numpy-1.6.2.win32-py3.1.exe>`_
    * `PyTango 8 for python 3.1 <http://pypi.python.org/packages/3.1/P/PyTango/PyTango-8.0.2.win32-py3.1.msi>`_
* For `Python 3.2 32 bits <http://www.python.org/ftp/python/3.2.3/python-3.2.3.msi>`_
    * `Numpy for python 3.2 <http://pypi.python.org/packages/3.2/n/numpy/numpy-1.6.2.win32-py3.2.exe>`_
    * `PyTango 8 for python 3.2 <http://pypi.python.org/packages/3.2/P/PyTango/PyTango-8.0.2.win32-py3.2.msi>`_

..
.. _PyTango-8.0.2.win32-py2.6.msi: http://pypi.python.org/packages/2.6/P/PyTango/PyTango-8.0.2.win32-py2.6.msi
.. _PyTango-8.0.2.win32-py2.6.exe: http://pypi.python.org/packages/2.6/P/PyTango/PyTango-8.0.2.win32-py2.6.exe
.. _PyTango-8.0.2.win32-py2.7.msi: http://pypi.python.org/packages/2.7/P/PyTango/PyTango-8.0.2.win32-py2.7.msi
.. _PyTango-8.0.2.win32-py2.7.exe: http://pypi.python.org/packages/2.7/P/PyTango/PyTango-8.0.2.win32-py2.7.exe
.. _PyTango-8.0.2.win32-py3.1.msi: http://pypi.python.org/packages/3.1/P/PyTango/PyTango-8.0.2.win32-py3.1.msi
.. _PyTango-8.0.2.win32-py3.1.exe: http://pypi.python.org/packages/3.1/P/PyTango/PyTango-8.0.2.win32-py3.1.exe
.. _PyTango-8.0.2.win32-py3.2.msi: http://pypi.python.org/packages/3.2/P/PyTango/PyTango-8.0.2.win32-py3.2.msi
.. _PyTango-8.0.2.win32-py3.2.exe: http://pypi.python.org/packages/3.2/P/PyTango/PyTango-8.0.2.win32-py3.2.exe


Compiling & installing
----------------------

Linux
~~~~~

Since PyTango 7 the build system used to compile PyTango is the standard python 
distutils.

Besides the binaries for the three dependencies mentioned above, you also need 
the development files for the respective libraries.

boost python dependency
#######################

PyTango has a dependency on the boost python library (>= 1.33). This means that
the shared library file **libboost-python.so** must be accessible to the 
compilation command.

.. note::

    If you use python >= 2.6.3 you MUST install boost python >= 1.41

Most linux distributions today provide a boost python package.

Furthermore, in order to be able to build PyTango, you also need the include
headers of boost python. They are normaly provided by a package called
boost_python-dev.

If, for some reason, you need to compile and install boost python, here is a
quick recipie:

    #. Download latest boost tar.gz file and extract it
    #. Download latest bjam (most linux distributions have a bjam package. If
       not, sourceforge provides a binary for many platforms)
    #. build and/or install:
    
       #. Simple build: in the root directory where you extracted boost type:
       
          ``bjam --with-python toolset=gcc variant=release threading=multi link=shared``
          
          this will produce in :file:`bin.v2/libs/python/build/gcc-<gcc_ver>/release/threading-multi` a file called :file:`libboost_python-gcc<gcc_ver>-mt-<boost_ver>.so.<boost_python_ver>`
          
       #. Install (you may need administrator permissions to do so):
       
          ``bjam --with-python toolset=gcc variant=release threading=multi link=shared install``
          
       #. Install in a different directory (<install_dir>):
       
          ``bjam --with-python toolset=gcc variant=release threading=multi link=shared install --prefix=<install_dir>``


boost, omniORB and TangoC++ configuration
#########################################

The second step is to make sure the three/four libraries (omniORB, tango, 
boost python and/or numpy) are accessible to the compilation command. So, for 
example, if you installed:

    ``boost python under /home/homer/local``
    
    ``omniORB under /home/homer/local1``
    
    ``tango under /home/homer/local2``
    
    ``numpy under /usr/lib/python2.6/site-packages/numpy``
    
you must export the three environment variables::

    export BOOST_ROOT=/home/homer/local
    export OMNI_ROOT=/home/homer/local1
    export TANGO_ROOT=/home/homer/local2
    
    # in openSUSE 11.1 this is the default base location for the include files
    export NUMPY_ROOT=/usr/lib/python2.6/site-packages/numpy/core

(for numpy this is the default base location for the include files. This is
distribution dependent. For example, ubuntu places a numpy directory under /usr/include,
so exporting NUMPY_ROOT is not necessary for this distribution)

For the libraries that were installed in the default system directory (/usr or /usr/local)
the above lines are not necessary.

.. _build-install:

build & install
###############

Finally::

    python setup.py build
    sudo python setup.py install
    
This will install PyTango in the system python installation directory and, since
version 8.0.0, it will also install :ref:`itango` as an IPython_ extension.
    
Or if you whish to install in a different directory::
    
    python setup.py build
    python setup.py install --prefix=/home/homer/local --ipython-local

(This will try to install :ref:`itango` as an IPython profile to the local
user, since probably there is no permission to write into the IPython_ extension
directory)

Or if you wish to use your own private python distribution::

    /home/homer/bin/python setup.py build
    /home/homer/bin/python setup.py install

For the last case above don't forget that boost python should have also been 
previously compiled with this private python distribution.

test
####

If you have IPython_ installed, the best way to test your PyTango installation
is by starting the new PyTango CLI called :ref:`itango` by typing on the command
line::

    $ itango

then, in ITango type:

.. sourcecode:: itango

    ITango [1]: PyTango.Release.version
    Result [1]: '8.0.2'

(if you are wondering, :ref:`itango` automaticaly does ``import PyTango`` for you!)

If you don't have IPython_ installed, to test the installation start a python console
and type:

    >>> import PyTango
    >>> PyTango.Release.version
    '8.0.2'

Windows
~~~~~~~

On windows, PyTango must be built using MS VC++.

... which is good fun specially if you only have express edition and multiple python versions to build with! 

Pre-requisites
##############

**Python**

Python must be installed in the following directory structure:

    <PythonBaseDir>\\<PlatformName>\\<PythonVersion>
    
Where:

    - <PythonBaseDir> is a directory that can be choosen by the you (ex: C:\\Python)
    - <PlatformName> **must** be either *win32* or *x64*
    - <PythonVersion> **must** be *26*, *27*, *31*, *32*, *33*

Example: Assuming you choose *C:\\Python* as PythonBaseDir, if you want to build
PyTango for python 2.7 on 64 bits you must install python in:

    c:\\python\\x64\\27

**Visual C++**

Python recommends compiling any python libraries using the same compiler
version. So, depending on the python version(s) you want PyTango to be
build, you need VC++ 9.0 (2008) or/and VC++ 10.0 (2010).

Here is the table of compilers and corresponding Visual C++ version used by CPython:

+----------------------+--------------+
| Visual C++ version   | Compiler     |
+======================+==============+
| Visual C++ 4.x       | MSC_VER=1000 |
+----------------------+--------------+
| Visual C++ 5         | MSC_VER=1100 |
+----------------------+--------------+
| Visual C++ 6         | MSC_VER=1200 |
+----------------------+--------------+
| Visual C++ .NET      | MSC_VER=1300 |
+----------------------+--------------+
| Visual C++ .NET 2003 | MSC_VER=1310 |
+----------------------+--------------+
| Visual C++ 2005      | MSC_VER=1400 |
+----------------------+--------------+
| Visual C++ 2008      | MSC_VER=1500 |
+----------------------+--------------+
| Visual C++ 2010      | MSC_VER=1600 |
+----------------------+--------------+
| Visual C++ 2011      | MSC_VER=1700 |
+----------------------+--------------+

+-------------------------+-----------------------------+
|        Python           |              VC++           |
| version  | architecture |                             |
+==========+==============+=============================+
|  2.6.6   |   32 bits    | MSC 1500 (Visual C++ 2008)  |
+----------+--------------+-----------------------------+
|  2.6.6   |   64 bits    | MSC 1500 (Visual C++ 2008)  |
+----------+--------------+-----------------------------+
|  2.7.3   |   32 bits    | MSC 1500 (Visual C++ 2008)  |
+----------+--------------+-----------------------------+
|  2.7.3   |   64 bits    | MSC 1500 (Visual C++ 2008)  |
+----------+--------------+-----------------------------+
|  3.1.4   |   32 bits    | MSC 1500 (Visual C++ 2008)  |
+----------+--------------+-----------------------------+
|  3.1.4   |   64 bits    | MSC 1500 (Visual C++ 2008)  |
+----------+--------------+-----------------------------+
|  3.2.3   |   32 bits    | MSC 1500 (Visual C++ 2008)  |
+----------+--------------+-----------------------------+
|  3.2.3   |   64 bits    | MSC 1500 (Visual C++ 2008)  |
+----------+--------------+-----------------------------+
|  3.3.0   |   32 bits    | MSC 1600 (Visual C++ 2010)  |
+----------+--------------+-----------------------------+
|  3.3.0   |   64 bits    | MSC 1600 (Visual C++ 2010)  |
+----------+--------------+-----------------------------+

**Visual C++ 9.0 (2008) express quick install guide**

1. Download and install VC++ 9.0 (2008) Express Edition
2. If you need to compile in 64 bits platform:
    2.1. Download and install Windows 7 SDK for .NET Framework 3.5 SP1 [a.k.a. Windows SDK 7.0]
        (**not** Windows SDK 7.1!)
    
    2.2. The vcvarsall.bat in VC++ 2008 Express looks for the x86_amd64 vcvarsx86_amd64.bat
         in all the wrong places. The easiest way to work around that is to navigate to the
         VC\\bin directory of your VC++ 2008 installation (in my case
         C:\\Program Files (x86)\\Microsoft Visual Studio 9.0\\VC\\bin). Copy vcvarsx86_amd64.bat,
         and paste into the VC\\bin\\x86_amd64 subdirectory.

**Visual C++ 10.0 (2010) express quick install guide**

1. Download and install VC++ 10.0 (2010) Express Edition
2. If you need to compile in 64 bits platform follow the instructions
   **in the order they appear** (not doing so may lead to
   `KB2519277 problem <http://support.microsoft.com/kb/2519277>`_ ):
    
    2.1. Visual Studio 2010 SP1
    
    2.2. Download and install Windows 7 SDK for .NET Framework 4.0 [a.k.a. Windows SDK 7.1]
    
    2.3. VC++ 2010 doesn't come with vcvarsx86_amd64.bat. But in this case, since the 
         environment setting tool is different than in VC++ 2008, all you have to do is 
         create a file called vcvarsx86_amd64.bat in VC\\bin\\x86_amd64 directory of your 
         VC++ 2010 installation (in my case 
         C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\bin\\x86_amd64) with the
         following content:
         
             @CALL "C:\\Program Files\\Microsoft SDKs\\Windows\\v7.1\\Bin\\SetEnv.Cmd" /Release /x64
         
         (adapt to your Windows SDK installation directory)
    
**Boost python**

Boost python DLL, lib and header files must be installed for the specific platform(s)
and python version(s) you which to build PyTango on. The directory structure for
the boost headers: 

    <BoostDir>\\include

The directory structure for the boost libraries:

    <BoostDir>\\multi\\release\\<MSVCVersion>\\<PlatformName>\\<Link>\\<RuntimeLink>\\<PythonVersion>

Where:
    
    - <BoostDir> the boost base directory (ex: C:\Boost-1.53.0)
    - <MSVCVersion> may be either *msvc-9.0* or *msvc-10.0*
    - <PlatformName> **must** be either *win32* or *x64*
    - <Link> **must** be either *static* or *shared*
    - <RuntimeLink> **must** be either *static* or *shared*
      (if Link==static, RuntimeLink can only be *static*)
    - <PythonVersion> **must** be *26*, *27*, *31*, *32*, *33*

**Boost python multi platform compilation quick build guide**
    
    - Download boost source code from http://wwww.boost.org
    - Extract boost to a directory (ex: c:\\workspace\\boost-1.53.0)
    - Download and place `boost_python_install.py <>` in your boost extract directory (ex: c:\\workspace\\boost-1.53.0\\boost_python_install.py)
      (adapt python versions you which to build)
    - Place the user-config.jam file in %HOMEPATH%%HOMEDIR% (adapt paths and python versions to your system)
    - Open a console
    - Switch to the boost directory 
    - Execute this script using python (ex: C:\\Python\\win32\\26\\python.exe boost_python_install.py)

**Tango**



.. _IPython: http://ipython.scipy.org/

