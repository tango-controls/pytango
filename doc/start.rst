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
        PyTango     [shape=box, label="PyTango 7.2"];
        Python      [shape=box, label="Python >=2.4"];
        boostpython [shape=box, label="boost python"];
        boostp1     [shape=box, label="boost >=1.33"];
        boostp2     [shape=box, label="boost >=1.41"];
        Tango       [shape=box, label="Tango >=7.2"];
        omniORB     [shape=box, label="omniORB >=4"];
        numpy       [shape=box, label="numpy >=1.1.0"];
        IPython     [shape=box, label="IPython >=0.10"];
        PyTango -> Python;
        PyTango -> Tango;
        PyTango -> numpy [style=dotted, label="mandatory in windows"];
        Tango -> omniORB;
        PyTango -> boostpython
        boostpython -> boostp1 [label="if python <2.6.3"];
        boostpython -> boostp2 [label="if python >=2.6.3"];
        PyTango -> IPython [style=dotted, label="optional"];
    }   

Don't be scared by the graph. Probably most of the packages are already installed.
The current PyTango version has four major dependencies:

- python (>= 2.4) (http://www.python.org/)
- omniORB (http://omniorb.sourceforge.net/)
- Tango (>= 7.2.0) (http://www.tango-controls.org/)
- boost python (http://www.boost.org):
    if python >= 2.6.3 then: boost-python >= 1.41
    else: boost-python >= 1.33
    We **strongly** recommend always using >= 1.41
  
plus two optional dependencies (activated by default) on:

- IPyhon (>=0.10) (http://ipython.scipy.org/) (necessary for :ref:`itango`)
- numpy (>= 1.1.0) (http://numpy.scipy.org/)

.. note::
    For the provided windows binary, numpy is MANDATORY!

Installing precompiled binaries
-------------------------------

Linux
~~~~~

The PyTango team does **not** provide a precompiled binary for Linux since this
would mean having to provide 12 different binaries: one for each major python
version (2.4, 2.5, 2.6, 2.7, 3.0 and 3.1) times 2 for both 32bits and 64bits.

Tango contributers have written packages for *at least* ubuntu and debian linux
distributions. Check the **Ubuntu GNU/Linux binary distribution** chapter under
`Tango downloads <http://www.tango-controls.org/download>`_ for more details.

.. _pytango-windows-bin:

Windows
~~~~~~~

PyTango team provides a limited set of binary PyTango distributables for
Windows XP/Vista/7. The complete list of binaries can be downloaded from
`PyTango PyPI website <http://pypi.python.org/pypi/PyTango/>`_.

.. _PyTango-7.2.2.win32-py2.6.msi: http://pypi.python.org/packages/2.6/P/PyTango/PyTango-7.2.2.win32-py2.6.msi
.. _PyTango-7.2.2.win32-py2.6.exe: http://pypi.python.org/packages/2.6/P/PyTango/PyTango-7.2.2.win32-py2.6.exe
.. _PyTango-7.2.2.win32-py2.7.msi: http://pypi.python.org/packages/2.7/P/PyTango/PyTango-7.2.2.win32-py2.7.msi
.. _PyTango-7.2.2.win32-py2.7.exe: http://pypi.python.org/packages/2.7/P/PyTango/PyTango-7.2.2.win32-py2.7.exe

+----------------------------------+--------------------------------------------------+----------------------------------------------+
| version                          | Dependencies                                     | Compilation env.                             |
+==================================+==================================================+==============================================+
| `PyTango-7.2.2.win32-py2.6.msi`_ | - Tango C++ >= 7.2.6 and < 8.0                   | - Tango 7.2.6 windows distribution           |
| `PyTango-7.2.2.win32-py2.6.exe`_ | - Python 2.6.x (where x >= 0)                    | - Python 2.6.6                               |
|                                  | - numpy 1.x (where x >= 1. Recommended x >= 5)   | - Numpy 1.5                                  |
|                                  |                                                  | - boost-python 1.41 mutithreaded dll         |
|                                  |                                                  | - Visual Studio 8.0 (2005)                   |
|                                  |                                                  | - Windows XP Pro 2002 SP3                    |
|                                  |                                                  | - PC: Intel Xeon E5440 @ 2.83GHz 1GB RAM     |
+----------------------------------+--------------------------------------------------+----------------------------------------------+
| `PyTango-7.2.2.win32-py2.7.msi`_ | - Tango C++ >= 7.2.6 and < 8.0                   | - Tango 7.2.6 windows distribution           |
| `PyTango-7.2.2.win32-py2.7.exe`_ | - Python 2.7.x (where x >= 0)                    | - Python 2.7.2                               |
|                                  | - numpy 1.x (where x >= 1. Recommended x >= 5)   | - Numpy 1.5                                  |
|                                  |                                                  | - boost-python 1.47 mutithreaded dll         |
|                                  |                                                  | - Visual Studio 8.0 (2005)                   |
|                                  |                                                  | - Windows XP Pro 2002 SP3                    |
|                                  |                                                  | - PC: Intel Xeon E5440 @ 2.83GHz 1GB RAM     |
+----------------------------------+--------------------------------------------------+----------------------------------------------+

Until version 7.2.2 (due to internal incompatibilities between tango C++ API
and PyTango), PyTango had to be shipped with an internal copy of tango and
omniORB DLLs. Since version 7.2.2 ( and tango C++ version 7.2.6) this is no
longer necessary. In other words, until 7.2.2 you could install and use PyTango
without having tango c++ installed. Starting from 7.2.2 you **must** have tango
C++ installed **and** the environment variable :envvar:`PATH` **must** include
the directory where the tango C++ DLLs are installed (usually
:file:`C:\\Program Files{ (x86)}\\tango\\win32_vc8\\win32_dll`).

Regarding boost-python, since VS hard links with the boost-python DLL file of
the machine where PyTango binary was originally compiled, PyTango ships with
it's own internal copy of the boost-python DLL.
Maybe in the future PyTango will link with the static version of boost-python
but for now we get to many errors at compile time so we are skipping this for
now. Anyway, it's just an internal developers detail. For you just means 250kb more
of memory usage in windows.

The binary was compiled with numpy dependency therefore you need to have *numpy*
installed in order to use PyTango.

If PyTango reports *DLL load failed* probably you are missing Visual Studio 2005
redistributable package. You can download and install it from
`Microsoft Visual C++ 2005 Redistributable Package (x86) <http://www.microsoft.com/download/en/details.aspx?id=3387>`_

+------------+-----------------------------------------------------------------+
| version    | Includes the following DLLs                                     |
+============+=================================================================+
| 7.2.2      | - boost python 1.41 (VC++8, multi-threaded)                     |
+------------+-----------------------------------------------------------------+
| 7.1.0      | - tango 7.1.1 (VC++ 8)                                          |
|            | - omniORB 4.1.4                                                 |
|            | - boost python 1.41 (VC++8, multi-threaded)                     |
+------------+-----------------------------------------------------------------+
| 7.1.0 rc1  | - tango 7.1.1 (VC++ 8)                                          |
|            | - omniORB 4.1.4                                                 |
|            | - boost python 1.41 beta 1 (VC++8, multi-threaded)              |
|            |   this version was used because it is the first version that    |
|            |   fixes a bug that prevents PyTango from being used with        | 
|            |   python >= 2.6.3                                               |
+------------+-----------------------------------------------------------------+


Compiling & installing
----------------------

Linux
~~~~~

Since PyTango 7 the build system used to compile PyTango is the standard python 
distutils.

Besides the binaries for the four dependencies mentioned above, you also need 
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
version 7.1.2, it will also install :ref:`itango` as an IPython_ extension.
    
Or if you whish to install in a different directory::
    
    python setup.py build
    python setup.py install --prefix=/home/homer/local --ipython-local

(This will try to install :ref`itango` as an IPython profile to the local
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
is by starting the new PyTango CLI called :ref`itango` by typing on the command
line:

    #. IPython <= 0.10::

        $ ipython -p tango

    #.IPython > 0.10::

        $ ipython --profile=tango


then, in ITango type:

.. sourcecode:: itango

    ITango <homer:10000> [1]: PyTango.Release.version
                  Result [1]: '7.1.2'

(if you are wondering, :ref`itango` automaticaly does ``import PyTango`` for you!)

If you don't have IPython_ installed, to test the installation start a python console
and type:

    >>> import PyTango
    >>> print PyTango.Release.version
    7.1.2

.. toctree::
    :hidden:

    Quick tour <quicktour>
    Quick tour (original) <quicktour_old>
    
.. _IPython: http://ipython.scipy.org/