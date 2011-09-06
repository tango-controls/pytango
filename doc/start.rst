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

- IPyhon (>=0.10) (http://ipython.scipy.org/) (necessary for :ref:`spock`)
- numpy (>= 1.1.0) (http://numpy.scipy.org/)

.. note::
    For the provided windows binary, numpy is MANDATORY!

Installing precompiled binaries
-------------------------------

The latest binaries for PyTango can be found at: http://www.tango-controls.org/download under
the tango bindings section.

Linux
~~~~~

The PyTango team does **not** provide a precompiled binary for Linux since this 
would mean having to provide 12 different binaries: one for each major python 
version (2.4, 2.5, 2.6, 2.7, 3.0 and 3.1) times 2 for both 32bits and 64bits.

Windows
~~~~~~~

PyTango team provides a binary PyTango distributable for Windows XP/Vista 32bits 
**for usage with python 2.6**.

The binary **comes with its's own boost-python, omniORB and Tango DLLs**

+------------+-----------------------------------------------------------------+
| version    | Includes the following DLLs                                     |
+============+=================================================================+
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

The binary was compiled with numpy dependency therefore you need to have *numpy*
installed in order to use PyTango.

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

Furthermore, in order to be able to build PyTango, you also need the include headers of
boost python. They are normaly provided by a package called boost_python-dev.

If, for some reason, you need to compile and install boost python, here is a 
quick recipie:

    #. Download latest boost tar.gz file and extract it
    #. Download latest bjam (most linux distributions have a bjam package. If not, 
       sourceforge provides a binary for many platforms)
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
version 7.1.2, it will also install :ref:`spock` as an IPython_ extension.
    
Or if you whish to install in a different directory::
    
    python setup.py build
    python setup.py install --prefix=/home/homer/local --ipython-local

(This will try to install :ref:`spock` as an IPython profile to the local
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
is by starting the new PyTango CLI called :ref:`spock` by typing on the command
line::

    $ ipython -p spock

then, in spock type:

.. sourcecode:: spock

    Spock <homer:10000> [1]: PyTango.Release.version
                 Result [1]: '7.1.2'

(if you are wondering, :ref:`spock` automaticaly does ``import PyTango`` for you!)

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