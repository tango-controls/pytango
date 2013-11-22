.. highlight:: python
   :linenothreshold: 5

.. _getting-started:

Getting started
===============

Quick installation: Linux
-------------------------

PyTango is available on linux as an official debian/ubuntu package::

    $ sudo apt-get install python-pytango

RPM packages are also available for RHEL & CentOS:

.. hlist::
   :columns: 2

   * `RHEL 5/CentOS 5 32bits <ftp://ftp.maxlab.lu.se/pub/maxlab/packages/el5/i386/repoview/index.html>`_
   * `RHEL 5/CentOS 5 64bits <ftp://ftp.maxlab.lu.se/pub/maxlab/packages/el5/x86_64/repoview/index.html>`_
   * `RHEL 6/CentOS 6 32bits <ftp://ftp.maxlab.lu.se/pub/maxlab/packages/el6/i386/repoview/index.html>`_
   * `RHEL 6/CentOS 6 64bits <ftp://ftp.maxlab.lu.se/pub/maxlab/packages/el6/x86_64/repoview/index.html>`_

From PyPi
~~~~~~~~~

You can also install the latest version from `PyPi`_.

First, make sure you have the following packages already installed (all of them
are available from the major official distribution repositories):

* `boost-python`_ (including boost-python-dev)
* `numpy`_ 
* `IPython`_ (optional, highly recommended)

Then install PyTango either from pip::

    $ pip install PyTango

or easy_install::

    $ easy_install -U PyTango

Quick installation: Windows
---------------------------

First, make sure `Python`_ and `numpy`_ are installed.

PyTango team provides a limited set of binary PyTango distributables for
Windows XP/Vista/7/8. The complete list of binaries can be downloaded from
`PyPI`_.

Select the proper windows package, download it and finally execute the 
installion wizard.


Compiling & installing
----------------------

Linux
~~~~~

Since PyTango 7 the build system used to compile PyTango is the standard python 
distutils.

Besides the binaries for the three dependencies mentioned above, you also need 
the development files for the respective libraries.

You can get the latest ``.tar.gz`` from `PyPI`_ or directly
the latest SVN checkout::

    $ svn co http://svn.code.sf.net/p/tango-cs/code/bindings/PyTango/trunk PyTango
    $ cd PyTango
    $ python setup.py build
    $ sudo python setup.py install

This will install PyTango in the system python installation directory and, since
version 8.0.0, it will also install :ref:`itango` as an IPython_ extension.

If whish to install in a different directory, replace the last line with::
    
    $ # private installation to your user (usually ~/.local/lib/python<X>.<Y>/site-packages
    $ python setup.py install --user

    $ # or specific installation directory
    $ python setup.py install --prefix=/home/homer/local

Windows
~~~~~~~

On windows, PyTango must be built using MS VC++.
Since it is rarely needed and the instructions are so complicated, I have
choosen to place the how-to in a separate text file. You can find it in the
source package under :file:`doc/windows_notes.txt`.

Testing your installation
-------------------------

If you have IPython_ installed, the best way to test your PyTango installation
is by starting the new PyTango CLI called :ref:`itango` by typing on the command
line::

    $ itango

then, in ITango type:

.. sourcecode:: itango

    ITango [1]: PyTango.Release.version
    Result [1]: '8.0.2'

(if you are wondering, :ref:`itango` automaticaly does ``import PyTango`` 
for you!)

If you don't have IPython_ installed, to test the installation start a
python console and type:

    >>> import PyTango
    >>> PyTango.Release.version
    '8.0.2'

