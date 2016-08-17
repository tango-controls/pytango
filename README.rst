PyTango
=======

Python binding for Tango_, a library dedicated to distributed control systems.


Description
-----------

PyTango_ exposes the complete `Tango C++ API`_ through the ``tango`` python module.
It also adds a bit of abstraction by taking advantage of the Python capabilites:

- ``tango.client`` provides a client access to device servers and databases.
- ``tango.server`` provides base classes to declare and run device servers.


Requirements
------------

PyTango_ is compatible with python 2 and python 3.

General dependencies:

-  libtango_ >= 9.2
-  `Boost.Python`_ >= 1.33

Python dependencies:

-  numpy_ >= 1.1
-  six_

Build dependencies:

- setuptools_
- sphinx_

Optional dependencies:

- futures_
- gevent_

.. note:: As a general rule, libtango_ and pytango_ should share the same major
	  and minor version (for a version ``X.Y.Z``, ``X`` and ``Y`` should
	  match)


Install
-------

PyTango_ is available on PyPI_ as ``pytango``::

    $ pip install pytango

Alternatively, PyTango_ can be built and installed from the
`sources`_::

    $ python setup.py install

In both cases, the installation takes a few minutes since the ``_tango`` boost
extension has to compile.


Usage
-----

To test the installation, import ``tango`` and check ``tango.__version__``::

    >>> import tango
    >>> tango.__version__
    '9.2.0'

For an interactive use, consider using ITango_, a tango IPython_ profile.


Documentation
-------------

Check out the documentation_ for more informations.



Support and contribution
------------------------

You can get support from the `Tango forums`_, for both Tango_ and PyTango_ questions.

All contributions,  `PR and bug reports`_ are welcome!


.. _Tango: http://tango-controls.org/
.. _Tango C++ API: http://esrf.eu/computing/cs/tango/tango_doc/kernel_doc/cpp_doc/index.html
.. _PyTango: http://github.com/tango-cs/PyTango
.. _PyPI: http://pypi.python.org/pypi/pytango

.. _libtango: http://tango-controls.org/downloads/source/
.. _Boost.Python: http://boost.org/doc/libs/1_61_0/libs/python/doc/html/index.html
.. _numpy: http://pypi.python.org/pypi/numpy
.. _six: http://pypi.python.org/pypi/six
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _sphinx: http://pypi.python.org/pypi/sphinx
.. _futures: http://pypi.python.org/pypi/futures
.. _gevent: http://pypi.python.org/pypi/gevents

.. _ITango: http://pypi.python.org/pypi/itango
.. _IPython: http://ipython.org

.. _documentation: http://pytango.readthedocs.io/en/latest
.. _Tango forums: http://tango-controls.org/community/forum
.. _PR and bug reports: PyTango_
.. _sources: PyTango_
