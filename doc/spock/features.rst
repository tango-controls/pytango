
.. _spock-features:

Features
========

Spock works like a normal python console, but it gives you in addition a nice
set of features from IPython_ like:

    - proper (bash-like) command completion
    - automatic expansion of python variables, functions, types
    - command history (with up/down arrow keys, %hist command)
    - help system ( object? syntax, help(object))
    - persistently store your favorite variables
    - color modes
 
(for a complete list checkout the `IPython tutorial <http://ipython.scipy.org/doc/manual/html/interactive/tutorial.html>`_)

Plus an additional set o Tango_ specific features:

    - automatic import of Tango objects to the console namespace (PyTango module,
      :class:`PyTango.DeviceProxy` (=Device), :class:`PyTango.Database`, :class:`PyTango.Group`
      and :class:`PyTango.AttributeProxy` (=Attribute))
    - device name completion
    - attribute name completion
    - automatic tango object member completion
    - list tango devices, classes, servers
    - customized tango error message
    - tango error introspection
    - switch database
    - refresh database
    - list tango devices, classes
    - store favorite tango objects
    - store favorite tango devices
    - tango color modes

Check the :ref:`spock-highlights` to see how to put these feature to good use :-)

.. _IPython: http://ipython.scipy.org/
.. _Tango: http://www.tango-controls.org/