.. highlight:: python
   :linenothreshold: 4

.. _spock:

Spock
=====

Spock is a PyTango CLI based on IPython_. It is designed to be used as an
IPython extension or a profile [#ExtProfile]_.

Spock is available since PyTango 7.1.2

.. note::
    'spock' used to be the name given to the CLI dedicated to Sardana. Now spock
    became a generic Tango CLI and the connection to Sardana is provided as an
    extension to spock. This sardana extension is NOT supplied by PyTango but is 
    available from the Sardana package.

You can start spock by typing on the command line::
    
    $ ipython -p spock

and you should get something like this:

.. image:: spock00.png

.. toctree::
    :maxdepth: 1

    features
    highlights

--------------------------------------------------------------------------------

.. [#ExtProfile] The difference between extension and profile is that an
   extension is installed in the IPython installation extension directory and
   therefore becomes available to all users of the machine automatically.
   As a profile, it must be installed in each user's ipython configuration directory
   (in linux, is usualy $HOME/.ipython).
   Note that the spock profile is a very short (3 lines of code) python file.
   See PyTango :ref:`build-install` for more information on how to install spock
   as an extension or a profile.

.. _IPython: http://ipython.scipy.org/
.. _Tango: http://www.tango-controls.org/