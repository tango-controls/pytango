.. currentmodule:: tango

.. _pytango-faq:

FAQ
===

Answers to general Tango questions can be found in the
`general tango tutorial <http://www.tango-controls.org/resources/tutorials>`_.

Please also check the `general tango how to <http://www.tango-controls.org/resources/howto/>`_.


**How can I report an issue?**

Bug reports are very valuable for the community.

Please open a new issue on the GitHub issues_ page.


**How can I contribute to PyTango and the documentation?**

Contribution are always welcome!

You can open pull requests on the GitHub PRs_ page.


**I got a libbost_python error when I try to import tango module...**

For instance::

    >>> import tango
    ImportError: libboost_python.so.1.53.0: cannot open shared object file: No such file or directory

You must check that you have the correct boost python installed on your computer.
To see which boost python file PyTango needs, type:

.. sourcecode:: console

    $ ldd /usr/lib64/python2.7/site-packages/tango/_tango.so
        linux-vdso.so.1 =>  (0x00007ffea7562000)
        libtango.so.9 => /lib64/libtango.so.9 (0x00007fac04011000)
        libomniORB4.so.1 => /lib64/libomniORB4.so.1 (0x00007fac03c62000)
        libboost_python.so.1.53.0 => not found
        [...]


**I have more questions, where can I ask?**

The `Tango forum <http://www.tango-controls.org/community/forum>`_ is a good place to get some support.
Meet us in the `Python section <http://www.tango-controls.org/community/forum/c/development/python/>`_.
