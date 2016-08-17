.. currentmodule:: tango

.. _pytango-faq:

FAQ
===

Answers to general Tango questions can be found in the
`general tango tutorial <http://www.tango-controls.org/resources/tutorials>`_.

Please also check the `general tango how to <http://www.tango-controls.org/resources/howto/>`_.

**I got a libbost_python error when I try to import PyTango module**

doing:
    >>> import PyTango
    ImportError: libboost_python-gcc43-mt-1_38.so.1.38.0: cannot open shared object file: No such file or directory

You must check that you have the correct boost python installed on your computer.
To see which boost python file PyTango needs type::

    $ ldd /usr/lib/python2.5/site-packages/PyTango/_PyTango.so
    linux-vdso.so.1 =>  (0x00007fff48bfe000)
    libtango.so.7 => /home/homer/local/lib/libtango.so.7 (0x00007f393fabb000)
    liblog4tango.so.4 => /home/homer/local/lib/liblog4tango.so.4 (0x00007f393f8a0000)
    **libboost_python-gcc43-mt-1_38.so.1.38.0 => not found**
    libpthread.so.0 => /lib/libpthread.so.0 (0x00007f393f65e000)
    librt.so.1 => /lib/librt.so.1 (0x00007f393f455000)
    libdl.so.2 => /lib/libdl.so.2 (0x00007f393f251000)
    libomniORB4.so.1 => /usr/local/lib/libomniORB4.so.1 (0x00007f393ee99000)
    libomniDynamic4.so.1 => /usr/local/lib/libomniDynamic4.so.1 (0x00007f393e997000)
    libomnithread.so.3 => /usr/local/lib/libomnithread.so.3 (0x00007f393e790000)
    libCOS4.so.1 => /usr/local/lib/libCOS4.so.1 (0x00007f393e359000)
    libgcc_s.so.1 => /lib/libgcc_s.so.1 (0x00007f393e140000)
    libc.so.6 => /lib/libc.so.6 (0x00007f393ddce000)
    libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f393dac1000)
    libm.so.6 => /lib/libm.so.6 (0x00007f393d83b000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f3940a4c000)


