.. currentmodule:: PyTango

.. highlight:: python
   :linenothreshold: 3
   

Mixing Tango classes (Python and C++) in a Python Tango device server
---------------------------------------------------------------------

Within the same python interpreter, it is possible to mix several Tango classes. 
Here is an example of the main function of a device server with two Tango classes
called IRMiror and PLC::

    import PyTango
    import sys

    if __name__ == '__main__':
        util = PyTango.Util(sys.argv)
        util.add_class(PLCClass, PLC, 'PLC')
        util.add_class(IRMirrorClass, IRMirror, 'IRMirror')
        
        U = PyTango.Util.instance()
        U.server_init()
        U.server_run()

:Line 6: The Tango class PLC is registered in the device server
:Line 7: The Tango class IRMirror is registered in the device server

It is also possible to add C++ Tango class in a Python device server as soon as:
    1. The Tango class is in a shared library
    2. It exist a C function to create the Tango class

For a Tango class called MyTgClass, the shared library has to be called 
MyTgClass.so and has to be in a directory listed in the LD_LIBRARY_PATH 
environment variable. The C function creating the Tango class has to be called 
_create_MyTgClass_class() and has to take one parameter of type "char \*" which 
is the Tango class name. Here is an example of the main function of the same 
device server than before but with one C++ Tango class called SerialLine::

    import PyTango
    import sys
    
    if __name__ == '__main__':
        py = PyTango.Util(sys.argv)
        util.add_class('SerialLine', 'SerialLine', language="c++")
        util.add_class(PLCClass, PLC, 'PLC')
        util.add_class(IRMirrorClass, IRMirror, 'IRMirror')
        
        U = PyTango.Util.instance()
        U.server_init()
        U.server_run()

:Line 6: The C++ class is registered in the device server
:Line 7 and 8: The two Python classes are registered in the device server

Server API
----------

.. toctree::
    :maxdepth: 2
    
    server
    device
    device_class
    logging
    attribute
    util
